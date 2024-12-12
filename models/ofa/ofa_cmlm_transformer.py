# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import random
from typing import Any, Dict, List, Optional, Tuple
from fairseq import utils

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import (
    NATransformerModel, NATransformerDecoder, NATransformerEncoder,
    FairseqNATEncoder, FairseqNATDecoder)
from fairseq.utils import new_arange, softmax, log_softmax
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.iterative_refinement_generator import DecoderOut

from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    SinusoidalPositionalEmbedding,
    GradMultiply
)
from torch import Tensor
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.distributed import fsdp_wrap

from .unify_transformer_layer import TransformerEncoderLayer, TransformerDecoderLayer
from .resnet import ResNet
from .frozen_bn import FrozenBatchNorm2d
from .ofa_posterior_transformer import PosteriorTransformerEncoder, PosteriorTransformerDecoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def BatchNorm2d(out_chan, momentum=0.1, eps=1e-3):
    return nn.SyncBatchNorm.convert_sync_batchnorm(
        nn.BatchNorm2d(out_chan, momentum=momentum, eps=eps)
    )


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def _skeptical_unmasking(output_scores, output_masks, p=None, boundary_len=None):
    sorted_index = output_scores.sort(-1)[1]
    if boundary_len is None:
        boundary_len = (
                (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
        ).long()
    skeptical_mask = new_arange(output_masks).type_as(boundary_len) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m


def make_token_bucket_position(bucket_size, max_position=DEFAULT_MAX_SOURCE_POSITIONS):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))
    log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid
    log_pos = log_pos.int()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos * sign).long()
    return bucket_pos + bucket_size - 1


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class PromptEncoder(torch.nn.Module):
    r"""
    Prompt encoder to generate prompts, including prompt, prefix, instance and instruction
    """

    def __init__(
            self,
            type,
            length,
            projection,
            embed_dim,
            proj_dim,
            layers,
            vocab_size):
        super().__init__()
        self.prefix_projection = projection

        if type == "prefix":
            layers = layers
            prompt_vocab_size = length

        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(prompt_vocab_size, embed_dim)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, proj_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_dim, layers * 2 * embed_dim)
            )
        else:
            if type == "prefix":
                self.embedding = torch.nn.Embedding(
                    prompt_vocab_size, layers * 2 * embed_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


@register_model("ofa_cmlm_transformer")
class OFACMLMNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        # self.no_predict_length = getattr(args, "src_upsample", 1) > 1
        self.no_predict_length = True
        self.src_upsample = getattr(args, "src_upsample", 1)
        self.axe_eps_idx = encoder.dictionary.axe_eps_idx
        self.latent_dim = getattr(args, "latent_dim", 0)
        self.max_steps = getattr(args, "max_updates", 300000)
        self.ctc_bs_alpha = getattr(args, "ctc_bs_alpha", 0.)
        self.ctc_bs_beta = getattr(args, "ctc_bs_beta", 0.)
        self.use_ctc_beamsearch = getattr(args, "use_ctc_beamsearch", False)
        # self.ctc_bs_beam = getattr(args, "ctc_bs_beam", 5)
        # self.ctc_bs_beam = 5
        # self.ctc_bs_beam = 20
        # self.ctc_bs_beam = 100
        self.ctc_bs_lm_path = getattr(args, "ctc_bs_lm_path")

        if self.latent_dim > 0:
            args_copy = copy.deepcopy(args)
            args_copy.encoder_layers = getattr(args, "posterior_layers", 3)
            args_copy.decoder_layers = getattr(args, "posterior_layers", 3)
            args_copy.src_upsample = 1
            args_copy.length_loss_factor_tgt_len = False
            args_copy.use_first_token_pred_len = False
            args_copy.add_first_token_encoder = False
            args_copy.src_embedding_copy = None
            self.q_encoder_y = PosteriorTransformerEncoder(args_copy, self.decoder.dictionary, self.decoder.embed_tokens)
            self.q_encoder_xy = PosteriorTransformerDecoder(args_copy, self.encoder.dictionary, self.encoder.embed_tokens)
            self.q_encoder_xy.embed_length = None
            self.prob_esitmator = torch.nn.Linear(args.encoder_embed_dim, self.latent_dim * 2)
            self.latent_map = torch.nn.Linear(self.latent_dim, args.encoder_embed_dim)

            from fairseq.models.nat.levenshtein_utils import VAEBottleneck
            self.bottleneck = VAEBottleneck(
                args.encoder_embed_dim,
                z_size=self.latent_dim,
                freebits=getattr(args, "freebits", 1.0))

        if self.use_ctc_beamsearch:
            from ctcdecode import CTCBeamDecoder
            self.ctcdecoder = CTCBeamDecoder(
                self.tgt_dict.symbols,
                model_path=self.ctc_bs_lm_path,
                alpha=self.ctc_bs_alpha,
                beta=self.ctc_bs_beta,
                cutoff_top_n=40,
                cutoff_prob=1,
                beam_width=self.ctc_bs_beam,
                num_processes=20,
                blank_id=self.axe_eps_idx,
                log_probs_input=True
            )

    def set_num_updates(self, updates):
        self._updates = updates
        super().set_num_updates(updates)

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument("--src-upsample", type=float,
                            help="if larger than 1, no length-prediction is used. upsample the source")
        # parser.add_argument("--src-upsample-bias", type=float, default=None)
        parser.add_argument("--dynamic-upsample", action='store_true')
        parser.add_argument("--num-queries", type=int, default=100)
        parser.add_argument("--decoder-input-type", default='query')
        parser.add_argument("--replace-target-embed", action='store_true')
        parser.add_argument('--glat-sampling-ratio', type=float, default=0.0,
                            help='if larger than 0, use GLAT sampling.')
        parser.add_argument('--glat-min-ratio', type=float, default=None)
        parser.add_argument('--glat-use-valid', action='store_true')
        parser.add_argument('--poisson-mask', action='store_true')
        parser.add_argument("--ctc-bs-alpha", type=float, default=0.0)
        parser.add_argument("--ctc-bs-beam", type=int, default=20)
        parser.add_argument("--ctc-bs-beta", type=float, default=0.0)
        parser.add_argument('--use-ctc-beamsearch', action='store_true')
        parser.add_argument('--ctc-bs-lm-path', default=None)
        parser.add_argument('--glat-edit-distance', action='store_true')
        parser.add_argument('--glat-random-distance', action='store_true')
        parser.add_argument('--two-passes-distill', action='store_true')
        parser.add_argument('--two-passes-kl', action='store_true')

        parser.add_argument('--latent-dim', type=int, default=0)
        parser.add_argument('--posterior-layers', type=int, default=3)
        parser.add_argument('--simple-prior', action='store_true')
        parser.add_argument('--ar-prior', action='store_true')
        parser.add_argument('--freebits', type=float, default=1.0)
        parser.add_argument('--bitfit', default=False, action='store_true',
                            help='use bitfit in the transformer')



        # Copied from unify_transformer.py
        """Add model-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument('--activation-fn',
        #                     choices=utils.get_available_activation_fns(),
        #                     help='activation function to use')
        # parser.add_argument('--dropout', type=float, metavar='D',
        #                     help='dropout probability')
        # parser.add_argument('--attention-dropout', type=float, metavar='D',
        #                     help='dropout probability for attention weights')
        # parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
        #                     help='dropout probability after activation in FFN.')
        # parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
        #                     help='path to pre-trained encoder embedding')
        # parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
        #                     help='encoder embedding dimension')
        # parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
        #                     help='encoder embedding dimension for FFN')
        # parser.add_argument('--encoder-layers', type=int, metavar='N',
        #                     help='num encoder layers')
        # parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
        #                     help='num encoder attention heads')
        # parser.add_argument('--encoder-normalize-before', action='store_true',
        #                     help='apply layernorm before each encoder block')
        # parser.add_argument('--encoder-learned-pos', action='store_true',
        #                     help='use learned positional embeddings in the encoder')
        # parser.add_argument('--bitfit', default=False, action='store_true',
        #                     help='use bitfit in the transformer')
        # parser.add_argument('--freeze-encoder', action='store_true',
        #                     help='freeze the parameters in the encoder')
        #
        #
        # parser.add_argument('--adapter', action='store_true',
        #                     help='use adapter in the model')
        # parser.add_argument('--adapter-dim', type=int, metavar='N',
        #                     help='adapter-down-dim')
        #
        # parser.add_argument('--encoder-prompt', action='store_true',
        #                     help='use prompt tuning in the encoder')
        # parser.add_argument('--encoder-prompt-type', type=str, metavar='STR',
        #                     choices=['prefix'],
        #                     help='the type of prompt tuning')
        # parser.add_argument('--encoder-prompt-projection', action='store_true',
        #                     help='use prompt projection')
        # parser.add_argument('--encoder-prompt-length', type=int, metavar='N',
        #                     help='use prompt tuning in the decoder')
        # parser.add_argument('--encoder-prompt-dim', type=int, metavar='N',
        #                     help='encoder prompt dimension if use encoder prompt projection')
        # parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
        #                     help='path to pre-trained decoder embedding')
        # parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
        #                     help='decoder embedding dimension')
        # parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
        #                     help='decoder embedding dimension for FFN')
        # parser.add_argument('--decoder-layers', type=int, metavar='N',
        #                     help='num decoder layers')
        # parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
        #                     help='num decoder attention heads')
        # parser.add_argument('--decoder-learned-pos', action='store_true',
        #                     help='use learned positional embeddings in the decoder')
        # parser.add_argument('--decoder-normalize-before', action='store_true',
        #                     help='apply layernorm before each decoder block')
        # parser.add_argument('--decoder-output-dim', type=int, metavar='N',
        #                     help='decoder output dimension (extra linear layer '
        #                          'if different from decoder embed dim')
        # parser.add_argument('--freeze-decoder', action='store_true',
        #                     help='freeze the parameters in the decoder')
        #
        # parser.add_argument('--decoder-prompt', action='store_true',
        #                     help='use prompt tuning in the decoder')
        # parser.add_argument('--decoder-prompt-type', type=str, metavar='STR',
        #                     choices=['prefix'],
        #                     help='the type of prompt tuning')
        # parser.add_argument('--decoder-prompt-length', type=int, metavar='N',
        #                     help='use prompt tuning in the decoder')
        # parser.add_argument('--decoder-prompt-projection', action='store_true',
        #                     help='use prompt projection')
        # parser.add_argument('--decoder-prompt-dim', type=int, metavar='N',
        #                     help='decoder prompt dimension if use decoder prompt projection')
        # parser.add_argument('--share-decoder-input-output-embed', action='store_true',
        #                     help='share decoder input and output embeddings')
        # parser.add_argument('--share-all-embeddings', action='store_true',
        #                     help='share encoder, decoder and output embeddings'
        #                          ' (requires shared dictionary and embed dim)')
        # parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
        #                     help='if set, disables positional embeddings (outside self attention)')
        # parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
        #                     help='comma separated list of adaptive softmax cutoff points. '
        #                          'Must be used with adaptive_loss criterion'),
        # parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
        #                     help='sets adaptive softmax dropout for the tail projections')
        # parser.add_argument('--layernorm-embedding', action='store_true',
        #                     help='add layernorm to embedding')
        # parser.add_argument('--no-scale-embedding', action='store_true',
        #                     help='if True, dont scale embeddings')
        # parser.add_argument('--checkpoint-activations', action='store_true',
        #                     help='checkpoint activations at each layer, which saves GPU '
        #                          'memory usage at the cost of some additional compute')
        # parser.add_argument('--offload-activations', action='store_true',
        #                     help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        # parser.add_argument('--no-cross-attention', default=False, action='store_true',
        #                     help='do not perform cross-attention')
        # parser.add_argument('--cross-self-attention', default=False, action='store_true',
        #                     help='perform cross+self-attention')
        # # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        # parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
        #                     help='LayerDrop probability for encoder')
        # parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
        #                     help='LayerDrop probability for decoder')
        # parser.add_argument('--encoder-layers-to-keep', default=None,
        #                     help='which layers to *keep* when pruning as a comma-separated list')
        # parser.add_argument('--decoder-layers-to-keep', default=None,
        #                     help='which layers to *keep* when pruning as a comma-separated list')
        # # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        # parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
        #                     help='iterative PQ quantization noise at training time')
        # parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
        #                     help='block size of quantization noise at training time')
        # parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
        #                     help='scalar quantization noise and scalar quantization at training time')
        # # args for Fully Sharded Data Parallel (FSDP) training
        # parser.add_argument(
        #     '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
        #     help=(
        #         'minimum number of params for a layer to be wrapped with FSDP() when '
        #         'training with --ddp-backend=fully_sharded. Smaller values will '
        #         'improve memory efficiency, but may make torch.distributed '
        #         'communication less efficient due to smaller input sizes. This option '
        #         'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
        #         '--offload-activations are passed.'
        #     )
        # )
        #
        parser.add_argument('--resnet-drop-path-rate', type=float,
                            help='resnet drop path rate')
        parser.add_argument('--encoder-drop-path-rate', type=float,
                            help='encoder drop path rate')
        parser.add_argument('--decoder-drop-path-rate', type=float,
                            help='encoder drop path rate')
        #
        # parser.add_argument('--token-bucket-size', type=int,
        #                     help='token bucket size')
        # parser.add_argument('--image-bucket-size', type=int,
        #                     help='image bucket size')
        #
        # parser.add_argument('--attn-scale-factor', type=float,
        #                     help='attention scale factor')
        # parser.add_argument('--freeze-resnet', action='store_true',
        #                     help='freeze resnet')
        parser.add_argument('--freeze-encoder-embedding', action='store_true',
                            help='freeze encoder token embedding')
        parser.add_argument('--freeze-decoder-embedding', action='store_true',
                            help='freeze decoder token embedding')
        parser.add_argument('--add-type-embedding', action='store_true',
                            help='add source/region/patch type embedding')
        # parser.add_argument('--interpolate-position', action='store_true',
        #                     help='interpolate position')
        #
        # parser.add_argument('--resnet-type', choices=['resnet50', 'resnet101', 'resnet152'],
        #                     help='resnet type')
        # parser.add_argument('--resnet-model-path', type=str, metavar='STR',
        #                     help='path to load resnet')
        # parser.add_argument('--code-image-size', type=int,
        #                     help='code image size')
        parser.add_argument('--patch-layernorm-embedding', action='store_true',
                            help='add layernorm to patch embedding')
        parser.add_argument('--code-layernorm-embedding', action='store_true',
                            help='add layernorm to code embedding')
        parser.add_argument('--entangle-position-embedding', action='store_true',
                            help='entangle position embedding')
        parser.add_argument('--disable-entangle', action='store_true',
                            help='disable entangle')
        parser.add_argument('--sync-bn', action='store_true',
                            help='sync batchnorm')
        #
        parser.add_argument('--scale-attn', action='store_true',
                            help='scale attn')
        parser.add_argument('--scale-fc', action='store_true',
                            help='scale fc')
        parser.add_argument('--scale-heads', action='store_true',
                            help='scale heads')
        parser.add_argument('--scale-resids', action='store_true',
                            help='scale resids')
        # fmt: on


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        cmlm_base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "freeze_encoder_embedding", False) or getattr(
                args, "encoder_prompt", False) or getattr(args, "decoder_prompt", False) or getattr(args, "adapter", False):
            encoder_embed_tokens.weight.requires_grad = False
        if getattr(args, "freeze_decoder_embedding", False) or getattr(
                args, "encoder_prompt", False) or getattr(args, "decoder_prompt", False) or getattr(args, "adapter", False):
            decoder_embed_tokens.weight.requires_grad = False
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if getattr(args, "encoder_prompt", False) or getattr(
                args, "decoder_prompt", False):
            encoder.requires_grad_(False)
            decoder.requires_grad_(False)
            if getattr(args, "encoder_prompt", False):
                encoder.encoder_prompt_encoder.requires_grad_(True)
            if getattr(args, "decoder_prompt", False):
                decoder.decoder_prompt_encoder.requires_grad_(True)
            if getattr(args, "adapter", False):
                for idx, layer in enumerate(encoder.layers):
                    layer.adapter.requires_grad_(True)
                for idx, layer in enumerate(decoder.layers):
                    layer.adapter.requires_grad_(True)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = CMLMNATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = CMLMNATransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
            **kwargs
    ):
        all_results = dict()

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        encoder_out = encoder_out._replace(
            encoder_embedding=encoder_out.encoder_out[1:].transpose(0, 1) \
                if (not self.no_predict_length) and getattr(self.args, "add_first_token_encoder", False) else \
                encoder_out.encoder_out.transpose(0, 1))

        # VAE part
        if self.latent_dim > 0:
            prior_prob = self.prob_esitmator(encoder_out.encoder_embedding)

            sampled_z, q_prob = self.bottleneck(
                self.q_encoder_xy.extract_features(src_tokens,
                                                   encoder_out=self.q_encoder_y(tgt_tokens, src_lengths=tgt_tokens.ne(self.pad).sum(-1)))[0], sampling=True)
            full_z = self.latent_map(sampled_z)
            encoder_out = encoder_out._replace(encoder_embedding=
                                               encoder_out.encoder_embedding + full_z)  # For simple, add Z on encoder out..
            kl, budget = self.bottleneck.compute_final_loss(q_prob, prior_prob, sampled_z)
            kl_loss = kl[src_tokens.ne(self.pad)].mean()
            all_results["kl_loss"] = {
                "loss": kl_loss, 'factor': 1.0,
            }
            all_results["add_logs"] = {'kl_budget': budget}

        # GLAT stuff
        if self.training and (getattr(self.args, "glat_sampling_ratio", 0) > 0):

            # go through pass-1 NAT
            if not getattr(self.args, "glat_use_valid", False):
                with torch.no_grad():
                    nat_word_ins_out, extra = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens if not self.no_predict_length else None,
                        encoder_out=encoder_out)
            else:
                self.eval()  # disable all dropout
                with torch.no_grad():
                    glat_encoder_out = self.forward_encoder([src_tokens, src_lengths])
                    prev_decoder_out = self.initialize_output_tokens(glat_encoder_out, src_tokens)
                    if prev_decoder_out.attn is not None:
                        full_z = prev_decoder_out.attn
                        glat_encoder_out = glat_encoder_out._replace(encoder_embedding=
                                                                     glat_encoder_out.encoder_embedding + full_z)  # For simple, add Z on encoder out..
                    nat_word_ins_out, extra = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens if not self.no_predict_length else None,
                        encoder_out=glat_encoder_out,
                    )
                self.train()  # back to normal

            # apply GLAT?
            if (getattr(self.args, "glat_sampling_ratio", 0) > 0):
                # compute hamming distance
                f_ratio = self.args.glat_sampling_ratio
                if getattr(self.args, "glat_min_ratio", None) is not None:
                    f_min_ratio = self.args.glat_min_ratio
                    f_ratio = f_ratio - (f_ratio - f_min_ratio) * (self._updates / float(self.max_steps))

                output_scores, output_tokens = nat_word_ins_out.max(-1)

                if self.no_predict_length:
                    from seqdist import ctc
                    alignment = ctc.viterbi_alignments(
                        log_softmax(nat_word_ins_out, dim=-1).transpose(0, 1),
                        tgt_tokens,
                        (~extra['padding_mask']).sum(-1),
                        tgt_tokens.ne(self.pad).sum(-1))

                    inter_tgt_tokens = ctc.interleave_blanks(tgt_tokens, self.axe_eps_idx)
                    aligned_tokens = torch.einsum('lbd,bd->bl', alignment, inter_tgt_tokens.float()).long()
                else:
                    inter_tgt_tokens = None
                    aligned_tokens = tgt_tokens.clone()

                # edit distance
                if getattr(self.args, "glat_edit_distance", False):
                    assert not getattr(self.args, "glat_random_distance", False)
                    from fairseq.models.nat.levenshtein_utils import _collapse, _get_edit_distance
                    out_tokens, _ = _collapse(output_tokens, output_scores, self.pad, self.axe_eps_idx)
                    edit_dis = _get_edit_distance(out_tokens, tgt_tokens, self.pad)
                    wer = edit_dis.type_as(output_scores) / tgt_tokens.ne(self.pad).sum(-1).type_as(output_scores)
                    mask_lens = f_ratio * wer * aligned_tokens.ne(self.pad).sum(-1).type_as(output_scores)
                # random
                elif getattr(self.args, "glat_random_distance", False):
                    assert not getattr(self.args, "glat_edit_distance", False)
                    bsz = aligned_tokens.size(0)
                    random_score = aligned_tokens.new_zeros((bsz,)).float().uniform_()
                    mask_lens = random_score * aligned_tokens.ne(self.pad).sum(1)
                # hamming
                else:
                    hamming_dis = output_tokens.ne(aligned_tokens).sum(-1)
                    mask_lens = (f_ratio * hamming_dis.type_as(output_scores))

                decoder_scores = output_scores.uniform_().masked_fill_(
                    aligned_tokens.eq(self.pad) | aligned_tokens.eq(self.bos) | aligned_tokens.eq(self.eos), 2.0)
                if getattr(self.args, "poisson_mask", False):
                    mask_lens = torch.poisson(mask_lens)
                else:
                    mask_lens = mask_lens.long()
                mask_lens = (aligned_tokens.ne(self.pad).sum(1) - mask_lens).clamp(min=1)
                glat_mask_ratio = mask_lens.sum() / float(aligned_tokens.ne(self.pad).sum())
                glat_mask = _skeptical_unmasking(decoder_scores, aligned_tokens.ne(self.pad),
                                                 boundary_len=mask_lens[:, None])
                prev_output_tokens = aligned_tokens.masked_fill(glat_mask, self.unk)

                if "add_logs" not in all_results:
                    all_results["add_logs"] = {'glat_mask': glat_mask_ratio, 'f_ratio': f_ratio}
                else:
                    all_results["add_logs"]['glat_mask'] = glat_mask_ratio
                    all_results["add_logs"]['f_ratio'] = f_ratio

        # length prediction
        if not self.no_predict_length:
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            all_results["length"] = {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.get_length_loss_factor(length_out, tgt_tokens.ne(self.pad))
            }
            prev_output_tokens = (prev_output_tokens,)

        # decoding
        word_ins_out, extra = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)

        factor = 1.

        if not self.no_predict_length:  # use the ground-truth length
            word_ins_mask = prev_output_tokens[0].eq(self.unk)
            word_ins_out = word_ins_out[:, :word_ins_mask.size(1)]
            out_mask = prev_output_tokens[0].ne(self.pad)
        else:
            word_ins_mask = None
            out_mask = ~extra['padding_mask']

        all_results["word_ins"] = {
            "out": word_ins_out, "tgt": tgt_tokens,
            "mask": word_ins_mask, "ls": self.args.label_smoothing,
            "nll_loss": True, "out_mask": out_mask, 'factor': factor,
        }
        if self.decoder.custom_loss is not None:
            all_results['decoder'] = self.decoder.custom_loss
        return all_results

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None, length_tgt=None):
        # no length prediction. output dummy initialization.
        if self.latent_dim > 0:
            prior_prob = self.prob_esitmator(encoder_out['encoder_embedding'][0])
            mean_vector = prior_prob[:, :, :self.latent_dim]
            full_z = self.latent_map(mean_vector)
        else:
            full_z = None

        if not self.no_predict_length:
            decoder_out = super().initialize_output_tokens(encoder_out, src_tokens, tgt_tokens, length_tgt)
            if full_z is None:
                return decoder_out
            return decoder_out._replace(attn=full_z)
        return DecoderOut(output_tokens=src_tokens, output_scores=None,
                          attn=full_z, step=0, max_step=0, history=None)  # HACK: use attn for now..

    def regenerate_length_beam(self, decoder_out, beam_size, encoder_out):
        if not self.no_predict_length:
            return super().regenerate_length_beam(decoder_out, beam_size, encoder_out)

        b, l, d = decoder_out.attn.size()
        if self.latent_dim > 0:
            prior_prob = self.prob_esitmator(encoder_out.encoder_embedding)
            mean_vector = prior_prob[:, :, :self.latent_dim]
            var = 0.1 * F.softplus(prior_prob[:, :, self.latent_dim:])
            full_z = self.latent_map(mean_vector + var * torch.zeros_like(mean_vector).normal_())
        else:
            full_z = None

        return decoder_out._replace(
            output_tokens=decoder_out.output_tokens.unsqueeze(1).expand(b, beam_size, l).reshape(b * beam_size, l),
            attn=full_z,
        )
        # from fairseq import pdb; pdb.set_trace()

    def forward_encoder(self, encoder_inputs):
        encoder_out = self.encoder(*encoder_inputs)
        encoder_out = encoder_out._replace(
            encoder_embedding=encoder_out.encoder_out[1:].transpose(0, 1) \
                if (not self.no_predict_length) and getattr(self.args, "add_first_token_encoder", False) else \
                encoder_out.encoder_out.transpose(0, 1))
        return encoder_out

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        if decoder_out.attn is not None:
            full_z = decoder_out.attn
            encoder_out = encoder_out._replace(encoder_embedding=
                                               encoder_out.encoder_embedding + full_z)  # For simple, add Z on encoder out..

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        decoder_output, extra = self.decoder(
            normalize=True,
            prev_output_tokens=(output_tokens,),
            encoder_out=encoder_out,
        )
        if self.use_ctc_beamsearch:
            topk = self.ctc_bs_beam  # * 2
            decoder_topk_scores, decoder_topk_index = decoder_output.topk(k=topk, dim=-1)

            # HACK: CTC beam-search requires the probability of blank, we put it in the end
            decoder_topk_scores = torch.cat(
                [decoder_topk_scores, decoder_output[..., self.axe_eps_idx:self.axe_eps_idx + 1]], -1)
            decoder_topk_index = torch.cat([decoder_topk_index,
                                            decoder_topk_index.new_ones(*decoder_topk_index.size()[:-1],
                                                                        1) * self.axe_eps_idx], -1)
            if decoder_topk_index.size(0) > 1:
                decoder_topk_scores[..., 0].masked_fill_(extra["padding_mask"], 0.)
                decoder_topk_scores[..., -1].masked_fill_(extra["padding_mask"], 0.)
                decoder_topk_scores[..., 1:-1].masked_fill_(extra["padding_mask"].unsqueeze(-1), float("-Inf"))
                decoder_topk_index[..., 0].masked_fill_(extra["padding_mask"], self.axe_eps_idx)

            beam_results, beam_scores, timesteps, out_lens = self.ctcdecoder.decode(decoder_topk_scores,
                                                                                    decoder_topk_index)
            # from fairseq import pdb; pdb.set_trace()
            _scores, _tokens = beam_scores[:, 0].to(decoder_output.device), beam_results[:, 0].to(
                decoder_output.device).long()
            out_lens = out_lens.to(decoder_output.device).type_as(_tokens)
            _scores = _scores[:, None].expand_as(_tokens)
            extra["padding_mask"] = new_arange(_tokens, *_tokens.size()) >= out_lens[:, :1]
        else:
            _scores, _tokens = decoder_output.max(-1)
        if not self.no_predict_length:
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
        else:
            output_tokens = _tokens.masked_fill(extra["padding_mask"], self.pad)
            output_scores = _scores.masked_fill(extra["padding_mask"], 0.)

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            assert not self.no_predict_length, "mask-predict only supports length prediction."
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )


class CMLMNATransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        # breakpoint()
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.src_upsample = getattr(args, "src_upsample", 1)
        # self.upsample_bias = getattr(args, "src_upsample_bias", None)
        self.dynamic = getattr(args, "dynamic_upsample", False)
        self.num_queries = getattr(args, "num_queries", 100)
        self.replace_target_embed = getattr(args, "replace_target_embed", False)
        self.decoder_input_type = getattr(args, "decoder_input_type", 'query')
        if self.src_upsample > 1:
            if not self.dynamic:
                # assert self.upsample_bias is None, "only support no bias"
                self.upsampler = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * self.src_upsample)
        self.embed_length = None  # do not use length prediction
        self.custom_loss = None

        # Copied from unify transformer
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.num_attention_heads = args.decoder_attention_heads

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.window_size = args.code_image_size // 8
        self.embed_positions = Embedding(args.max_target_positions + 2, embed_dim)
        self.embed_image_positions = Embedding(args.image_bucket_size ** 2 + 1, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        self.image_pos_ln = LayerNorm(embed_dim)
        self.pos_scaling = float(embed_dim / self.num_attention_heads * args.attn_scale_factor) ** -0.5
        self.self_pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.self_pos_k_linear = nn.Linear(embed_dim, embed_dim)
        self.cross_pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.cross_pos_k_linear = nn.Linear(embed_dim, embed_dim)

        if getattr(args, "code_layernorm_embedding", False):
            self.code_layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.code_layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, args.decoder_drop_path_rate, args.decoder_layers)]
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn, drop_path_rate=dpr[i])
                for i in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

        token_bucket_size = args.token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        )

        image_bucket_size = args.image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        image_position_idx = torch.arange(self.window_size).unsqueeze(0).expand(self.window_size, self.window_size) + \
                             torch.arange(self.window_size).unsqueeze(1) * image_bucket_size + 1
        image_position_idx = torch.cat([torch.tensor([0]), image_position_idx.view(-1)])
        image_position_idx = torch.cat([image_position_idx, torch.tensor([1024] * 769)])
        self.image_rel_pos_table_list = nn.ModuleList(
            [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        )

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.register_buffer("image_position_idx", image_position_idx)
        self.entangle_position_embedding = args.entangle_position_embedding

        if self.decoder_input_type == 'query':
            self.query_embed = nn.Embedding(self.num_queries, embed_dim)
        # elif self.decoder_input_type == 'encoder_out':


    def build_decoder_layer(self, args, no_encoder_attn=False, drop_path_rate=0.0):
        layer = TransformerDecoderLayer(args, no_encoder_attn, drop_path_rate= \
            drop_path_rate, use_adapter=getattr(args, "adapter", False), adapter_dim=getattr(args, "adapter_dim", 200))
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def dynamic_upsample(self, x, mask):
        l = x.new_ones(x.size(1), x.size(0)) * self.src_upsample
        l = l.masked_fill(mask, 0)
        e = torch.cumsum(l, 1)
        c = e - l / 2
        t = e[:, -1].ceil().long()
        t = new_arange(t, t.max())[None, :].expand(l.size(0), -1)  # B x L2
        t_mask = t >= e[:, -1:]  # target padding mask
        w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3
        w = w.float()
        w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
        t_w = F.softmax(w, dim=1)  # B x L x L2
        t_x = torch.einsum('bst,sbd->btd', t_w.type_as(x), x)
        return t_x, t_mask, w

    def get_rel_pos_bias(self, seq_len, idx):
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_image_rel_pos_bias(self, seq_len, idx):
        image_position_idx = self.image_position_idx[:seq_len]
        rp_bucket = self.image_rp_bucket[image_position_idx][:, image_position_idx]
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(2, 0, 1)
        return values

    def get_pos_info(self, batch_size, tgt_len, tgt_pos_embed, src_pos_embed=None, use_image=False):
        tgt_pos_embed = self.image_pos_ln(tgt_pos_embed) if use_image else self.pos_ln(tgt_pos_embed)
        if src_pos_embed is not None:
            src_len = src_pos_embed.size(1)
            pos_q = self.cross_pos_q_linear(tgt_pos_embed).view(
                batch_size, tgt_len, self.num_attention_heads, -1
            ).transpose(1, 2) * self.pos_scaling
            pos_k = self.cross_pos_k_linear(src_pos_embed).view(
                batch_size, src_len, self.num_attention_heads, -1
            ).transpose(1, 2)
        else:
            src_len = tgt_pos_embed.size(1)
            pos_q = self.self_pos_q_linear(tgt_pos_embed).view(
                batch_size, tgt_len, self.num_attention_heads, -1
            ).transpose(1, 2) * self.pos_scaling
            pos_k = self.self_pos_k_linear(tgt_pos_embed).view(
                batch_size, src_len, self.num_attention_heads, -1
            ).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))
        return abs_pos_bias

    def forward_embedding(self, prev_output_tokens, embedding_copy=None, encoder_out=None, code_masks=None):

        B, L, D = encoder_out['encoder_embedding'][0].size()


        if self.decoder_input_type == 'query':
            x = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).clone()
        else:
            x = encoder_out['encoder_embedding'][0][:, :1024, :]
            
        mask = torch.zeros(x.shape[0], x.shape[1]).bool().to(x.device)

        if self.replace_target_embed and self.training and prev_output_tokens is not None:
            assert prev_output_tokens.size(1) == x.size(1), "length must match"
            tgt_embed, _ = super().forward_embedding(prev_output_tokens)
            tgt_mask = prev_output_tokens.ne(self.unk).unsqueeze(-1).expand_as(x)
            x = x.masked_scatter(tgt_mask, tgt_embed[tgt_mask])

        prompt_tokens = None
        prompt_padding_mask = None
        prompt_kv_list = None
        if self.args.decoder_prompt:
            bsz, seq_len = prev_output_tokens.shape[0], prev_output_tokens.shape[1]
            if self.args.decoder_prompt_type in ("prefix"):
                prompt_tokens = torch.arange(
                    0, self.args.decoder_prompt_length).to(
                    prev_output_tokens.device)
                prompt_tokens = prompt_tokens.unsqueeze(0).expand(bsz, -1)
                prompt_padding_mask = torch.zeros_like(prompt_tokens).to(prompt_tokens.device)
            prompt_kv_list = self.get_decoder_prompt(prompt_tokens)
        # bs, slen = prev_output_tokens.size()
        # if alignment_layer is None:
        #     alignment_layer = self.num_layers - 1

        # enc: Optional[Tensor] = None
        # padding_mask: Optional[Tensor] = None
        # if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
        #     enc = encoder_out["encoder_out"][0]
        #     assert (
        #             enc.size()[1] == bs
        #     ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        # if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
        #     padding_mask = encoder_out["encoder_padding_mask"][0]

        bsz, tgt_len = x.shape[0], x.shape[1]
        # token_position_idx = utils.new_arange(prev_output_tokens)
        token_position_idx = utils.new_arange(x, x.shape[0], x.shape[1])
        tgt_pos_embed = self.embed_positions(token_position_idx)
        if code_masks is not None and torch.any(code_masks):
            breakpoint()
            # image_position_idx = self.image_position_idx[:prev_output_tokens.size(1)].unsqueeze(0).expand(bsz, tgt_len)
            # tgt_pos_embed[code_masks] = self.embed_image_positions(image_position_idx)[code_masks]

        # self attn position bias
        self_abs_pos_bias = self.get_pos_info(bsz, tgt_len, tgt_pos_embed, use_image=False)
        # if code_masks is not None and torch.any(code_masks):
            # self_image_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, use_image=True)
            # self_abs_pos_bias[code_masks] = self_image_abs_pos_bias[code_masks]

        # cross attn position bias
        src_pos_embed = encoder_out['position_embeddings'][0]
        cross_abs_pos_bias = self.get_pos_info(bsz, tgt_len, tgt_pos_embed, src_pos_embed=src_pos_embed)
        # if code_masks is not None and torch.any(code_masks):
        #     cross_image_abs_pos_bias = self.get_pos_info(bsz, tgt_len, tgt_pos_embed, src_pos_embed=src_pos_embed, use_image=True)
        #     cross_abs_pos_bias[code_masks] = cross_image_abs_pos_bias[code_masks]
        cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])


        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.entangle_position_embedding is not None and not self.args.disable_entangle:
            x += tgt_pos_embed

        if self.layernorm_embedding is not None:
            if code_masks is None or not code_masks.any() or not getattr(self, "code_layernorm_embedding", False):
                x = self.layernorm_embedding(x)
            elif code_masks is not None and code_masks.all():
                x = self.code_layernorm_embedding(x)
            else:
                x[~code_masks] = self.layernorm_embedding(x[~code_masks])
                x[code_masks] = self.code_layernorm_embedding(x[code_masks])

        x = self.dropout_module(x)

        return x, mask, self_abs_pos_bias, cross_abs_pos_bias




    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return self.max_target_positions

    def forward(self, encoder_out, normalize=False, prev_output_tokens=None, code_masks=None, step=0, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=self.src_embedding_copy,
            code_masks=code_masks
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=None,
            code_masks=None,
            **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        x, decoder_padding_mask, self_abs_pos_bias, cross_abs_pos_bias = self.forward_embedding(
            prev_output_tokens,
            encoder_out=encoder_out,
            code_masks=code_masks,
            embedding_copy=embedding_copy)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        seq_len = x.shape[0]
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):
            self_attn_bias = self_abs_pos_bias.clone()
            if code_masks is None or not code_masks.any():
                self_attn_bias += self.get_rel_pos_bias(seq_len, i).unsqueeze(0)
            elif code_masks is not None and code_masks.all():
                self_attn_bias += self.get_image_rel_pos_bias(seq_len, i).unsqueeze(0)
            else:
                self_attn_bias[~code_masks] += self.get_rel_pos_bias(seq_len, i).unsqueeze(0)
                self_attn_bias[code_masks] += self.get_image_rel_pos_bias(seq_len, i).unsqueeze(0)
            self_attn_bias = self_attn_bias.reshape(-1, *self_attn_bias.size()[-2:])

            # if self.args.decoder_prompt:
            #     if self.args.decoder_prompt_type != "prompt":
            #         prompt_kv = prompt_kv_list[idx]
            #     else:
            #         if idx == 0:
            #             prompt_kv = prompt_kv_list[idx]
            #         else:
            #             prompt_kv = None
            # else:
            #     prompt_kv = None
            #

            x, attn, _ = layer(
                x,
                encoder_out['encoder_out'][0],
                encoder_out['encoder_padding_mask'][0],
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                self_attn_bias=self_attn_bias,
                cross_attn_bias=cross_abs_pos_bias,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states, "padding_mask": decoder_padding_mask}

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        # version_key = "{}.version".format(name)
        # if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
        #     # earlier checkpoints did not normalize after the stack of layers
        #     self.layer_norm = None
        #     self.normalize = False
        #     state_dict[version_key] = torch.Tensor([1])

        prefix = name + "." if name != "" else ""
        image_params = ["image_position_idx"]
        for image_param in image_params:
            state_dict[prefix + image_param] = self.state_dict()[image_param]
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        if len(state_dict["decoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
            num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["decoder.embed_image_positions.weight"])
            embed_dim = state_dict["decoder.embed_image_positions.weight"].size(1)
            new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
            nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_pos_embed_to_add = new_pos_embed_to_add.to(
                dtype=state_dict["decoder.embed_image_positions.weight"].dtype,
            )
            state_dict["decoder.embed_image_positions.weight"] = torch.cat(
                [state_dict["decoder.embed_image_positions.weight"], new_pos_embed_to_add]
            )
        return state_dict


#
class CMLMNATransformerEncoder(NATransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.args = args
        if getattr(args, "encoder_prompt", False):
            self.encoder_prompt_encoder = PromptEncoder(
                type=args.encoder_prompt_type,
                length=args.encoder_prompt_length,
                projection=args.encoder_prompt_projection,
                embed_dim=args.encoder_embed_dim,
                proj_dim=args.encoder_prompt_dim,
                layers=args.encoder_layers,
                vocab_size=args.vocab_size)
        self.encoder_dropout = nn.Dropout(p=0.2)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.num_attention_heads = args.encoder_attention_heads

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if getattr(args, "add_type_embedding", False):
            self.type_embedding = Embedding(2, embed_dim, padding_idx=None)
        else:
            self.type_embedding = None

        if getattr(args, "sync_bn", False):
            norm_layer = BatchNorm2d
        else:
            if getattr(args, "freeze_resnet", False):
                norm_layer = FrozenBatchNorm2d
            else:
                norm_layer = None

        if args.resnet_type == 'resnet101':
            self.embed_images = ResNet([3, 4, 23], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        elif args.resnet_type == 'resnet152':
            self.embed_images = ResNet([3, 8, 36], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        elif args.resnet_type == 'resnet50':
            self.embed_images = ResNet([3, 4, 6], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        else:
            raise NotImplementedError
        self.image_proj = Linear(1024, embed_dim)
        if getattr(args, "resnet_model_path", None):
            print("load resnet {}".format(args.resnet_model_path))
            resnet_state_dict = torch.load(self.args.resnet_model_path)
            self.embed_images.load_state_dict(resnet_state_dict)
        if getattr(args, "patch_layernorm_embedding", False):
            self.patch_layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.patch_layernorm_embedding = None

        self.embed_positions = Embedding(args.max_source_positions + 2, embed_dim)
        self.embed_image_positions = Embedding(args.image_bucket_size ** 2 + 1, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        self.image_pos_ln = LayerNorm(embed_dim)
        self.pos_scaling = float(embed_dim / args.encoder_attention_heads * args.attn_scale_factor) ** -0.5
        self.pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_k_linear = nn.Linear(embed_dim, embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, args.encoder_drop_path_rate, args.encoder_layers)]
        self.layers.extend(
            [self.build_encoder_layer(args, drop_path_rate=dpr[i]) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        token_bucket_size = args.token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.encoder_layers)]
        )

        image_bucket_size = args.image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        self.image_rel_pos_table_list = nn.ModuleList(
            [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.encoder_layers)]
        )

        self.patch_image_size = args.patch_image_size
        self.orig_patch_image_size = args.orig_patch_image_size

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.entangle_position_embedding = args.entangle_position_embedding


    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return self.max_source_positions

    def build_encoder_layer(self, args, drop_path_rate=0.0):
        layer = TransformerEncoderLayer(args, drop_path_rate=drop_path_rate, \
                                        use_adapter=getattr(args, "adapter", False), adapter_dim=getattr(args, "adapter_dim", 200))
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        values = values.permute([0, 3, 1, 2])
        return values.contiguous()

    def get_image_rel_pos_bias(self, image_position_ids, idx):
        bsz, seq_len = image_position_ids.shape
        rp_bucket_size = self.image_rp_bucket.size(1)

        rp_bucket = self.image_rp_bucket.unsqueeze(0).expand(
            bsz, rp_bucket_size, rp_bucket_size
        ).gather(1, image_position_ids[:, :, None].expand(bsz, seq_len, rp_bucket_size)
                 ).gather(2, image_position_ids[:, None, :].expand(bsz, seq_len, seq_len))
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(0, 3, 1, 2)
        return values

    def get_patch_images_info(self, patch_images, sample_patch_num, device):
        image_embed = self.embed_images(patch_images)
        h, w = image_embed.shape[-2:]
        image_num_patches = h * w
        image_padding_mask = patch_images.new_zeros((patch_images.size(0), image_num_patches)).bool()
        image_position_idx = torch.arange(w).unsqueeze(0).expand(h, w) + \
                             torch.arange(h).unsqueeze(1) * self.args.image_bucket_size + 1
        image_position_idx = image_position_idx.view(-1).to(device)
        image_position_ids = image_position_idx[None, :].expand(patch_images.size(0), image_num_patches)

        image_embed = image_embed.flatten(2).transpose(1, 2)
        if sample_patch_num is not None:
            patch_orders = [
                random.sample(range(image_num_patches), k=sample_patch_num)
                for _ in range(patch_images.size(0))
            ]
            patch_orders = torch.LongTensor(patch_orders).to(device)
            image_embed = image_embed.gather(
                1, patch_orders.unsqueeze(2).expand(-1, -1, image_embed.size(2))
            )
            image_num_patches = sample_patch_num
            image_padding_mask = image_padding_mask.gather(1, patch_orders)
            image_position_ids = image_position_ids.gather(1, patch_orders)
        orig_num_patches = (self.orig_patch_image_size // 16) ** 2
        orig_hw = self.orig_patch_image_size // 16
        if getattr(self.args, "interpolate_position", False) and image_num_patches > orig_num_patches:
            old_image_position_ids = torch.arange(orig_hw).unsqueeze(0).expand(orig_hw, orig_hw) + \
                                     torch.arange(orig_hw).unsqueeze(1) * self.args.image_bucket_size + 1
            old_image_position_ids = old_image_position_ids.to(device)
            old_image_pos_embed = self.embed_image_positions(old_image_position_ids)
            old_image_pos_embed = old_image_pos_embed.reshape(1, orig_hw, orig_hw, -1).permute(0, 3, 1, 2)
            image_pos_embed = F.interpolate(old_image_pos_embed, size=(h, w), mode='bilinear')
            image_pos_embed = image_pos_embed.permute(0, 2, 3, 1).reshape(1, image_num_patches, -1)
            image_pos_embed = image_pos_embed.expand(patch_images.size(0), -1, -1)
        else:
            image_pos_embed = self.embed_image_positions(image_position_ids)

        return image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed

    def get_encoder_prompt(self, prompt_tokens):
        past_key_values = self.encoder_prompt_encoder(prompt_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            (self.args.encoder_layers) * 2,
            self.args.encoder_attention_heads,
            self.args.encoder_embed_dim // self.args.encoder_attention_heads,
        )
        past_key_values = self.encoder_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward_embedding(
            self,
            src_tokens,
            image_embed: Optional[torch.Tensor] = None,
            image_embed_2: Optional[torch.Tensor] = None,
            token_embedding: Optional[torch.Tensor] = None,
            pos_embed: Optional[torch.Tensor] = None,
            image_pos_embed: Optional[torch.Tensor] = None,
            image_pos_embed_2: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.entangle_position_embedding and pos_embed is not None:
            x += pos_embed
        if self.type_embedding is not None:
            x += self.type_embedding(src_tokens.new_zeros(x.size()[:2]))
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        # embed raw images
        if image_embed is not None:
            image_embed = self.image_proj(image_embed)
            image_x = image_embed = self.embed_scale * image_embed
            if self.entangle_position_embedding and image_pos_embed is not None:
                image_x += image_pos_embed
            if self.type_embedding is not None:
                image_x += self.type_embedding(src_tokens.new_ones(image_x.size()[:2]))
            if self.patch_layernorm_embedding is not None:
                image_x = self.patch_layernorm_embedding(image_x)
            image_x = self.dropout_module(image_x)
            if self.quant_noise is not None:
                image_x = self.quant_noise(image_x)
            x = torch.cat([image_x, x], dim=1)
            embed = torch.cat([image_embed, embed], dim=1)

        if image_embed_2 is not None:
            assert self.type_embedding is not None
            image_embed_2 = self.image_proj(image_embed_2)
            image_x_2 = image_embed_2 = self.embed_scale * image_embed_2
            if self.entangle_position_embedding and image_pos_embed_2 is not None:
                image_x_2 += image_pos_embed_2
            if self.type_embedding is not None:
                image_x_2 += self.type_embedding(src_tokens.new_full(image_x_2.size()[:2], fill_value=2))
            if self.patch_layernorm_embedding is not None:
                image_x_2 = self.patch_layernorm_embedding(image_x_2)
            image_x_2 = self.dropout_module(image_x_2)
            if self.quant_noise is not None:
                image_x_2 = self.quant_noise(image_x_2)
            x = torch.cat([image_x_2, x], dim=1)
            embed = torch.cat([image_embed_2, embed], dim=1)

        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths,
            patch_images: Optional[torch.Tensor] = None,
            patch_images_2: Optional[torch.Tensor] = None,
            patch_masks: Optional[torch.Tensor] = None,
            code_masks: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            sample_patch_num: Optional[int] = None,
            **kwarg,
    ):

        prompt_tokens = None
        prompt_padding_mask = None
        prompt_kv_list = None
        if self.args.encoder_prompt:
            bsz, seq_len = src_tokens.shape[0], src_tokens.shape[1]
            if self.args.encoder_prompt_type in ("prefix"):
                prompt_tokens = torch.arange(
                    0, self.args.encoder_prompt_length).to(
                    src_tokens.device)
                prompt_tokens = prompt_tokens.unsqueeze(0).expand(bsz, -1)
                prompt_padding_mask = torch.zeros_like(prompt_tokens).to(prompt_tokens.device)
            prompt_kv_list = self.get_encoder_prompt(prompt_tokens)
        image_embed = None
        image_embed_2 = None
        image_pos_embed = None
        image_pos_embed_2 = None
        if patch_images is not None:
            image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed = \
                self.get_patch_images_info(patch_images, sample_patch_num, src_tokens.device)
            image_padding_mask[~patch_masks] = True
        if patch_images_2 is not None:
            image_embed_2, image_num_patches_2, image_padding_mask_2, image_position_ids_2, image_pos_embed_2 = \
                self.get_patch_images_info(patch_images_2, sample_patch_num, src_tokens.device)
            image_padding_mask_2[~patch_masks] = True

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if patch_images is not None:
            encoder_padding_mask = torch.cat([image_padding_mask, encoder_padding_mask], dim=1)
        if patch_images_2 is not None:
            encoder_padding_mask = torch.cat([image_padding_mask_2, encoder_padding_mask], dim=1)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        # TODO this embedding function is different from ofa, check why.
        pos_embed = self.embed_positions(utils.new_arange(src_tokens))
        x, encoder_embedding = self.forward_embedding(
            src_tokens, image_embed, image_embed_2, token_embeddings,
            pos_embed, image_pos_embed, image_pos_embed_2
        )

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        pos_embed = self.pos_ln(pos_embed)
        if patch_images is not None:
            image_pos_embed = self.image_pos_ln(image_pos_embed)
            pos_embed = torch.cat([image_pos_embed, pos_embed], dim=1)
        if patch_images_2 is not None:
            image_pos_embed_2 = self.image_pos_ln(image_pos_embed_2)
            pos_embed = torch.cat([image_pos_embed_2, pos_embed], dim=1)

        pos_q = self.pos_q_linear(pos_embed).view(
            pos_embed.size(0), pos_embed.size(1), self.num_attention_heads, -1
        ).transpose(1, 2) * self.pos_scaling
        pos_k = self.pos_k_linear(pos_embed).view(
            pos_embed.size(0), pos_embed.size(1), self.num_attention_heads, -1
        ).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        if prompt_padding_mask is not None:
            encoder_padding_mask = torch.cat([prompt_padding_mask, encoder_padding_mask], dim=1)

        cached_fields = {
            'x': x.clone(),
            'abs_pos_bias': abs_pos_bias.clone(),
            'src_tokens': src_tokens.clone(),
            'self_attn_bias': [],
            'encoder_padding_mask': encoder_padding_mask.clone(),
            'pos_embed': pos_embed
        }

        # encoder layers
        for idx, layer in enumerate(self.layers):
            self_attn_bias = abs_pos_bias.clone()
            self_attn_bias[:, :, -src_tokens.size(1):, -src_tokens.size(1):] += self.get_rel_pos_bias(src_tokens, idx)
            if patch_images_2 is not None:
                self_attn_bias[:, :, :image_num_patches_2, :image_num_patches_2] += \
                    self.get_image_rel_pos_bias(image_position_ids_2, idx)
                self_attn_bias[:, :, image_num_patches_2:image_num_patches_2 + image_num_patches,
                image_num_patches_2:image_num_patches_2 + image_num_patches] += \
                    self.get_image_rel_pos_bias(image_position_ids, idx)
            elif patch_images is not None:
                self_attn_bias[:, :, :x.size(0) - src_tokens.size(1), :x.size(0) - src_tokens.size(1)] += \
                    self.get_image_rel_pos_bias(image_position_ids, idx)
            self_attn_bias = self_attn_bias.reshape(-1, self_attn_bias.size(2), self_attn_bias.size(2))
            if self.args.encoder_prompt:
                if self.args.encoder_prompt_type != "prompt":
                    prompt_kv = prompt_kv_list[idx]
                else:
                    if idx == 0:
                        prompt_kv = prompt_kv_list[idx]
                    else:
                        prompt_kv = None
            else:
                prompt_kv = None

            cached_fields['self_attn_bias'].append(self_attn_bias.clone())

            x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, \
                      self_attn_bias=self_attn_bias, prompt_kv=prompt_kv)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.args.encoder_prompt:
            encoder_padding_mask = encoder_padding_mask[:, prompt_tokens.size(1):]
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            # TODO necessity?
            "encoder_embedding": [x.transpose(0, 1)],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "position_embeddings": [pos_embed],  # B x T x C
            "cached_fields": cached_fields
        }


    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        # version_key = "{}.version".format(name)
        # if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
        #     # earlier checkpoints did not normalize after the stack of layers
        #     self.layer_norm = None
        #     self.normalize = False
        #     state_dict[version_key] = torch.Tensor([1])

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        if len(state_dict["encoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
            num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["encoder.embed_image_positions.weight"])
            embed_dim = state_dict["encoder.embed_image_positions.weight"].size(1)
            new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
            nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_pos_embed_to_add = new_pos_embed_to_add.to(
                dtype=state_dict["encoder.embed_image_positions.weight"].dtype,
            )
            state_dict["encoder.embed_image_positions.weight"] = torch.cat(
                [state_dict["encoder.embed_image_positions.weight"], new_pos_embed_to_add]
            )
        return state_dict


@register_model_architecture("ofa_cmlm_transformer", "ofa_cmlm_transformer")
def cmlm_base_architecture(args):
    # from transfomrmer legacy
    # args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    # args.no_token_positional_embeddings = getattr(
    #     args, "no_token_positional_embeddings", False
    # )
    # args.adaptive_input = getattr(args, "adaptive_input", False)
    # args.no_cross_attention = getattr(args, "no_cross_attention", False)
    # args.cross_self_attention = getattr(args, "cross_self_attention", False)
    #
    # args.decoder_output_dim = getattr(
    #     args, "decoder_output_dim", args.decoder_embed_dim
    # )
    # args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    #
    # args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    # args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    # args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    # args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    # args.offload_activations = getattr(args, "offload_activations", False)
    # if args.offload_activations:
    #     args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


    # from unify transformer
    args.encoder_prompt = getattr(args, "encoder_prompt", False)
    args.encoder_prompt_length = getattr(args, "encoder_prompt_length", 100)
    args.encoder_prompt_type = getattr(args, "encoder_prompt_type", "prefix")
    args.encoder_prompt_projection = getattr(args, "encoder_prompt_projection", False)
    args.encoder_prompt_dim = getattr(args, "encoder_prompt_dim", 2 * args.encoder_embed_dim)

    args.decoder_prompt = getattr(args, "decoder_prompt", False)
    args.decoder_prompt_length = getattr(args, "decoder_prompt_length", 100)
    args.decoder_prompt_type = getattr(args, "decoder_prompt_type", "prefix")
    args.decoder_prompt_projection = getattr(args, "decoder_prompt_projection", False)
    args.decoder_prompt_dim = getattr(args, "decoder_prompt_dim", 2 * args.encoder_embed_dim)


    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

