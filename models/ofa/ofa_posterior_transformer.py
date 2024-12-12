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


class PosteriorTransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.src_upsample = getattr(args, "src_upsample", 1)
        # self.upsample_bias = getattr(args, "src_upsample_bias", None)
        self.dynamic = getattr(args, "dynamic_upsample", False)
        self.replace_target_embed = getattr(args, "replace_target_embed", False)
        self.decoder_input_type = getattr(args, "decoder_input_type", 'parameters')
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
        # self.max_target_positions = args.max_target_positions * 2

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
        # self.embed_positions = Embedding(args.max_target_positions * 2 + 2, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        # self.image_pos_ln = LayerNorm(embed_dim)
        self.pos_scaling = float(embed_dim / self.num_attention_heads * args.attn_scale_factor) ** -0.5
        # self.self_pos_q_linear = nn.Linear(embed_dim, embed_dim)
        # self.self_pos_k_linear = nn.Linear(embed_dim, embed_dim)
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

        # token_bucket_size = args.token_bucket_size
        # token_num_rel_dis = 2 * token_bucket_size - 1
        # token_rp_bucket = make_token_bucket_position(token_bucket_size)
        # self.token_rel_pos_table_list = nn.ModuleList(
        #     [Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        # )

        # image_bucket_size = args.image_bucket_size
        # image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        # image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        # image_position_idx = torch.arange(self.window_size).unsqueeze(0).expand(self.window_size, self.window_size) + \
        #                      torch.arange(self.window_size).unsqueeze(1) * image_bucket_size + 1
        # image_position_idx = torch.cat([torch.tensor([0]), image_position_idx.view(-1)])
        # image_position_idx = torch.cat([image_position_idx, torch.tensor([1024] * 769)])
        # self.image_rel_pos_table_list = nn.ModuleList(
        #     [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        # )

        # self.register_buffer("token_rp_bucket", token_rp_bucket)
        # self.register_buffer("image_rp_bucket", image_rp_bucket)
        # self.register_buffer("image_position_idx", image_position_idx)
        self.entangle_position_embedding = args.entangle_position_embedding



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


    def get_rel_pos_bias(self, seq_len, idx):
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    # def get_image_rel_pos_bias(self, seq_len, idx):
    #     image_position_idx = self.image_position_idx[:seq_len]
    #     rp_bucket = self.image_rp_bucket[image_position_idx][:, image_position_idx]
    #     values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
    #     values = values.permute(2, 0, 1)
    #     return values

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

    # def forward_embedding(self, x, embedding_copy=None, encoder_out=None, code_masks=None):
    #
    #     bsz, tgt_len = x.shape[0], x.shape[1]
    #     token_position_idx = utils.new_arange(x, x.shape[0], x.shape[1])
    #     tgt_pos_embed = self.embed_positions(token_position_idx)
    #
    #     # cross attn position bias
    #     src_pos_embed = encoder_out['position_embeddings'][0]
    #     cross_abs_pos_bias = self.get_pos_info(bsz, tgt_len, tgt_pos_embed, src_pos_embed=src_pos_embed)
    #     cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])
    #
    #
    #     if self.quant_noise is not None:
    #         x = self.quant_noise(x)
    #
    #     if self.project_in_dim is not None:
    #         x = self.project_in_dim(x)
    #
    #     if self.entangle_position_embedding is not None and not self.args.disable_entangle:
    #         x += tgt_pos_embed
    #
    #     if self.layernorm_embedding is not None:
    #         if code_masks is None or not code_masks.any() or not getattr(self, "code_layernorm_embedding", False):
    #             x = self.layernorm_embedding(x)
    #         elif code_masks is not None and code_masks.all():
    #             x = self.code_layernorm_embedding(x)
    #         else:
    #             x[~code_masks] = self.layernorm_embedding(x[~code_masks])
    #             x[code_masks] = self.code_layernorm_embedding(x[code_masks])
    #
    #     x = self.dropout_module(x)
    #
    #     return x, mask, self_abs_pos_bias, cross_abs_pos_bias


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
            given_embeddings=None,
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
        # x, decoder_padding_mask, self_abs_pos_bias, cross_abs_pos_bias = self.forward_embedding(
        #     prev_output_tokens,
        #     encoder_out=encoder_out,
        #     code_masks=code_masks,
        #     embedding_copy=embedding_copy)
        cached_fields = encoder_out['cached_fields']
        x = cached_fields['x']
        decoder_padding_mask = cached_fields['encoder_padding_mask']
        tgt_pos_embed = cached_fields['pos_embed']

        x = x.transpose(0, 1)
        bsz, tgt_len = x.shape[0], x.shape[1]
        # token_position_idx = utils.new_arange(x, x.shape[0], x.shape[1])
        # tgt_pos_embed = self.embed_positions(token_position_idx)

        # cross attn position bias
        src_pos_embed = encoder_out['position_embeddings'][0]
        cross_abs_pos_bias = self.get_pos_info(bsz, tgt_len, tgt_pos_embed, src_pos_embed=src_pos_embed)
        cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])

        # for idx, layer in enumerate(self.layers):
        #
        #     x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, \
        #               self_attn_bias=self_attn_bias, prompt_kv=prompt_kv)
        #     if return_all_hiddens:
        #         assert encoder_states is not None
        #         encoder_states.append(x)

        # # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # seq_len = x.shape[0]
        # attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):
            self_attn_bias = cached_fields['self_attn_bias'][i]

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
        

        prefix = name + "." if name != "" else ""
        # image_params = ["image_position_idx"]
        # for image_param in image_params:
        #     state_dict[prefix + image_param] = self.state_dict()[image_param]
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        # if len(state_dict["decoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
        #     num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["decoder.embed_image_positions.weight"])
        #     embed_dim = state_dict["decoder.embed_image_positions.weight"].size(1)
        #     new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
        #     nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim ** -0.5)
        #     new_pos_embed_to_add = new_pos_embed_to_add.to(
        #         dtype=state_dict["decoder.embed_image_positions.weight"].dtype,
        #     )
        #     state_dict["decoder.embed_image_positions.weight"] = torch.cat(
        #         [state_dict["decoder.embed_image_positions.weight"], new_pos_embed_to_add]
        #     )
        return state_dict


#
class PosteriorTransformerEncoder(NATransformerEncoder):
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

        # if args.resnet_type == 'resnet101':
        #     self.embed_images = ResNet([3, 4, 23], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        # elif args.resnet_type == 'resnet152':
        #     self.embed_images = ResNet([3, 8, 36], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        # elif args.resnet_type == 'resnet50':
        #     self.embed_images = ResNet([3, 4, 6], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        # else:
        #     raise NotImplementedError
        # self.image_proj = Linear(1024, embed_dim)
        # if getattr(args, "resnet_model_path", None):
        #     print("load resnet {}".format(args.resnet_model_path))
        #     resnet_state_dict = torch.load(self.args.resnet_model_path)
        #     self.embed_images.load_state_dict(resnet_state_dict)
        # if getattr(args, "patch_layernorm_embedding", False):
        #     self.patch_layernorm_embedding = LayerNorm(embed_dim)
        # else:
        #     self.patch_layernorm_embedding = None

        # self.embed_positions = Embedding(args.max_source_positions + 2, embed_dim)
        # self.embed_image_positions = Embedding(args.image_bucket_size ** 2 + 1, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        # self.image_pos_ln = LayerNorm(embed_dim)
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
        #
        # if self.args.latent_dim > 0:
        #     self.posterior_layers = nn.ModuleList([])
        #     breakpoint()
        #     posterior_dpr = [x.item() for x in torch.linspace(0, args.encoder_drop_path_rate, args.posterior_layers)]
        #     self.posterior_layers.extend(
        #         [self.build_encoder_layer(args, drop_path_rate=posterior_dpr[i]) for i in range(args.posterior_layers)]
        #     )
        #     self.num_posterior_layers = len(self.posterior_layers)
        #

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
        #
        # image_bucket_size = args.image_bucket_size
        # image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        # image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        # self.image_rel_pos_table_list = nn.ModuleList(
        #     [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.encoder_layers)]
        # )

        self.patch_image_size = args.patch_image_size
        self.orig_patch_image_size = args.orig_patch_image_size

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        # self.register_buffer("image_rp_bucket", image_rp_bucket)
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

        if self.add_first_token:
            breakpoint()

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
            posterior_mode=None,
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
        # encoder layers
        # if not posterior_mode:
        #     loop_layers = self.layers
        # else:
        #     loop_layers = self.posterior_layers
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

        # if len(state_dict["encoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
        #     num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["encoder.embed_image_positions.weight"])
        #     embed_dim = state_dict["encoder.embed_image_positions.weight"].size(1)
        #     new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
        #     nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim ** -0.5)
        #     new_pos_embed_to_add = new_pos_embed_to_add.to(
        #         dtype=state_dict["encoder.embed_image_positions.weight"].dtype,
        #     )
        #     state_dict["encoder.embed_image_positions.weight"] = torch.cat(
        #         [state_dict["encoder.embed_image_positions.weight"], new_pos_embed_to_add]
        #     )
        return state_dict


