# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

"""
OFA
"""
from typing import Optional

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.utils import new_arange, softmax, log_softmax
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .ofa_cmlm_transformer import OFACMLMNATransformerModel

logger = logging.getLogger(__name__)


def _skeptical_unmasking(output_scores, output_masks, p=None, boundary_len=None):
    sorted_index = output_scores.sort(-1)[1]
    if boundary_len is None:
        boundary_len = (
                (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
        ).long()
    skeptical_mask = new_arange(output_masks).type_as(boundary_len) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("na_ofa")
class NAOFAModel(OFACMLMNATransformerModel):
    __jit_unused_properties__ = ["supported_targets"]

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @staticmethod
    def add_args(parser):
        super(NAOFAModel, NAOFAModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-classifier",
            type=str,
            choices=['mlp', 'linear'],
            help="type of pooler classifier",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward_encoder(self, encoder_inputs):
        encoder_out = self.encoder(**encoder_inputs)
        return encoder_out

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        if decoder_out.attn is not None:
            full_z = decoder_out.attn
            # encoder_out = encoder_out._replace(encoder_embedding=
            #                                    encoder_out.encoder_embedding + full_z)  # For simple, add Z on encoder out..
            encoder_out['encoder_embedding'][0] = encoder_out['encoder_embedding'][0] + full_z

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

            # breakpoint()
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

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens: Optional[torch.Tensor] = None,
            tgt_lengths: Optional[torch.Tensor] = None,
            patch_images: Optional[torch.Tensor] = None,
            patch_images_2: Optional[torch.Tensor] = None,
            patch_masks: Optional[torch.Tensor] = None,
            code_masks: Optional[torch.Tensor] = None,
            sample_patch_num: Optional[int] = None,
            features_only: bool = False,
            classification_head_name: Optional[str] = None,
            token_embeddings: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True
        all_results = dict()

        encoder_out = self.encoder(
            # tgt_tokens,
            src_tokens,
            src_lengths=src_lengths,
            patch_images=patch_images,
            patch_masks=patch_masks,
            patch_images_2=patch_images_2,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            sample_patch_num=sample_patch_num
        )
        # VAE part
        if self.latent_dim > 0:
            prior_prob = self.prob_esitmator(encoder_out['encoder_embedding'][0])

            posterior_encoder_out = self.q_encoder_y(
                tgt_tokens,
                src_lengths=tgt_lengths,
                patch_images=None,
                patch_masks=None,
                patch_images_2=None,
                token_embeddings=token_embeddings,
                return_all_hiddens=return_all_hiddens,
                sample_patch_num=sample_patch_num,
            )
            posterior_encoder_out['cached_fields'] = encoder_out['cached_fields']
            posterior_dec_out = self.q_encoder_xy.extract_features(src_tokens, encoder_out=posterior_encoder_out)[0]
            sampled_z, q_prob = self.bottleneck(posterior_dec_out, sampling=True)
            full_z = self.latent_map(sampled_z)
            encoder_out['encoder_embedding'][0] = encoder_out['encoder_embedding'][0] + full_z # For simple, add Z on encoder out..
            kl, budget = self.bottleneck.compute_final_loss(q_prob, prior_prob, sampled_z)
            kl_loss = kl[~encoder_out['encoder_padding_mask'][0]].mean()
            all_results["kl_loss"] = {
                "loss": kl_loss, 'factor': 1.0,
            }
            all_results["add_logs"] = {'kl_budget': budget}

        word_ins_out, extra = self.decoder(
            encoder_out,
            prev_output_tokens=prev_output_tokens,
            code_masks=code_masks,
        )
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

    def register_embedding_tokens(self, ans2label_dict, src_dict, bpe):
        """Register embedding tokens"""
        logger.info("Registering embedding tokens")
        self.ans_tensor_list = []
        for i in range(len(ans2label_dict)):
            ans = src_dict[-len(ans2label_dict) + i]
            ans = ans[5:-1].replace('_', ' ')
            ans_tensor = src_dict.encode_line(
                line=bpe.encode(' {}'.format(ans.lower())),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            self.ans_tensor_list.append(ans_tensor)

    def register_classification_head(
            self, name, num_classes=None, inner_dim=None, use_two_images=False, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = OFAClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            pooler_classifier=self.args.pooler_classifier,
            use_two_images=use_two_images,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
                ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
                ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes
                        != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim
                        != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
                loaded_dict_size == len(self.encoder.dictionary) + 1
                and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        if loaded_dict_size < len(self.encoder.dictionary):
            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            if getattr(self, "ans_tensor_list", None):
                assert len(new_lang_embed_to_add) == len(self.ans_tensor_list)
                for i, ans_tensor in enumerate(self.ans_tensor_list):
                    ans_embed = F.embedding(ans_tensor, state_dict["encoder.embed_tokens.weight"])
                    ans_embed = ans_embed.sum(0) / ans_embed.size(0)
                    new_lang_embed_to_add[i] = ans_embed
            else:
                nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [state_dict["encoder.embed_tokens.weight"], new_lang_embed_to_add]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [state_dict["decoder.embed_tokens.weight"], new_lang_embed_to_add]
            )
            state_dict["decoder.output_projection.weight"] = torch.cat(
                [state_dict["decoder.output_projection.weight"], new_lang_embed_to_add]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class OFAClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            activation_fn,
            pooler_dropout,
            pooler_classifier,
            use_two_images=False,
            do_spectral_norm=False,
    ):
        super().__init__()
        self.pooler_classifier = pooler_classifier
        self.use_two_images = use_two_images
        input_dim = input_dim * 2 if use_two_images else input_dim
        if pooler_classifier == "mlp":
            self.dense = nn.Linear(input_dim, inner_dim)
            self.activation_fn = utils.get_activation_fn(activation_fn)
            self.dropout = nn.Dropout(p=pooler_dropout)
            self.out_proj = nn.Linear(inner_dim, num_classes)
        elif pooler_classifier == "linear":
            self.dropout = nn.Dropout(p=pooler_dropout)
            self.out_proj = nn.Linear(input_dim, num_classes)
        else:
            raise NotImplementedError

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        if self.pooler_classifier == 'mlp':
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
            x = self.out_proj(x)
        elif self.pooler_classifier == 'linear':
            x = features
            x = self.dropout(x)
            x = self.out_proj(x)
        else:
            raise NotImplementedError
        return x


@register_model_architecture("na_ofa", "na_ofa_large")
def ofa_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.0)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_classifier = getattr(args, "pooler_classifier", "mlp")

    args.resnet_drop_path_rate = getattr(args, "resnet_drop_path_rate", 0.0)
    args.encoder_drop_path_rate = getattr(args, "encoder_drop_path_rate", 0.0)
    args.decoder_drop_path_rate = getattr(args, "decoder_drop_path_rate", 0.0)

    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    args.token_bucket_size = getattr(args, "token_bucket_size", 256)
    args.image_bucket_size = getattr(args, "image_bucket_size", 42)

    args.freeze_encoder_embedding = getattr(args, "freeze_encoder_embedding", False)
    args.freeze_decoder_embedding = getattr(args, "freeze_decoder_embedding", False)
    args.add_type_embedding = getattr(args, "add_type_embedding", True)
    args.attn_scale_factor = getattr(args, "attn_scale_factor", 2)

    args.code_image_size = getattr(args, "code_image_size", 128)
    args.patch_layernorm_embedding = getattr(args, "patch_layernorm_embedding", True)
    args.code_layernorm_embedding = getattr(args, "code_layernorm_embedding", True)
    args.entangle_position_embedding = getattr(args, "entangle_position_embedding", False)
    args.disable_entangle = getattr(args, "disable_entangle", False)
    args.sync_bn = getattr(args, "sync_bn", False)

    args.scale_attn = getattr(args, "scale_attn", False)
    args.scale_fc = getattr(args, "scale_fc", False)
    args.scale_heads = getattr(args, "scale_heads", False)
    args.scale_resids = getattr(args, "scale_resids", False)

    args.orig_patch_image_size = getattr(args, "orig_patch_image_size", 256)


@register_model_architecture("na_ofa", "na_ofa_base")
def na_ofa_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.resnet_type = getattr(args, "resnet_type", "resnet101")
    ofa_large_architecture(args)


@register_model_architecture("na_ofa", "na_ofa_huge")
def na_ofa_huge_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1280)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    ofa_large_architecture(args)


@register_model_architecture("na_ofa", "na_ofa_medium")
def ofa_medium_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 512)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.resnet_type = getattr(args, "resnet_type", "resnet101")
    ofa_large_architecture(args)


@register_model_architecture("na_ofa", "na_ofa_tiny")
def ofa_tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.resnet_type = getattr(args, "resnet_type", "resnet50")
    ofa_large_architecture(args)
