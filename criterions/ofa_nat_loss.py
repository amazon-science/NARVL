# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.scoring import bleu
import sacrebleu
import random

@register_criterion("ofa_nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):

    def __init__(
        self, task, label_smoothing, predict_target, loss_type
        ):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.mask_index = task.tgt_dict.unk()
        self.padding_idx = task.tgt_dict.pad()
        self.eps_index = task.tgt_dict.axe_eps_idx
        self.predict_target = predict_target
        self.loss_type = loss_type
        self.tgt_dict = task.tgt_dict
        self.bpe = task.bpe
        if loss_type == 'dtw':
            from fairseq.criterions.nat_dtw_loss import SoftDTW
            self.dtw_loss = SoftDTW(gamma=self.dtw_gamma)   # gamma ?

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            '--label-smoothing',
            default=0.,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing',
        )
        parser.add_argument(
            '--predict-target',
            default='all',
            choices=['all', 'partial-all', 'partial-mask']
        )
        parser.add_argument(
            '--loss-type',
            default='nll',
            choices=['nll', 'ctc']
        )

    def _compute_loss(
        self, outputs, targets, masks=None, 
        label_smoothing=0.0, 
        name="loss", 
        factor=1.0, 
        loss_type='nll', 
        out_masks=None,
        conf=None
    ):

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            targets = targets.to(outputs.device)
            
            if loss_type == 'ctc':
                logits_lengths = out_masks.sum(1).long()
                target_lengths = targets.ne(self.padding_idx).sum(1).long()
                logits = utils.log_softmax(outputs, dim=-1) * conf
                logits = logits.transpose(0, 1)
                _scores, _tokens = logits.max(-1)
                # breakpoint()
                if random.random() < 0.01:
                    # print('logit',  _tokens[0:20, 0])
                    # print('target', targets[0][0:20])
                    decoded_sequence = self.tgt_dict.string(targets[0]).split(' ')
                    print('target', self.bpe.decode(' '.join(decoded_sequence)))

                    decoded_sequence = self.tgt_dict.string(_tokens[:, 0]).split(' ')
                    print('prediction', self.bpe.decode(' '.join(decoded_sequence)))
                    # breakpoint()

                nll_loss = F.ctc_loss(logits,
                                      targets,
                                      logits_lengths,
                                      target_lengths,
                                      blank=self.eps_index,
                                      reduction='mean',
                                      zero_infinity=True)

            else:   # we only compute loss for masked tokens.
                if masks is not None:
                    outputs, targets = outputs[masks], targets[masks]
                logits = utils.log_softmax(outputs, dim=-1)  # try fixing logsoftmax??

                if random.random() < 0.01:
                    logits_tmp = logits.transpose(0, 1)
                    _scores, _tokens = logits_tmp.max(-1)
                    decoded_sequence = self.tgt_dict.string(targets[0]).split(' ')
                    print('target', self.bpe.decode(' '.join(decoded_sequence)))

                    decoded_sequence = self.tgt_dict.string(_tokens[:, 0]).split(' ')
                    print('prediction', self.bpe.decode(' '.join(decoded_sequence)))

                if True:
                    logits = logits.transpose(1, 2)
                    min_len = min(logits.shape[-1], targets.shape[1])
                    logits = logits[:, :, :min_len]
                    targets = targets[:, :min_len]
                    losses = F.nll_loss(logits, targets, reduction='none')
                else:  # soft-labels
                    # _scores, _tokens = logits.max(-1)
                    losses = F.kl_div(logits, targets, reduction='none')
                    losses = losses.sum(-1)
                nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (
                    1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss
        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if isinstance(sample, list):
            breakpoint()

        outputs = model(**sample["net_input"],
                        tgt_tokens=sample['target'],
                        tgt_lengths=sample['tgt_lengths']
                        )

        losses, nll_loss = [], []

        conf = sample['conf'][:, None, None] if 'conf' in sample and sample['conf'] is not None else 1

        for obj in outputs:
            if obj == 'add_logs':
                continue
            elif outputs[obj].get("loss", None) is None:
                loss_type = outputs[obj].get("type", "nll")
                if obj == 'word_ins':
                    loss_type = self.loss_type

                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0),
                    loss_type=loss_type,
                    conf=conf,
                    out_masks=outputs[obj].get("out_mask", None)
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if 'add_logs' in outputs:
            logging_output.update({
                name + "-log": outputs['add_logs'][name] 
                    for name in outputs['add_logs']
                    if outputs['add_logs'][name] is not None
                })

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
            if key[-4:] == "-log":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-4],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
