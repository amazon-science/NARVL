# Non-Autoregressive Sequence-to-Sequence Vision-Language Models

This repo shares the code for CVPR 2024 paper https://arxiv.org/abs/2403.02249.

## Abstract

Sequence-to-sequence vision-language models are showing promise, but their applicability is limited by their inference latency due to their autoregressive way of generating predictions. We propose a parallel decoding sequence-to-sequence vision-language model, trained with a Query-CTC loss, that marginalizes over multiple inference paths in the decoder. This allows us to model the joint distribution of tokens, rather than restricting to conditional distribution as in an autoregressive model. The resulting model, NARVL, achieves performance on-par with its state-of-the-art autoregressive counterpart, but is faster at inference time, reducing from the linear complexity associated with the sequential generation of tokens to a paradigm of constant time joint inference.


## Installation

```
conda create --name narvl python=3.9 -y
conda activate narvl
cd /path/to/NARVL
pip install -r requirements.txt
pip install -e ./fairseq/
pip install torch torchvision -y
export PYTHONPATH=/path/to/NARVL/fairseq
```

ctcdecode pakcage can be found here: https://github.com/parlance/ctcdecode


## Scripts

Train and evaluation scripts can be found in run_scripts folder.


## Related Projects

- [fairseq](https://github.com/pytorch/fairseq): A PyTorch-based sequence modeling toolkit.
- [OFA](https://github.com/OFA-Sys/OFA): A Unified Transformer for Vision-Language Tasks.


## Citation

If you use this code or the ideas from the paper, please cite:

```
@inproceedings{shi2024non,
  title={Non-autoregressive Sequence-to-Sequence Vision-Language Models},
  author={Shi, Kunyu and Dong, Qi and Goncalves, Luis and Tu, Zhuowen and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13603--13612},
  year={2024}
}

```