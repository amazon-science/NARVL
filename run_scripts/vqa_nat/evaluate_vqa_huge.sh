#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8182

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
#split=val
split=test

data=/path/vqa_data/vqa_${split}.tsv
ans2label_file=/path/vqa_data/trainval_ans2label.pkl
checkpoint_path=/path/NARVL/nat_ofa_huge_vqa_6_distill_from_huge/checkpoint_last.pt


result_path=/path/NARVL/nat_ofa_huge_vqa_6_distill_from_huge/results

selected_cols=0,5,2,3,4
valid_batch_size=20

python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --batch-size=4 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --iter-decode-max-iter 0 \
    --iter-decode-collapse-repetition \
    --ema-eval \
    --beam-search-vqa-eval \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\",\"valid_batch_size\":\"${valid_batch_size}\"}"