#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7095
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

#export CUDA_VISIBLE_DEVICES=0
#export GPUS_PER_NODE=1

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=dev
#split=test

data_dir=/path/snli_ve_data
data=${data_dir}/snli_ve_dev.tsv
#data=${data_dir}/snli_ve_test.tsv
checkpoint_path=/path/NARVL/nat_ofa_snli_ve_pretrained_3/checkpoint_5_9000.pt
result_path=/path/NARVL/nat_ofa_snli_ve_pretrained_3/results/
selected_cols=0,2,3,4,5

 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"