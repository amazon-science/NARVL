#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6091
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1


########################## Evaluate Refcoco ##########################
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3

data_dir=/path/refcoco
data=${data_dir}/refcoco_val.tsv

checkpoint_path=/path/NARVL/nat_ofa_refcoco_base_seed2/checkpoint_best.pt
save_path=/path/NARVL/nat_ofa_refcoco_base_tmp

result_path=${save_path}/result

split='refcoco_val'
torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

#    --decoder_xformers_att_config='{"name": "scaled_dot_product"}' \


data=${data_dir}/refcoco_testA.tsv
split='refcoco_testA'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=${data_dir}/refcoco_testB.tsv
split='refcoco_testB'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"




######################### Evaluate Refcocoplus ##########################
model_root=/path/NARVL/nat_ofa_refcocoplus_pretrained_5_10epoch
checkpoint_path=${model_root}/checkpoint_last.pt
save_path=${model_root}/results

data_dir=/path/refcocoplus
data=${data_dir}/refcocoplus_val.tsv
result_path=${save_path}/refcocoplus
split='refcocoplus_val'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=${data_dir}/refcocoplus_testA.tsv
split='refcocoplus_testA'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=${data_dir}/refcocoplus_testB.tsv
split='refcocoplus_testB'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"



########################## Evaluate Refcocog ##########################
model_root=/path/NARVL/nat_ofa_refcocog_pretrained_5_12epoch
checkpoint_path=${model_root}/checkpoint_last.pt
save_path=${model_root}/results_debug

data_dir=/path/refcocog
data=${data_dir}/refcocog_val.tsv
result_path=${save_path}/refcocog
split='refcocog_val'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=${data_dir}/refcocog_test.tsv
split='refcocog_test'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${checkpoint_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --iter-decode-max-iter 0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
