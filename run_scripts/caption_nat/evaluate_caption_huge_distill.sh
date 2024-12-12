#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data_dir=/path/caption_data
#data=${data_dir}/caption_test.tsv
data=/path/caption_data/caption_video_converted_val.tsv
save_root=/path/NARVL/nat_ofa_huge_caption_pretrained_distill_from_huge_20
model_path=${save_root}/checkpoint_last.pt
result_path=/path/NARVL/nat_ofa_huge_caption_pretrained_distill_video/results

#selected_cols=1,4,2
selected_cols=0,1,2
split='test'

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${model_path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=1 \
    --iter-decode-collapse-repetition \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --iter-decode-max-iter 0 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

#python coco_eval.py ${result_path}/test_predict.json ${data_dir}/test_caption_coco_format.json
