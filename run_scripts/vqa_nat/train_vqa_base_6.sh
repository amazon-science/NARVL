#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# To use the shuffled data (if exists), please uncomment the Line 24.

# Number of GPUs per GPU worker
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=2
GPUS_PER_NODE=8
#GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
export MASTER_PORT=8313

data_dir=/path/vqa_data
#data=${data_dir}/vqa_train_1_tiny.tsv,${data_dir}/vqa_train_1_tiny.tsv
#data=${data_dir}/vqa_train.tsv,${data_dir}/vqa_val.tsv
# Note: If you have shuffled the data in advance, please uncomment the line below.
data=${data_dir}/vqa_train_1.tsv,${data_dir}/vqa_train_2.tsv,${data_dir}/vqa_train_3.tsv,${data_dir}/vqa_train_4.tsv,${data_dir}/vqa_train_5.tsv,${data_dir}/vqa_train_6.tsv,${data_dir}/vqa_train_7.tsv,${data_dir}/vqa_train_8.tsv,${data_dir}/vqa_train_9.tsv,${data_dir}/vqa_train_10.tsv,${data_dir}/vqa_val.tsv
ans2label_file=/path/vqa_data/trainval_ans2label.pkl
selected_cols=0,5,2,3,4

#save_path=/path/NARVL/nat_ofa_vqa_6
save_path=/path/NARVL/nat_ofa_vqa_6_conf
mkdir -p save_path

restore_file=/path/downloaded_checkpoints/ofa_base.pt


bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=vqa_gen
arch=na_ofa_base
criterion=ofa_nat_loss
label_smoothing=0.1
batch_size=4
update_freq=16
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_object_length=30
max_tgt_length=30
num_bins=1000
patch_image_size=480

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0
max_epoch=10
warmup_ratio=0.04
#warmup_ratio=0.00
lr=5e-5

# Specify the inference type in validation after each fine-tuning epoch
# As mentioned in the readme, you can choose from allcand or beamsearch evaluation, default to allcand
val_inference_type=beamsearch

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
    ${data} \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${restore_file} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=0.01 \
    --optimizer=adam \
    --adam-betas="(0.9,0.999)" \
    --adam-eps=1e-08 \
    --clip-norm=1.0 \
    --lr-scheduler=polynomial_decay \
    --lr=${lr} \
    --max-epoch=${max_epoch} \
    --warmup-ratio=${warmup_ratio} \
    --log-format=simple \
    --log-interval=10 \
    --fixed-validation-seed=7 \
    --keep-last-epochs=15 \
    --save-interval=1 --validate-interval=1 \
    --best-checkpoint-metric=vqa_score --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-object-length=${max_object_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --freeze-encoder-embedding \
    --freeze-decoder-embedding \
    --ans2label-file=${ans2label_file} \
    --valid-batch-size=20 \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --prompt-type=none \
    --fp16 \
    --fp16-scale-window=512 \
    --add-object \
    ${uses_ema} \
    ${store_ema} \
    ${ema_fp32} \
    --ema-decay=${ema_decay} \
    --ema-start-update=${ema_start_update} \
    --val-inference-type=${val_inference_type} \
    --dynamic-upsample --src-upsample 1 \
    --num-queries 6 \
    --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --share-all-embeddings \
    --predict-target 'all' --loss-type 'ctc' \
    --max-tokens 4096 --tensorboard-logdir ${save_path} \
    --add-blank-id-to-dict \
    --tensorboard-logdir ${save_path} \
    --num-workers=0
