#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7061

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8
save_path=/path/NARVL/nat_ofa_snli_ve_pretrained_3
mkdir -p save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

data_dir=/path/snli_ve_data
data=${data_dir}/snli_ve_train.tsv,${data_dir}/snli_ve_dev.tsv
restore_file=/path/downloaded_checkpoints/ofa_base.pt
selected_cols=0,2,3,4,5

task=snli_ve
arch=na_ofa_base
criterion=ofa_nat_loss
label_smoothing=0.0
lr=5e-5
max_epoch=5
warmup_ratio=0.06
batch_size=4
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=480

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
    $data \
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
    --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
    --lr-scheduler=polynomial_decay --lr=${lr} \
    --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
    --log-format=simple --log-interval=10 \
    --fixed-validation-seed=7 \
    --keep-best-checkpoints=1 \
    --save-interval=1 --validate-interval=1 \
    --save-interval-updates=500 --validate-interval-updates=500 \
    --best-checkpoint-metric=snli_score --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --prompt-type=none \
    --add-caption \
    --fp16 \
    --fp16-scale-window=512 \
    --dynamic-upsample --src-upsample 1 \
    --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --share-all-embeddings \
    --predict-target 'all' --loss-type 'ctc' \
    --max-tokens 4096 --tensorboard-logdir ${save_path} \
    --num-queries 3 \
    --add-blank-id-to-dict \
    --disable-validation \
    --tensorboard-logdir ${save_path} \
    --num-workers=0
