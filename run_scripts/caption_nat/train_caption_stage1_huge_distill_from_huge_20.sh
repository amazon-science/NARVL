#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=2057
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export GPUS_PER_NODE=1
export GPUS_PER_NODE=8

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

data_dir=/path/caption_data
data=${data_dir}/caption_stage1_train_distill_huge.tsv,${data_dir}/caption_val_distill.tsv
restore_file=/path/downloaded_checkpoints/ofa_huge.pt
selected_cols=0,4,2

task=caption
arch=na_ofa_huge
criterion=ofa_nat_loss
label_smoothing=0.1
lr=1e-5
max_epoch=5
warmup_ratio=0.06
batch_size=2
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
eval_cider_cached=${data_dir}/cider_cached_tokens/coco-valid-words.p
drop_worst_ratio=0.2
drop_worst_after=6000

save_path=/path/NARVL/nat_ofa_huge_caption_pretrained_distill_from_huge_20
mkdir -p save_path

python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=${MASTER_PORT} ../../train.py \
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
    --no-epoch-checkpoints --keep-best-checkpoints=1 \
    --save-interval=1 --validate-interval=1 \
    --save-interval-updates=5000 --validate-interval-updates=500 \
    --eval-cider \
    --eval-cider-cached-tokens=${eval_cider_cached} \
    --eval-args='{"beam":5,"max_len_b":16,"no_repeat_ngram_size":3}' \
    --best-checkpoint-metric=cider --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --freeze-encoder-embedding \
    --freeze-decoder-embedding \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --fp16 \
    --fp16-scale-window=512 \
    --dynamic-upsample --src-upsample 1 \
    --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --share-all-embeddings \
    --predict-target 'all' --loss-type 'ctc' \
    --num-queries 20 \
    --max-tokens 4096 --tensorboard-logdir ${save_path} \
    --add-blank-id-to-dict \
    --disable-validation \
    --tensorboard-logdir ${save_path} \
    --num-workers=0