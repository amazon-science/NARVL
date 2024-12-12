ln vqa_train_distill_base.tsv vqa_train_distill_base_1.tsv
for idx in `seq 1 9`;do shuf vqa_train_distill_base_${idx}.tsv > vqa_train_distill_base_$[${idx}+1].tsv;done # each file is used for an epoch


ln vqa_train_distill_huge.tsv vqa_train_distill_huge_1.tsv
for idx in `seq 1 9`;do shuf vqa_train_distill_huge_${idx}.tsv > vqa_train_distill_huge_$[${idx}+1].tsv;done # each file is used for an epoch
