export OUTPUT_DIR=./bert_results/

python run_simi_cls.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=bert-base-chinese \
    --train_file=../../data/afqmc_public/train.json \
    --dev_file=../../data/afqmc_public/dev.json \
    --test_file=../../data/afqmc_public/dev.json \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=16 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42