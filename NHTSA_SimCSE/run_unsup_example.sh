#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/COMPLAINTS_RECEIVED_2020_2023/com_data/train.txt \
    --output_dir result \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 215 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    # --fp16 \
    # "$@"

