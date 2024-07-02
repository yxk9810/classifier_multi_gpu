#!/bin/bash
export WANDB_DISABLED="true"


python -m reranker.evaluate \
--output_dir ./outputs/ \
--model_name_or_path ./outputs \
--train_data ./data/train.jsonl \
--dev_data ./data/dev.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 24 \
--gradient_accumulation_steps 1 \
--fp16 \
--save_total_limit 5 \
--logging_steps 50 \
--dataloader_num_workers 25 \
--do_eval \
--save_steps 1000 \

