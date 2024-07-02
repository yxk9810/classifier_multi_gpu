#!/bin/bash
export WANDB_DISABLED="true"

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6006

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS  -m reranker.run \
--output_dir ./outputs/ \
--model_name_or_path google-bert/bert-base-chinese \
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

