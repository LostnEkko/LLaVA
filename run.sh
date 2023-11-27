#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=2 \
# python -m llava.serve.cli \
#     --model-path /home/ziheng/ssd-drive1/projects/LLaVA/HF-llava-v1.5-7b/llava-v1.5-7b \
#     --image-file "https://llava-vl.github.io/static/images/view.jpg"


CUDA_VISIBLE_DEVICES=0 \
python -m llava.serve.cli \
    --model-base /home/ziheng/ssd-drive1/projects/LLaVA/HF-llava-v1.5-13b/llava-v1.5-13b \
    --model-path /home/ziheng/ssd-drive1/projects/LLaVA/checkpoints/13b-chat-pretrianed-proj-withlearnedpad-2/checkpoint-7600 \
    --image-file "https://llava.hliu.cc/file=/nobackup/haotian/code/LLaVA/llava/serve/examples/extreme_ironing.jpg" \
    --concat-projection