#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=2 \
# python -m llava.serve.cli \
#     --model-path /home/ziheng/ssd-drive1/projects/LLaVA/HF-llava-v1.5-7b/llava-v1.5-7b \
#     --image-file "https://llava-vl.github.io/static/images/view.jpg"


CUDA_VISIBLE_DEVICES=2 \
python -m llava.serve.cli_adaptive \
    --model-base /home/ziheng/ssd-drive1/projects/LLaVA/HF-llava-v1.5-7b/hf-llama2-7b-chat \
    --model-path /home/ziheng/ssd-drive1/projects/LLaVA/checkpoints/7b-chat-verify-earlyfuse-alllayers-multiheadgating \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --concat-projection \
    --attn-adapter