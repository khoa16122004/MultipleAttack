#!/bin/sh
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

python main.py --pretrained llava-onevision-qwen2-7b-ov --model_name llava_qwen --max_query 50 --lambda_ 50 --epsilon 0.01 --prefix_path "test" 