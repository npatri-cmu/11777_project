#!/bin/bash
#SBATCH -J attn_acc
#SBATCH -A cis260325p
#SBATCH -p GPU-shared,GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH -t 03:00:00
#SBATCH -o /jet/home/xwang93/11777/11777_multimodal_reasoning/attention_run.log
#SBATCH -e /jet/home/xwang93/11777/11777_multimodal_reasoning/attention_run.log

echo "Starting on $(hostname) at $(date)"
nvidia-smi

export HF_HOME=/jet/home/xwang93/11777/.cache/huggingface
export TRANSFORMERS_CACHE=/jet/home/xwang93/11777/.cache/huggingface/hub
export PIP_CACHE_DIR=/jet/home/xwang93/11777/.cache/pip
export CONDA_PKGS_DIRS=/jet/home/xwang93/11777/conda_pkgs
export PYTHONUNBUFFERED=1

export PATH=/jet/home/xwang93/11777/conda_envs/rq1/bin:$PATH
export PYTHON=/jet/home/xwang93/11777/conda_envs/rq1/bin/python

cd /jet/home/xwang93/11777/11777_multimodal_reasoning

echo "=== Core Experiment: Attention vs Accuracy ==="
python scripts/attention_vs_accuracy.py \
    --model checkpoints/Qwen3-VL-8B-Instruct \
    --data_root data/vstar \
    --dtype float16 \
    --out outputs/attention_vs_accuracy

echo "Finished at $(date)"

