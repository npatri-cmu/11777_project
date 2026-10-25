#!/bin/bash
# Source from Bash on the cluster before setup, downloads, analysis or GPU jobs.
export RQ2_PROJECT_DIR=/home/npatri/rq2_visual_faithfulness
export RQ2_ENV_PREFIX=/home/npatri/rq2_visual_faithfulness/.conda/rq2
export HF_HOME=/home/npatri/rq2_visual_faithfulness/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export MPLCONFIGDIR=/home/npatri/rq2_visual_faithfulness/.cache/matplotlib
export TORCH_HOME=/home/npatri/rq2_visual_faithfulness/.cache/torch
export XDG_CACHE_HOME=/home/npatri/rq2_visual_faithfulness/.cache
export PIP_CACHE_DIR=/home/npatri/rq2_visual_faithfulness/.cache/pip
export CONDA_PKGS_DIRS=/home/npatri/rq2_visual_faithfulness/.cache/conda-pkgs
export TMPDIR=/home/npatri/rq2_visual_faithfulness/.cache/tmp
cd "$RQ2_PROJECT_DIR" || return 1
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$TORCH_HOME" "$PIP_CACHE_DIR" \
  "$CONDA_PKGS_DIRS" "$TMPDIR" "$RQ2_PROJECT_DIR/results" /home/npatri/slurm_logs
