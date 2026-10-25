#!/usr/bin/env bash
set -euo pipefail
source /home/npatri/rq2_visual_faithfulness/scripts/hpc_env.sh
# Use the existing Conda installation, including in non-interactive shells.
RQ2_CONDA_EXE="${CONDA_EXE:-conda}"
command -v "$RQ2_CONDA_EXE" >/dev/null || { echo 'Conda is unavailable; submit from your Conda-enabled shell.' >&2; exit 1; }
source "$("$RQ2_CONDA_EXE" info --base)/etc/profile.d/conda.sh"
if [[ ! -d "$RQ2_ENV_PREFIX/conda-meta" ]]; then
  conda create --yes --prefix "$RQ2_ENV_PREFIX" python=3.11 pip
fi
conda activate "$RQ2_ENV_PREFIX"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "This environment must use Python 3.11"'
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-hpc.lock
mkdir -p results
python -m pip freeze > results/environment-hpc.txt
conda list --explicit > results/conda-hpc-explicit.txt
python -m pip check
python -m pytest -q
