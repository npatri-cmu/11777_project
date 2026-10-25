#!/bin/bash
#SBATCH --job-name=rq2
#SBATCH --partition=brisket
#SBATCH --qos=brisket_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/npatri/rq2_visual_faithfulness
#SBATCH --output=/home/npatri/slurm_logs/rq2_%j.out
#SBATCH --error=/home/npatri/slurm_logs/rq2_%j.err
set -euo pipefail
source /home/npatri/rq2_visual_faithfulness/scripts/hpc_env.sh
# Use the existing Conda installation, including in non-interactive shells.
RQ2_CONDA_EXE="${CONDA_EXE:-conda}"
command -v "$RQ2_CONDA_EXE" >/dev/null || { echo 'Conda is unavailable; submit from your Conda-enabled shell.' >&2; exit 1; }
source "$("$RQ2_CONDA_EXE" info --base)/etc/profile.d/conda.sh"
conda activate "$RQ2_ENV_PREFIX"
export OMP_NUM_THREADS=8
PHASE="${1:?Pass smoke, dev, or eval}"
case "$PHASE" in
  smoke|dev|eval) ;;
  *) echo 'Phase must be smoke, dev, or eval'; exit 2 ;;
esac
python -m scripts.gpu_preflight
if [[ "$PHASE" == smoke ]]; then
  python -m scripts.smoke_test --gpu
else
  # Leave five minutes before Slurm's twelve-hour hard limit. Resubmit to resume.
  python -m scripts.run_vcd --split "$PHASE" --max-seconds "${MAX_SECONDS:-42900}"
fi
