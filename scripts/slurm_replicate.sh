#!/bin/bash
#SBATCH --job-name=tacit-replicate
#SBATCH --account=aip-boyuwang
#SBATCH --time=0:07:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/step1
LOG=results/logs/$(date +%m-%d)-${SLURM_JOB_ID: -3}-replicate.log
exec >>"$LOG" 2>&1
echo "job $SLURM_JOB_ID start $(date)"

module purge || true
module load StdEnv/2023
module load python/3.11

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index 'torch==2.11.0+computecanada' numpy scipy scikit-learn -q

export PYTHONUNBUFFERED=1
python -u src/replicate.py --acts results/acts --derived data/tacit_runs/derived --out results/step1
echo "end $(date)"
