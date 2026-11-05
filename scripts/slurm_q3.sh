#!/bin/bash
#SBATCH --job-name=tacit-q3
#SBATCH --account=aip-boyuwang
#SBATCH --time=1:24:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/acts
LOG=results/logs/$(date +%m-%d)-${SLURM_JOB_ID: -3}-q3.log
exec >>"$LOG" 2>&1
echo "job $SLURM_JOB_ID node $SLURMD_NODENAME start $(date)"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index 'torch==2.11.0+computecanada' numpy scipy scikit-learn joblib transformers accelerate tokenizers safetensors huggingface_hub -q
python -c "import torch; assert torch.cuda.is_available(), 'no cuda'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/src

python -u src/extract2.py --input data/q3/views.jsonl --out results/acts/q3_views.pt --layers 12-28
python -u src/extract2.py --input data/q3/joint.jsonl --out results/acts/q3_joint.pt --layers 12-28
echo "end $(date)"
