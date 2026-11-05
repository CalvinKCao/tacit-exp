#!/bin/bash
#SBATCH --job-name=tacit-step2
#SBATCH --account=aip-boyuwang
#SBATCH --time=0:21:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/acts results/step2
LOG=results/logs/$(date +%m-%d)-${SLURM_JOB_ID: -3}-step2.log
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
pip freeze > results/logs/step2_pip_freeze.txt

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TACIT_REPO=$PWD/repos/Tacit
export PYTHONPATH=$PWD/src:$TACIT_REPO

python -u src/extract2.py --input data/k2/joint.jsonl --out results/acts/k2_joint.pt
python -u src/extract2.py --input data/k2/views.jsonl --out results/acts/k2_views.pt
for k in qwen3guard-4b agentdog; do
    python -u src/guards.py --kind $k --out results/step2/guards_$k.json
done
echo "end $(date)"
