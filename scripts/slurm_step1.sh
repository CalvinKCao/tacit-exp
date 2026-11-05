#!/bin/bash
#SBATCH --job-name=tacit-step1
#SBATCH --account=aip-boyuwang
#SBATCH --time=0:25:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/acts results/step1
LOG=results/logs/$(date +%m-%d)-${SLURM_JOB_ID: -3}-step1.log
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
pip freeze > results/logs/step1_pip_freeze.txt

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TACIT_REPO=$PWD/repos/Tacit
export TACIT_DATA=$PWD/repos/Tacit/data/raw
export TACIT_RUNS=$PWD/data/tacit_runs
export PYTHONPATH=$TACIT_REPO
test -f "$TACIT_RUNS/derived/haico_pairs.jsonl" || { echo "missing derived pair files"; exit 1; }

for d in tracesafe-pairs tracesafe-placebo haico-pairs; do
    echo "extract $d $(date)"
    python -u src/extract.py --dataset $d --out results/acts/$d.pt
done

python -u src/replicate.py --acts results/acts --derived $TACIT_RUNS/derived --out results/step1
echo "end $(date)"
