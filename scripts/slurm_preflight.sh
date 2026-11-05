#!/bin/bash
#SBATCH --job-name=tacit-preflight
#SBATCH --account=aip-boyuwang
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs
LOG=results/logs/$(date +%m-%d)-${SLURM_JOB_ID: -3}-preflight.log
exec >>"$LOG" 2>&1

echo "job $SLURM_JOB_ID on $SLURMD_NODENAME"
nvidia-smi -L

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index 'torch==2.11.0+computecanada' numpy scipy scikit-learn joblib -q
pip install --no-index transformers accelerate tokenizers safetensors huggingface_hub -q || echo "some hf wheels missing from the wheelhouse"

for h in github.com huggingface.co pypi.org; do
    if curl -s -m 8 -o /dev/null -I https://$h; then echo "reach $h yes"; else echo "reach $h no"; fi
done

python -u scripts/gpu_probe.py results/logs/preflight_gpu.json
avail_wheels transformers huggingface_hub vllm 2>/dev/null || true
echo done
