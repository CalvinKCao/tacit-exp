# Preflight

Run on 2026-11-02 from the local WSL box (no GPU here). GPU work goes to Killarney (L40S, 48 GB, account aip-boyuwang). Nothing below touched a GPU, and no secrets were printed. `HF_TOKEN` is not set locally.

## Summary

| # | Item | Intended source | How verified | Status |
|---|------|-----------------|--------------|--------|
| 1 | Compute (GPU, VRAM, disk, RAM, CUDA, SDPA) | Killarney L40S | not reachable from here; `scripts/slurm_preflight.sh` written | NOT CHECKED, needs cluster (action 1) |
| 2 | Network: github.com, huggingface.co, arxiv.org, pypi.org | local HEAD requests | all returned HTTP 200 locally; compute-node access unknown (Alliance nodes usually have none) | AVAILABLE locally, UNKNOWN on compute nodes |
| 3 | Paper PDF | `<PDF_PATH>` was never filled in, so I fetched arxiv.org/pdf/2609.33039 | HTTP 200, 602,075 bytes, sha256 `8dee899c322f47784566f4c0e17a7602e8f53d9732a574f9b9f577ea02ab8109`, saved to `paper/tacit_2609.33039.pdf` | AVAILABLE |
| 4 | Tacit repo | github.com/CSSLab/Tacit | `git ls-remote` + shallow clone to `repos/Tacit`, commit `995692d3cd42c32e08cf0db8a5fae55fec272f1a` (2026-09-23) | AVAILABLE |
| 5a | Qwen3-4B-Instruct-2507 (the repo's checkpoint) | HF `Qwen/Qwen3-4B-Instruct-2507` | HF API: not gated, apache-2.0, 13 files | AVAILABLE |
| 5b | Qwen3-4B (hybrid thinking model) | HF `Qwen/Qwen3-4B` | HF API: not gated, apache-2.0 | AVAILABLE |
| 5c | Qwen3Guard-Gen-4B | HF `Qwen/Qwen3Guard-Gen-4B` | HF API: not gated, apache-2.0 | AVAILABLE |
| 5d | AgentDoG-Qwen3-4B | HF `AI45Research/AgentDoG-Qwen3-4B` | HF API: not gated, apache-2.0, 22 files | AVAILABLE |
| 5e | Llama-Guard-3-8B | HF `meta-llama/Llama-Guard-3-8B` | HF API: gated = manual, llama3.1 license | GATED (action 3) |
| 5f | Llama-3.1-8B-Instruct (optional) | HF `meta-llama/Llama-3.1-8B-Instruct` | HF API: gated = manual, llama3.1 license | GATED (action 3) |
| 6a | TraceSafe | github.com/cycraft-corp/TraceSafe `data/` (the HF copy `CyCraftAI/TraceSafe` is gated = auto) | all 13 `golden_*.jsonl` downloaded, 90 records each (1,170 total, matching the repo README). Tacit loader gives n=2250, unsafe=1080, matching `data/README.md` | AVAILABLE |
| 6b | HAICOSYSTEM | HF `Xuhui/haicosystem` (not gated) | `all_episodes.jsonl` (315,534,179 bytes, 7,940 lines) and `all_environments.jsonl` (132 lines). Tacit loader parses 7,937 episodes with rewards; the pair builder gives 258 content pairs | AVAILABLE |
| 6c | Other data the Tacit scripts reference (R-Judge, ATBench, ASSEBench, OpenAgentSafety, AgentDojo) | listed in `repos/Tacit/data/README.md` | repos and HF dataset reachable (ATBench not gated, ASSEBench and OAS `git ls-remote` ok); not downloaded since this study does not use them | AVAILABLE, not fetched, not needed |
| 7 | Python dependencies | `repos/Tacit/requirements.txt` | pins exist on PyPI; see below | see below |

## Findings that matter for the plan

1. **Checkpoint ambiguity.** `tacit/config.py` maps the key `qwen3-4b-instruct` to `Qwen/Qwen3-4B-Instruct-2507`. That is the only Qwen backbone the repo extracts with. `Qwen/Qwen3-4B` is a different model (hybrid thinking). I will use Instruct-2507 as primary since it matches the released code; it is logged in QUESTIONS.md. Using plain Qwen3-4B as well would cost about 10 minutes of GPU for Step 1 only, so say if you want it.
2. **No pretrained readouts ship with Tacit.** I searched for `.pt/.pkl/.npz/.npy/.joblib/.safetensors` and found none. Only frozen splits (`data/splits/split_seed4{2..5}.json`) and AgentDojo trajectories ship. So readout (2), TACIT's released readout zero-shot, will be reported as SKIPPED.
3. **Shipped code that I reuse.** `tacit/activations/extract.py` (pools `mean` and `last`, all layers, float16, `max_length=16384`, head/tail truncation in `modeling.encode_truncated`), `scripts/data/build_haico_pairs.py` and `scripts/data/build_tracesafe_placebo.py`. Both pair builders ran locally: HAICOSYSTEM content pairs = 258 (median Jaccard 0.591), TraceSafe placebo records = 180. Both match the paper counts.
4. **TraceSafe pair structure.** Each record in a category file has `original_trace` (the benign member) and `new_trace` (the mutated one), plus `difference` and `mutation_metadata`. MissingTypeHint and AmbiguousArg have 90 records each, so 180 pairs. I did not use the user_query match across files, because that does not line up with the benign file.
5. **Source of TraceSafe.** The HF copy is gated (auto-approval, needs a token). The official GitHub repo carries the same 13 files in `data/`, so I took them from there. Say so if you want the HF copy compared by hash.
6. **Dependency pins.** Tacit pins torch 2.11.0, transformers 5.9.0, huggingface-hub 1.15.0, accelerate 1.9.0, tokenizers 0.22.2 and vllm 0.21.0, all of which exist on PyPI. The Alliance wheelhouse may not carry transformers 5.9.0 or vllm; the preflight job prints what is there. Guards run through vllm in the repo; I will fall back to plain transformers generate for the three guards if vllm is not installable, and log that as a deviation.
7. **Local venv.** `.venv` at the repo root holds numpy 2.2.6, scipy, scikit-learn and huggingface_hub 1.6.0 (enough to run the data builders). No torch locally.

## Gating rule

Tacit repo, Qwen3-4B (Instruct-2507), TraceSafe and HAICOSYSTEM are AVAILABLE, so I do not have to stop. I am stopping anyway for one reason: item 1 (GPU, SDPA, venv on compute nodes, and whether the compute node can reach HF) has to be checked on the cluster, and the model weights have to be cached on the login node before any extraction job can run. Only Llama-Guard-3-8B and Llama-3.1-8B are gated; those analyses are SKIPPED-PENDING until access is granted.

## ACTION FOR USER

### Action 1: run the cluster preflight (GPU, VRAM, SDPA, wheelhouse, compute-node network)

Needs the project on the cluster first. I could not find the Killarney login hostname in the workspace, so the rsync line is commented out; fill in your own. About 330 MB of raw data is regenerated on the cluster in action 2, so only the small code folders need copying.

```bash
# from the local box, from /home/cao/interp (host name is yours to fill in)
# rsync -av --exclude .venv --exclude temp --exclude 'repos/Tacit/data/raw' --exclude 'repos/Tacit/data/agentdojo' ./ ccao87@<killarney-login-host>:/scratch/ccao87/interp/

# on the cluster
cd "$SCRATCH/interp"
sbatch scripts/slurm_preflight.sh
# squeue -u "$USER"
# tail -f results/logs/*preflight.log
```

Verification to paste back: the end of `results/logs/*-preflight.log` and the file `results/logs/preflight_gpu.json`. It should show gpu = NVIDIA L40S, vram_gb about 44.5, `cuda_available` true, `sdpa_ok` true, scratch free of at least 60 GB, and the reachability lines for github, huggingface and pypi. Without this I cannot confirm the 60 GB disk and RAM requirements; I can still prepare all code and data builders.

### Action 2: fetch models and data on the login node (login nodes have internet, compute nodes usually do not)

About 32 GB for the four open models (Qwen3-4B-Instruct-2507 8 GB, Qwen3Guard-Gen-4B 8 GB, AgentDoG-Qwen3-4B 8 GB, optional plain Qwen3-4B 8 GB), 330 MB of data.

```bash
cd "$SCRATCH/interp"
export HF_HOME="$SCRATCH/hf_cache"
export TACIT_DATA="$SCRATCH/interp/repos/Tacit/data/raw"

# one time: python env on the login node only for downloading
module purge || true
module load StdEnv/2023 python/3.11
python -m venv "$SCRATCH/dl_env" && source "$SCRATCH/dl_env/bin/activate"
pip install -q huggingface_hub

# repo, if the rsync above did not bring it
# git clone https://github.com/CSSLab/Tacit repos/Tacit && git -C repos/Tacit checkout 995692d3cd42c32e08cf0db8a5fae55fec272f1a

# models
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
huggingface-cli download Qwen/Qwen3Guard-Gen-4B
huggingface-cli download AI45Research/AgentDoG-Qwen3-4B
# huggingface-cli download Qwen/Qwen3-4B

# data
mkdir -p "$TACIT_DATA/TraceSafe/data" "$TACIT_DATA/HAICOSYSTEM"
for n in 0_benign 6_AmbiguousArg 10_MissingTypeHint; do
  curl -sLO --output-dir "$TACIT_DATA/TraceSafe/data" https://raw.githubusercontent.com/cycraft-corp/TraceSafe/main/data/golden_$n.jsonl
done
for f in all_episodes.jsonl all_environments.jsonl; do
  curl -sLO --output-dir "$TACIT_DATA/HAICOSYSTEM" https://huggingface.co/datasets/Xuhui/haicosystem/resolve/main/$f
done

# checks to paste back
wc -l "$TACIT_DATA"/TraceSafe/data/*.jsonl
wc -l "$TACIT_DATA"/HAICOSYSTEM/*.jsonl
sha256sum "$TACIT_DATA"/HAICOSYSTEM/all_episodes.jsonl
du -sh "$HF_HOME"
```

Expected: 90 lines in each of the three TraceSafe files, 7940 and 132 lines for the two HAICOSYSTEM files, `all_episodes.jsonl` of 315,534,179 bytes (`ls -l`). The three TraceSafe files above are all that Step 1 needs; if you want the other ten for the loader count check, loop over 1..12 as well.

### Action 3: Llama access (only needed for the Llama-Guard-3-8B text guard and the optional Llama-3.1-8B replication)

Request access at https://huggingface.co/meta-llama/Llama-Guard-3-8B and https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (license: Llama 3.1 community license, manual approval). Create a read token at https://huggingface.co/settings/tokens and set it on the login node.

```bash
# export HF_TOKEN=<your token>   # set it in your shell, do not paste it into files or chat
# huggingface-cli download meta-llama/Llama-Guard-3-8B
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
# du -sh "$HF_HOME"
```

About 16 GB each. Verification: `huggingface-cli scan-cache | grep -i llama`. Without this, readout (6) is run on Qwen3Guard and AgentDoG only, and Llama-Guard rows plus the Llama replication are marked SKIPPED-PENDING.

### What I can do without any of this

Write PLAN.md, build the 2x2 compositional-harm generator and its tests (pure Python, local), run the TF-IDF and tree leakage checks locally, and write all extraction and analysis code. Everything that needs the model needs actions 1 and 2.
