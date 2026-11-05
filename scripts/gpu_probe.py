import json
import os
import shutil
import sys

import torch

out = {}
out["torch"] = torch.__version__
out["cuda_runtime"] = torch.version.cuda
out["cuda_available"] = torch.cuda.is_available()
assert out["cuda_available"], "no cuda, stopping"

props = torch.cuda.get_device_properties(0)
out["gpu"] = props.name
out["vram_gb"] = round(props.total_memory / 2**30, 1)
out["bf16_supported"] = torch.cuda.is_bf16_supported()

q = torch.randn(1, 8, 2048, 64, device="cuda", dtype=torch.bfloat16)
y = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
out["sdpa_ok"] = bool(torch.isfinite(y).all())

try:
    import flash_attn
    out["flash_attn"] = flash_attn.__version__
except Exception as e:
    out["flash_attn"] = "missing: " + type(e).__name__

for name in ["transformers", "huggingface_hub", "tokenizers", "accelerate", "safetensors", "numpy", "scipy", "sklearn", "joblib"]:
    try:
        m = __import__(name)
        out[name] = getattr(m, "__version__", "?")
    except Exception as e:
        out[name] = "missing: " + type(e).__name__

scratch = os.environ.get("SCRATCH", ".")
out["scratch_free_gb"] = round(shutil.disk_usage(scratch).free / 2**30, 1)
out["tmpdir_free_gb"] = round(shutil.disk_usage(os.environ.get("SLURM_TMPDIR", "/tmp")).free / 2**30, 1)
out["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)

print(json.dumps(out, indent=2))
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=2)
