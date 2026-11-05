import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def make_batches(lengths, budget, max_bs):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        longest = max([lengths[j] for j in cur] + [lengths[i]])
        if cur and (longest * (len(cur) + 1) > budget or len(cur) >= max_bs):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def encode(tok, text, spans, max_length):
    enc = tok(text, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    n = len(ids)
    keep = list(range(n))
    truncated = n > max_length
    if truncated:
        head = max_length // 2
        keep = list(range(head)) + list(range(n - (max_length - head), n))
    new_pos = {old: new for new, old in enumerate(keep)}
    out = {}
    arr = np.array(offs).reshape(-1, 2)
    for name, (s, e) in spans.items():
        toks = np.nonzero((arr[:, 1] > s) & (arr[:, 0] < e))[0].tolist()
        kept = [new_pos[i] for i in toks if i in new_pos]
        out[name] = {"tokens": kept, "n_total": len(toks), "fully_kept": len(kept) == len(toks) and len(toks) > 0}
    return [ids[i] for i in keep], truncated, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="8-30")
    ap.add_argument("--span-pool", default="b_action")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.layers.split("-")]
    layers = list(range(lo, hi + 1))

    items = [json.loads(l) for l in open(a.input)]
    if a.limit:
        items = items[:a.limit]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    enc = [encode(tok, it["text"], it.get("spans", {}), a.max_length) for it in items]
    lengths = [len(e[0]) for e in enc]
    print(f"{len(items)} items, tokens min/median/max {min(lengths)}/{sorted(lengths)[len(lengths)//2]}/{max(lengths)}, total {sum(lengths)}, truncated {sum(e[1] for e in enc)}", flush=True)

    n, h, L = len(items), model.config.hidden_size, len(layers)
    res = {k: torch.zeros(n, L, h, dtype=torch.float16) for k in ("last", "mean", "span_last", "span_mean")}
    span_ok = torch.zeros(n, dtype=torch.bool)
    t0, done = time.time(), 0
    for b in make_batches(lengths, a.budget, 16):
        T = max(lengths[i] for i in b)
        inp = torch.full((len(b), T), pad_id, dtype=torch.long)
        mask = torch.zeros(len(b), T, dtype=torch.long)
        for r, i in enumerate(b):
            inp[r, :lengths[i]] = torch.tensor(enc[i][0])
            mask[r, :lengths[i]] = 1
        with torch.no_grad():
            out = model(input_ids=inp.cuda(), attention_mask=mask.cuda(), output_hidden_states=True, use_cache=False)
        m = mask.cuda().unsqueeze(-1).to(torch.float32)
        for k, l in enumerate(layers):
            x = out.hidden_states[l]
            res["mean"][b, k] = ((x.float() * m).sum(1) / m.sum(1)).to(torch.float16).cpu()
            for r, i in enumerate(b):
                res["last"][i, k] = x[r, lengths[i] - 1].to(torch.float16).cpu()
                sp = enc[i][2].get(a.span_pool)
                if sp and sp["tokens"]:
                    idx = torch.tensor(sp["tokens"], device=x.device)
                    xs = x[r, idx]
                    res["span_mean"][i, k] = xs.float().mean(0).to(torch.float16).cpu()
                    res["span_last"][i, k] = xs[-1].to(torch.float16).cpu()
                    span_ok[i] = True
        done += len(b)
        if done % 100 < len(b):
            print(f"  {done}/{n} {time.time()-t0:.0f}s", flush=True)

    flags = {name: [bool(e[2][name]["fully_kept"]) for e in enc] for name in (enc[0][2].keys() if enc[0][2] else [])}
    meta = {"model": MODEL_ID, "layers": layers, "max_length": a.max_length, "dtype": "bfloat16", "attn": "sdpa", "torch": str(torch.__version__)}
    torch.save({**res, "ids": [it["id"] for it in items], "n_tokens": torch.tensor(lengths), "truncated": torch.tensor([e[1] for e in enc]),
                "span_flags": flags, "span_ok": span_ok, "meta": meta}, a.out)
    print("saved", a.out, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
