import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.environ.get("TACIT_REPO", "repos/Tacit"))
from tacit.utils import modeling
from tacit.data import tracesafe, tracesafe_placebo, haico_pairs

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAYERS = list(range(8, 31))
PAIR_CATS = ("MissingTypeHint", "AmbiguousArg")


def pick_tracesafe_pairs():
    out = []
    for t in tracesafe.load():
        cat = t.uid.removesuffix("-orig").split("tracesafe-golden_", 1)[1].split("_", 1)[1].rsplit("-", 1)[0]
        if cat in PAIR_CATS:
            out.append(t)
    return out


def load_traces(name):
    if name == "tracesafe-pairs":
        return pick_tracesafe_pairs()
    if name == "tracesafe-placebo":
        return tracesafe_placebo.load()
    if name == "haico-pairs":
        return haico_pairs.load()
    if name == "jsonl":
        raise ValueError("use extract_texts for jsonl input")
    raise ValueError(name)


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


def run(texts, labels, uids, out_path, max_length=16384, budget=20000, max_bs=16, limit=None):
    if limit:
        texts, labels, uids = texts[:limit], labels[:limit], uids[:limit]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    ids_list, trunc = [], []
    for t in texts:
        ids, tr = modeling.encode_truncated(tok, t, max_length)
        ids_list.append(ids)
        trunc.append(bool(tr))
    lengths = [int(x.shape[0]) for x in ids_list]
    print(f"{len(texts)} traces, tokens min/median/max = {min(lengths)}/{sorted(lengths)[len(lengths)//2]}/{max(lengths)}, truncated {sum(trunc)}", flush=True)

    n, h = len(texts), model.config.hidden_size
    last = torch.zeros(n, len(LAYERS), h, dtype=torch.float16)
    mean = torch.zeros(n, len(LAYERS), h, dtype=torch.float16)

    batches = make_batches(lengths, budget, max_bs)
    t0 = time.time()
    done = 0
    for b in batches:
        T = max(lengths[i] for i in b)
        inp = torch.full((len(b), T), pad_id, dtype=torch.long)
        mask = torch.zeros(len(b), T, dtype=torch.long)
        for r, i in enumerate(b):
            inp[r, :lengths[i]] = ids_list[i]
            mask[r, :lengths[i]] = 1
        with torch.no_grad():
            out = model(input_ids=inp.cuda(), attention_mask=mask.cuda(), output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        m = mask.cuda().unsqueeze(-1).to(torch.float32)
        for k, l in enumerate(LAYERS):
            x = hs[l]
            for r, i in enumerate(b):
                last[i, k] = x[r, lengths[i] - 1].to(torch.float16).cpu()
            mean[b, k] = ((x.float() * m).sum(1) / m.sum(1)).to(torch.float16).cpu()
        done += len(b)
        print(f"  {done}/{n} {time.time()-t0:.0f}s", flush=True)

    meta = {"model": MODEL_ID, "layers": LAYERS, "max_length": max_length, "dtype": "bfloat16",
            "attn": "sdpa", "torch": str(torch.__version__), "hidden_states_index": "0=embeddings, l=output of block l"}
    torch.save({"last": last, "mean": mean, "labels": torch.tensor(labels), "uids": uids,
                "n_tokens": torch.tensor(lengths), "truncated": torch.tensor(trunc), "meta": meta}, out_path)
    print("saved", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=int, default=20000)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    traces = load_traces(a.dataset)
    texts = [modeling.render(tok, t, rich_tools=False) for t in traces]
    run(texts, [t.label for t in traces], [t.uid for t in traces], a.out, limit=a.limit, budget=a.budget)


if __name__ == "__main__":
    main()
