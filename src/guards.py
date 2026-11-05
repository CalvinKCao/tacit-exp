import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.environ.get("TACIT_REPO", "repos/Tacit"))
import compo
from tacit.guards import moderation as M
from tacit.guards import agentdog as AD

IDS = {"qwen3guard-4b": "Qwen/Qwen3Guard-Gen-4B", "agentdog": "AI45Research/AgentDoG-Qwen3-4B"}


def branch_ids(tok):
    ids = {"safe": [], "unsafe": []}
    for i in range(len(tok)):
        b = M._branch(tok.decode([i]))
        if b:
            ids[b].append(i)
    return {k: torch.tensor(v) for k, v in ids.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--input", default="data/k2/joint.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12288)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input)]
    rows = [r for r in rows if r["split"] == a.split]
    tok = AutoTokenizer.from_pretrained(IDS[a.kind])
    model = AutoModelForCausalLM.from_pretrained(IDS[a.kind], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    bid = branch_ids(tok)
    print(a.kind, len(rows), "traces", flush=True)

    out = []
    for n, r in enumerate(rows):
        fam, inst = r["scen"].rsplit("-", 1)
        sc = compo.scenario(fam, int(inst))
        auth, act, _ = compo.CELLS[r["cell"]]
        trace = compo.to_trace(sc, compo.turns_joint(sc, auth, act), uid=r["id"])
        if a.kind == "agentdog":
            prompt = AD.build_prompt(tok, trace, a.max_len)[0]
        else:
            prompt = M.make_prompt(trace, tok, a.max_len - 2048)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=16, do_sample=False, output_scores=True, return_dict_in_generate=True)
        gen = g.sequences[0, enc.input_ids.shape[1]:].tolist()
        texts = [tok.decode([t]) for t in gen]
        resp = tok.decode(gen)
        pred = M.parse_verdict("qwen" if a.kind == "qwen3guard-4b" else "agentdog", resp) if a.kind == "qwen3guard-4b" else int(resp.strip().lower().startswith("unsafe"))
        if a.kind == "qwen3guard-4b":
            idx = M._decisive_index("qwen", texts)
        else:
            idx = next((i for i, t in enumerate(texts) if M._branch(t)), None)
        score = float(pred)
        if idx is not None:
            p = torch.softmax(g.scores[idx][0].float(), -1).cpu()
            u, s = float(p[bid["unsafe"]].sum()), float(p[bid["safe"]].sum())
            if u + s > 0:
                score = u / (u + s)
        out.append({"id": r["id"], "label": r["label"], "cell": r["cell"], "pred": int(pred), "score": score, "found": idx is not None, "resp": resp[:60]})
        if n % 40 == 0:
            print(n, r["id"], pred, round(score, 3), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
