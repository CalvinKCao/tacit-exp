import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

CATS = {"MissingTypeHint", "AmbiguousArg"}
SEED = 2026


def category(uid):
    uid = uid.removesuffix("-orig").removesuffix("-placebo")
    return uid.split("tracesafe-golden_", 1)[1].split("_", 1)[1].rsplit("-", 1)[0]


def rank_acc(sa, sb):
    v = np.sign(np.asarray(sa, float) - np.asarray(sb, float))
    return float((v > 0).mean() + 0.5 * (v == 0).mean())


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def boot_ci(d, rng, n=2000):
    d = np.asarray(d)
    s = np.sign(d)
    accs = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        x = s[idx]
        accs.append((x > 0).mean() + 0.5 * (x == 0).mean())
    return [float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))]


def oof_scores(H, A, B, folds):
    out = np.zeros(len(A))
    for tr, te in folds:
        v = unit((H[A[tr]] - H[B[tr]]).mean(0))
        out[te] = H[A[te]] @ v - H[B[te]] @ v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="results/acts")
    ap.add_argument("--derived", required=True)
    ap.add_argument("--out", default="results/step1")
    ap.add_argument("--pool", default="last")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ts = torch.load(f"{a.acts}/tracesafe-pairs.pt", weights_only=False)
    hp = torch.load(f"{a.acts}/haico-pairs.pt", weights_only=False)
    pl = torch.load(f"{a.acts}/tracesafe-placebo.pt", weights_only=False)
    layers = ts["meta"]["layers"]
    Xt = ts[a.pool].float().numpy()
    Xh = hp[a.pool].float().numpy()
    Xp = pl[a.pool].float().numpy()
    tu = {u: i for i, u in enumerate(ts["uids"])}
    hu = {u: i for i, u in enumerate(hp["uids"])}
    pu = {u: i for i, u in enumerate(pl["uids"])}
    man = json.load(open(f"{a.derived}/haico_pairs_manifest.json"))

    ctx = [(tu[u], tu[u + "-orig"]) for u in ts["uids"] if not u.endswith("-orig") and u + "-orig" in tu and category(u) in CATS]
    arms = {"context": ("ts", np.array([x for x, _ in ctx]), np.array([y for _, y in ctx]))}
    for key, sel in (("content_strict", "content"), ("content_broad", "any")):
        ok = [p for p in man[sel] if p["unsafe"] in hu and p["safe"] in hu]
        arms[key] = ("hp", np.array([hu[p["unsafe"]] for p in ok]), np.array([hu[p["safe"]] for p in ok]))
    plc = []
    for u in ts["uids"]:
        if u.endswith("-orig") or u + "-orig" not in tu or category(u) not in CATS:
            continue
        if u + "-placebo" in pu:
            plc.append((tu[u], tu[u + "-orig"], pu[u + "-placebo"]))
    PA, PB, PQ = (np.array([r[i] for r in plc]) for i in range(3))
    print({k: len(v[1]) for k, v in arms.items()}, "placebo aligned", len(plc), flush=True)
    assert len(arms["context"][1]) == 180 and len(arms["content_strict"][1]) == 258, "pair counts off"

    split_rng = np.random.default_rng(31415)
    splits = {}
    for k in ("context", "content_strict"):
        n = len(arms[k][1])
        splits[k] = []
        for _ in range(5):
            idx = split_rng.permutation(n)
            splits[k].append((idx[:n // 2], idx[n // 2:]))
    boot_rng = np.random.default_rng(SEED + 1)
    nb = 500
    boot_w = {k: boot_rng.poisson(1.0, size=(nb, len(arms[k][1]))).astype(np.float32) for k in ("context", "content_strict")}

    res = {"layers": layers, "n": {k: len(v[1]) for k, v in arms.items()}, "n_placebo": len(plc), "per_layer": {}}
    dirs_save = {}
    for k, l in enumerate(layers):
        a_ = Xt[:, k, :].astype(np.float64)
        b_ = Xh[:, k, :].astype(np.float64)
        sd = np.concatenate([a_, b_], 0).std(0) + 1e-6
        H = {"ts": a_ / sd, "hp": b_ / sd}
        G = Xp[:, k, :].astype(np.float64) / sd
        row = {"oof": {}, "oof_ci": {}, "split_half": {}, "cos": {}}
        dirs, diffs, bdirs = {}, {}, {}
        for name, (src, A, B) in arms.items():
            folds = list(GroupKFold(5).split(np.arange(len(A)), groups=np.arange(len(A))))
            d = oof_scores(H[src], A, B, folds)
            row["oof"][name] = rank_acc(d, np.zeros(len(d)))
            row["oof_ci"][name] = boot_ci(d, rng, 500)
            D = (H[src][A] - H[src][B]).astype(np.float32)
            diffs[name] = D
            dirs[name] = unit(D.mean(0))
            if name in boot_w:
                bd = boot_w[name] @ D
                bdirs[name] = bd / (np.linalg.norm(bd, axis=1, keepdims=True) + 1e-12)
        for name in ("context", "content_strict"):
            row["split_half"][name] = float(np.mean([unit(diffs[name][i].mean(0)) @ unit(diffs[name][j].mean(0)) for i, j in splits[name]]))
        row["cos"]["tool|content"] = float(dirs["context"] @ dirs["content_strict"])
        bc = np.sum(bdirs["context"] * bdirs["content_strict"], axis=1)
        row["cos"]["tool|content_ci"] = [float(np.quantile(bc, 0.025)), float(np.quantile(bc, 0.975))]
        row["transfer"] = {}
        for kd, v in dirs.items():
            for kt, (src, A, B) in arms.items():
                row["transfer"][f"{kd}->{kt}"] = rank_acc(H[src][A] @ v, H[src][B] @ v)

        folds = list(GroupKFold(5).split(np.arange(len(PA)), groups=np.arange(len(PA))))
        sa, sb, sq = np.zeros(len(PA)), np.zeros(len(PA)), np.zeros(len(PA))
        for tr, te in folds:
            v = unit((H["ts"][PA[tr]] - H["ts"][PB[tr]]).mean(0))
            sa[te], sb[te], sq[te] = H["ts"][PA[te]] @ v, H["ts"][PB[te]] @ v, G[PQ[te]] @ v
        row["placebo"] = {"unsafe_vs_benign": rank_acc(sa, sb), "placebo_vs_benign": rank_acc(sq, sb), "unsafe_vs_placebo": rank_acc(sa, sq)}
        row["placebo_ci"] = boot_ci(sq - sb, rng, 500)
        res["per_layer"][l] = row
        dirs_save[f"v_tool_L{l}"] = dirs["context"].astype(np.float32)
        dirs_save[f"v_content_L{l}"] = dirs["content_strict"].astype(np.float32)
        dirs_save[f"sd_L{l}"] = sd.astype(np.float32)
        print(f"L{l} tool {row['oof']['context']:.3f} content {row['oof']['content_strict']:.3f} placebo {row['placebo']['placebo_vs_benign']:.3f} cos {row['cos']['tool|content']:+.3f}", flush=True)

    pl_ = res["per_layer"]
    tool_peak = max(layers, key=lambda l: pl_[l]["oof"]["context"])
    cont_peak = max(layers, key=lambda l: pl_[l]["oof"]["content_strict"])
    p_t, p_c = pl_[tool_peak], pl_[cont_peak]
    checks = {
        "tool_peak_acc": (p_t["oof"]["context"], 0.93 <= p_t["oof"]["context"] <= 1.0),
        "tool_peak_layer_intermediate": (tool_peak, 10 <= tool_peak <= 26),
        "placebo_at_tool_peak": (p_t["placebo"]["placebo_vs_benign"], 0.49 <= p_t["placebo"]["placebo_vs_benign"] <= 0.59),
        "content_peak_acc": (p_c["oof"]["content_strict"], 0.68 <= p_c["oof"]["content_strict"] <= 0.79),
        "abs_cos_at_tool_peak": (p_t["cos"]["tool|content"], abs(p_t["cos"]["tool|content"]) < 0.2),
        "abs_cos_at_content_peak": (p_c["cos"]["tool|content"], abs(p_c["cos"]["tool|content"]) < 0.2),
        "split_half_tool_at_own_peak": (p_t["split_half"]["context"], 0.76 <= p_t["split_half"]["context"] <= 0.86),
        "split_half_content_at_own_peak": (p_c["split_half"]["content_strict"], 0.76 <= p_c["split_half"]["content_strict"] <= 0.86),
    }
    checks["split_half_any_in_range"] = (None, checks["split_half_tool_at_own_peak"][1] or checks["split_half_content_at_own_peak"][1])
    res["peaks"] = {"tool": tool_peak, "content": cont_peak}
    res["checks"] = {k: {"value": v[0], "pass": bool(v[1])} for k, v in checks.items()}
    res["layer22"] = pl_[22]

    json.dump(res, open(f"{a.out}/replication.json", "w"), indent=1)
    np.savez(f"{a.out}/directions.npz", **dirs_save)
    with open(f"{a.out}/curves.csv", "w") as f:
        f.write("layer,tool_oof,content_oof,content_broad_oof,placebo_vs_benign,cos_tool_content,split_tool,split_content\n")
        for l in layers:
            r = pl_[l]
            f.write(f"{l},{r['oof']['context']:.4f},{r['oof']['content_strict']:.4f},{r['oof']['content_broad']:.4f},{r['placebo']['placebo_vs_benign']:.4f},{r['cos']['tool|content']:.4f},{r['split_half']['context']:.4f},{r['split_half']['content_strict']:.4f}\n")
    print("peaks", res["peaks"])
    for k, v in res["checks"].items():
        print("PASS" if v["pass"] else "FAIL", k, v["value"])


if __name__ == "__main__":
    main()
