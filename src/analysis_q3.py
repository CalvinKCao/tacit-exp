import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import qlib as Q

SEED = 2026
SETS = ["k2", "k4_start", "k4_mid", "k8_start", "k8_mid", "k16_start", "k16_mid", "k2_pad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="results/acts")
    ap.add_argument("--q2", default="results/step2/q2.json")
    ap.add_argument("--out", default="results/step3")
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sel = json.load(open(a.q2))["selected"]
    pool, layer, C = sel["joint"]
    ppool, player, pC = sel["prefix"]
    early_C, (mh, ma) = sel["early"]["pca_C"], sel["early"]["mlp"]
    print("using joint", sel["joint"], "prefix", sel["prefix"], flush=True)

    rows = []
    for l in open("data/q3/joint.jsonl"):
        r = json.loads(l)
        r.pop("text")
        rows.append(r)
    J = Q.Acts(f"{a.acts}/q3_joint.pt")
    V = Q.Acts(f"{a.acts}/q3_views.pt")
    assert J.ids == [r["id"] for r in rows]
    XJ = J.feat(pool, layer)
    XP = J.feat(ppool, player)
    XV = V.feat(pool, layer)
    flags = J.d["span_flags"]
    by_set = {s: [i for i, r in enumerate(rows) if r["set"] == s] for s in SETS}
    key = lambda i: (rows[i]["scen"], rows[i]["cell"])
    for s in SETS:
        by_set[s].sort(key=key)
    ref = [key(i) for i in by_set["k2"]]
    for s in SETS:
        assert [key(i) for i in by_set[s]] == ref
    scen = np.array([k[0] for k in ref])
    cellv = np.array([k[1] for k in ref])
    fam = np.array([rows[i]["family"] for i in by_set["k2"]])
    spl = np.array([rows[i]["split"] for i in by_set["k2"]])
    y = np.array([rows[i]["label"] for i in by_set["k2"]])
    tr, va, te = (np.nonzero(spl == s)[0] for s in ("train", "val", "test"))
    vte = np.concatenate([va, te])

    vid = lambda name: V.index[name]
    A_idx = np.array([vid(f"q3|{s}|A|{'R' if c[0] == 'R' else 'P'}") for s, c in zip(scen, cellv)])
    B_idx = np.array([vid(f"q3|{s}|B|{'X' if c[1] == 'X' else 'Y'}") for s, c in zip(scen, cellv)])
    filler_rows = [i for i, vid_ in enumerate(V.ids) if vid_.startswith("q3|filler|")]
    filler_pos = {int(V.ids[i].split("|")[2]): i for i in filler_rows}

    A_fit = np.concatenate([A_idx[tr], B_idx[tr], filler_rows])
    yA = np.concatenate([y[tr], y[tr], np.zeros(len(filler_rows), int)])
    probe = Q.fit_lr(XV[A_fit], yA, C)
    s_final = probe(XV)
    s_oof = np.zeros(len(y))
    sB_oof = np.zeros(len(y))
    for ftr, fte in GroupKFold(5).split(tr, groups=scen[tr]):
        rows_tr = tr[ftr]
        f = Q.fit_lr(XV[np.concatenate([A_idx[rows_tr], B_idx[rows_tr], filler_rows])], np.concatenate([y[rows_tr], y[rows_tr], np.zeros(len(filler_rows), int)]), C)
        s_oof[tr[fte]] = f(XV[A_idx[tr[fte]]])
        sB_oof[tr[fte]] = f(XV[B_idx[tr[fte]]])
    sA = s_final[A_idx]
    sB = s_final[B_idx]
    sA_use = sA.copy()
    sB_use = sB.copy()
    sA_use[tr], sB_use[tr] = s_oof[tr], sB_oof[tr]

    scores = {s: {} for s in SETS}
    k2_rows = by_set["k2"]
    k2_joint = Q.fit_lr(XJ[k2_rows][tr], y[tr], C)
    k2_prefix = Q.fit_lr(XP[k2_rows][tr], y[tr], pC)
    for s in SETS:
        rs = by_set[s]
        K = rows[rs[0]]["K"]
        Xs, Ps = XJ[rs], XP[rs]
        scores[s]["joint"] = Q.fit_lr(Xs[tr], y[tr], C)(Xs)
        scores[s]["joint_trained_on_K2"] = k2_joint(Xs)
        scores[s]["prefix_B_action"] = Q.fit_lr(Ps[tr], y[tr], pC)(Ps)
        scores[s]["prefix_trained_on_K2"] = k2_prefix(Ps)
        fl = [[filler_pos[p] for p in rows[i]["filler_ids"]] for i in rs]
        fsc = np.array([[s_final[j] for j in f] for f in fl], dtype=float).reshape(len(rs), K - 2)
        allsc = np.concatenate([sA_use[:, None], fsc, sB_use[:, None]], 1)
        allsc_eval = np.concatenate([sA[:, None], fsc, sB[:, None]], 1)
        scores[s]["late_max"] = allsc_eval.max(1)
        scores[s]["late_mean"] = allsc_eval.mean(1)
        srt = lambda m: -np.sort(-m, axis=1)
        lg = LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000).fit(srt(allsc)[tr], y[tr])
        scores[s]["late_logistic_sorted"] = lg.predict_proba(srt(allsc_eval))[:, 1]
        if K == 2 and s == "k2_pad":
            scores[s]["early_pca50_logistic"] = scores["k2"]["early_pca50_logistic"]
            scores[s]["early_mlp"] = scores["k2"]["early_mlp"]
            continue
        h = 0 if rows[rs[0]]["placement"] == "start" else len(fl[0]) // 2
        order = []
        for n_, f in enumerate(fl):
            chain = [A_idx[n_]] + list(f) + [B_idx[n_]]
            if h:
                chain = list(f[:h]) + [A_idx[n_]] + list(f[h:]) + [B_idx[n_]]
            order.append(chain)
        XE = np.stack([np.concatenate([XV[j] for j in c]) for c in order])
        scores[s]["early_pca50_logistic"] = Q.fit_pca_lr(XE[tr], y[tr], early_C)(XE)
        scores[s]["early_mlp"] = Q.fit_mlp(XE[tr], y[tr], mh, ma)(XE)
        print("done", s, flush=True)

    res = {"settings": {"pool": pool, "layer": layer, "C": C, "prefix": [ppool, player, pC], "early": [early_C, mh, ma]}, "sets": {}, "n_test": int(len(te)), "n_test_scenarios": int(len(set(scen[te])))}
    raws = {}
    for name, rows_eval in (("test", te), ("val_plus_test", vte)):
        raw_by_set = {}
        for s in SETS:
            tab = Q.boot_table({k: v[rows_eval] for k, v in scores[s].items()}, y[rows_eval], scen[rows_eval], n=a.nboot, seed=SEED)
            raw_by_set[s] = tab.pop("_raw")
            res["sets"].setdefault(s, {})[name] = tab
        raws[name] = raw_by_set
        for s in SETS:
            print(name, s, {k: round(v["macro_f1"], 3) for k, v in res["sets"][s][name].items()}, flush=True)
    res["H3"] = {}
    for name in raws:
        res["H3"][name] = {}
        for k in scores["k2"]:
            r16 = raws[name]["k16_mid"][k]
            r2 = raws[name]["k2"][k]
            d = r16[:, 0] - r2[:, 0]
            res["H3"][name][k] = {"macro_f1_k16_mid_minus_k2": {"diff": float(np.nanmean(d)), "ci": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]},
                                 "macro_f1_k8_mid_minus_k2": {"diff": float(np.nanmean(raws[name]["k8_mid"][k][:, 0] - r2[:, 0])),
                                                              "ci": [float(np.nanquantile(raws[name]["k8_mid"][k][:, 0] - r2[:, 0], 0.025)), float(np.nanquantile(raws[name]["k8_mid"][k][:, 0] - r2[:, 0], 0.975))]},
                                 "macro_f1_k2pad_minus_k16_mid": {"diff": float(np.nanmean(raws[name]["k2_pad"][k][:, 0] - r16[:, 0])),
                                                                  "ci": [float(np.nanquantile(raws[name]["k2_pad"][k][:, 0] - r16[:, 0], 0.025)), float(np.nanquantile(raws[name]["k2_pad"][k][:, 0] - r16[:, 0], 0.975))]}}
    json.dump(res, open(f"{a.out}/q3.json", "w"), indent=1)
    with open(f"{a.out}/q3_table.csv", "w") as f:
        f.write("eval,set,readout,macro_f1,macro_f1_lo,macro_f1_hi,auroc,auroc_lo,auroc_hi\n")
        for name in ("test", "val_plus_test"):
            for s in SETS:
                for k, r in res["sets"][s][name].items():
                    f.write(f"{name},{s},{k},{r['macro_f1']:.4f},{r['macro_f1_ci'][0]:.4f},{r['macro_f1_ci'][1]:.4f},{r['auroc']:.4f},{r['auroc_ci'][0]:.4f},{r['auroc_ci'][1]:.4f}\n")
    print("saved")


if __name__ == "__main__":
    main()
