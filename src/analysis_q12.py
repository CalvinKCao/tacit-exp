import argparse
import csv
import json
import os

import numpy as np
from sklearn.model_selection import GroupKFold

import qlib as Q

LAYERS = list(range(12, 29))
CS = [0.001, 0.003, 0.01, 0.1]
SEED = 2026


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="results/acts")
    ap.add_argument("--out", default="results/step2")
    ap.add_argument("--step1", default="results/step1/directions.npz")
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open("data/k2/joint.jsonl")]
    for r in rows:
        r.pop("text")
    J = Q.Acts(f"{a.acts}/k2_joint.pt")
    V = Q.Acts(f"{a.acts}/k2_views.pt")
    assert J.ids == [r["id"] for r in rows]
    N = len(rows)
    scen = np.array([r["scen"] for r in rows])
    fam = np.array([r["family"] for r in rows])
    split = np.array([r["split"] for r in rows])
    cell = np.array([r["cell"] for r in rows])
    y = np.array([r["label"] for r in rows])
    main4 = cell != "PpX"
    vA = np.array([V.index[f"{r['scen']}|A|{r['auth']}"] for r in rows])
    vB = np.array([V.index[f"{r['scen']}|B|{r['action']}"] for r in rows])

    scens = sorted(set(scen))
    pos = {c: np.array([np.nonzero((scen == s) & (cell == c))[0][0] for s in scens]) for c in ("RX", "RY", "PX", "PY", "PpX")}
    sfam = np.array([fam[np.nonzero(scen == s)[0][0]] for s in scens])

    d1 = np.load(a.step1)
    rng = np.random.default_rng(SEED)
    q1 = {"layers": J.layers, "n_pairs": len(scens), "per_layer": {}}
    splits = []
    for _ in range(5):
        idx = rng.permutation(len(scens))
        splits.append((idx[:len(idx) // 2], idx[len(idx) // 2:]))
    pw = np.random.default_rng(SEED + 1).poisson(1.0, size=(500, len(scens))).astype(np.float32)
    fold_s = list(GroupKFold(5).split(np.arange(len(scens)), groups=np.arange(len(scens))))
    fam_ids = np.unique(sfam, return_inverse=True)[1]
    fold_f = list(GroupKFold(6).split(np.arange(len(scens)), groups=fam_ids))
    for l in J.layers:
        X = J.get("last", l) / d1[f"sd_L{l}"]
        D = X[pos["RX"]] - X[pos["PX"]]
        v = unit(D.mean(0))
        vt, vc = d1[f"v_tool_L{l}"], d1[f"v_content_L{l}"]
        v_auth = unit((X[pos["RY"]] - X[pos["PY"]]).mean(0))
        v_act = unit((X[pos["PX"]] - X[pos["PY"]]).mean(0))
        row = {"cos": {"comp|tool": float(v @ vt), "comp|content": float(v @ vc), "comp|auth_on_Y": float(v @ v_auth), "comp|act_under_P": float(v @ v_act)}}
        bd = pw @ D
        bd = bd / (np.linalg.norm(bd, axis=1, keepdims=True) + 1e-12)
        for nm, w in (("tool", vt), ("content", vc)):
            c = bd @ w
            row["cos"][f"comp|{nm}_ci"] = [float(np.quantile(c, 0.025)), float(np.quantile(c, 0.975))]
        row["split_half"] = float(np.mean([unit(D[i].mean(0)) @ unit(D[j].mean(0)) for i, j in splits]))
        for nm, folds in (("oof_scenario", fold_s), ("oof_family", fold_f)):
            sc_ = np.zeros(len(scens))
            for tr, te in folds:
                w = unit(D[tr].mean(0))
                sc_[te] = D[te] @ w
            row[nm] = Q.rank_acc(sc_, np.zeros(len(sc_)))
            if nm == "oof_scenario":
                ci = []
                bi = np.random.default_rng(SEED + 2)
                for _ in range(1000):
                    ii = bi.integers(0, len(sc_), len(sc_))
                    ci.append(Q.rank_acc(sc_[ii], np.zeros(len(ii))))
                row["oof_scenario_ci"] = [float(np.quantile(ci, 0.025)), float(np.quantile(ci, 0.975))]
        cross = {}
        for nm, w in (("v_tool", vt), ("v_content", vc), ("v_auth_on_Y", v_auth), ("v_comp_infold", v)):
            s = X @ w
            cross[nm] = {"RX>PX": Q.rank_acc(s[pos["RX"]], s[pos["PX"]]), "RX>PpX": Q.rank_acc(s[pos["RX"]], s[pos["PpX"]]), "PpX>PX": Q.rank_acc(s[pos["PpX"]], s[pos["PX"]]),
                         "RY>PY": Q.rank_acc(s[pos["RY"]], s[pos["PY"]]), "RX>RY": Q.rank_acc(s[pos["RX"]], s[pos["RY"]])}
        row["cross"] = cross
        q1["per_layer"][l] = row
        print("Q1 L%d oof %.3f cos tool %+.3f content %+.3f split %.3f" % (l, row["oof_scenario"], row["cos"]["comp|tool"], row["cos"]["comp|content"], row["split_half"]), flush=True)
    pk = max(J.layers, key=lambda l: q1["per_layer"][l]["oof_scenario"])
    q1["peak_layer"] = pk
    json.dump(q1, open(f"{a.out}/q1.json", "w"), indent=1)
    with open(f"{a.out}/q1_curves.csv", "w") as f:
        f.write("layer,oof_scenario,oof_family,cos_tool,cos_content,cos_auth_on_Y,cos_act,split_half\n")
        for l in J.layers:
            r = q1["per_layer"][l]
            f.write(f"{l},{r['oof_scenario']:.4f},{r['oof_family']:.4f},{r['cos']['comp|tool']:.4f},{r['cos']['comp|content']:.4f},{r['cos']['comp|auth_on_Y']:.4f},{r['cos']['comp|act_under_P']:.4f},{r['split_half']:.4f}\n")

    tr = np.nonzero((split == "train") & main4)[0]
    va = np.nonzero((split == "val") & main4)[0]
    te_all = np.nonzero(split == "test")[0]
    te = np.nonzero((split == "test") & main4)[0]

    def select(getX, pools, rows_tr, rows_va):
        best, bkey = None, None
        for pool in pools:
            for l in LAYERS:
                X = getX(pool, l)
                for C in CS:
                    p = Q.fit_lr(X[rows_tr], y[rows_tr], C)(X[rows_va])
                    f1, auc = Q.metrics(y[rows_va], p)
                    key = (round(f1, 3), round(auc, 3))
                    if best is None or key > best:
                        best, bkey = key, (pool, l, C)
        return bkey, best

    sel = {}
    out = {}
    jfeat = lambda pool, l: J.feat(pool, l)
    sel["joint"], v = select(jfeat, Q.pool_names(), tr, va)
    print("joint selected", sel["joint"], v, flush=True)
    sel["prefix"], v = select(lambda pool, l: J.feat(pool, l), Q.pool_names(True), tr, va)
    print("prefix selected", sel["prefix"], v, flush=True)
    sel["agentA"], v = select(lambda pool, l: V.feat(pool, l)[vA], Q.pool_names(), tr, va)
    print("agentA selected", sel["agentA"], v, flush=True)
    sel["agentB"], v = select(lambda pool, l: V.feat(pool, l)[vB], Q.pool_names(), tr, va)
    print("agentB selected", sel["agentB"], v, flush=True)

    scores = {}

    def fitscore(name, X, C, rows_fit=tr):
        f = Q.fit_lr(X[rows_fit], y[rows_fit], C)
        return f(X)

    pool, l, C = sel["joint"]
    scores["joint_selected"] = fitscore("j", jfeat(pool, l), C)
    scores["joint_paper_L22_last+_C0.003"] = fitscore("jp", jfeat("last+", 22), 0.003)
    pool, l, C = sel["prefix"]
    scores["prefix_B_action"] = fitscore("p", J.feat(pool, l), C)

    XA = V.feat(sel["agentA"][0], sel["agentA"][1])[vA]
    XB = V.feat(sel["agentB"][0], sel["agentB"][1])[vB]
    sA = fitscore("a", XA, sel["agentA"][2])
    sB = fitscore("b", XB, sel["agentB"][2])
    scores["agentA_only"], scores["agentB_only"] = sA, sB
    oofA = np.zeros(N)
    oofB = np.zeros(N)
    gtr = scen[tr]
    for fit_tr, fit_te in GroupKFold(5).split(tr, groups=gtr):
        oofA[tr[fit_te]] = Q.fit_lr(XA[tr[fit_tr]], y[tr[fit_tr]], sel["agentA"][2])(XA[tr[fit_te]])
        oofB[tr[fit_te]] = Q.fit_lr(XB[tr[fit_tr]], y[tr[fit_tr]], sel["agentB"][2])(XB[tr[fit_te]])
    srt = lambda a_, b_: np.stack([np.maximum(a_, b_), np.minimum(a_, b_)], 1)
    scores["late_max"] = np.maximum(sA, sB)
    scores["late_mean"] = (sA + sB) / 2
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(C=10.0, class_weight="balanced").fit(srt(oofA[tr], oofB[tr]), y[tr])
    scores["late_logistic_sorted"] = lg.predict_proba(srt(sA, sB))[:, 1]

    pool, l, _ = sel["joint"]
    XE = np.concatenate([V.feat(pool, l)[vA], V.feat(pool, l)[vB]], 1)
    best, bC = None, None
    for C in (0.01, 0.1, 1.0):
        p = Q.fit_pca_lr(XE[tr], y[tr], C)(XE[va])
        k = tuple(round(x, 3) for x in Q.metrics(y[va], p))
        if best is None or k > best:
            best, bC = k, C
    scores["early_pca50_logistic"] = Q.fit_pca_lr(XE[tr], y[tr], bC)(XE)
    best, bm = None, None
    for hid in (32, 128):
        for al in (1e-3, 1e-1):
            p = Q.fit_mlp(XE[tr], y[tr], hid, al)(XE[va])
            k = tuple(round(x, 3) for x in Q.metrics(y[va], p))
            if best is None or k > best:
                best, bm = k, (hid, al)
    scores["early_mlp"] = Q.fit_mlp(XE[tr], y[tr], *bm)(XE)
    sel["early"] = {"pca_C": bC, "mlp": bm, "layer_pool": [pool, l]}

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression as LR2
    texts = [json.loads(l_)["text"] for l_ in open("data/k2/joint.jsonl")]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=4000, sublinear_tf=True)
    Xt = vec.fit_transform([texts[i] for i in tr])
    Xall = vec.transform(texts)
    scores["text_tfidf_logreg"] = LR2(C=1.0, max_iter=2000, class_weight="balanced").fit(Xt, y[tr]).predict_proba(Xall)[:, 1]
    scores["text_tfidf_gbt"] = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(Xt.toarray(), y[tr]).predict_proba(Xall.toarray())[:, 1]

    idx = {r["id"]: i for i, r in enumerate(rows)}
    for k_ in ("qwen3guard-4b", "agentdog"):
        fp = f"{a.out}/guards_{k_}.json"
        if not os.path.exists(fp):
            print("missing", fp)
            continue
        g = json.load(open(fp))
        s = np.full(N, np.nan)
        for r in g:
            s[idx[r["id"]]] = r["score"]
        scores[f"guard_{k_}"] = s

    ytest = y[te]
    table = Q.boot_table({k: v[te] for k, v in scores.items()}, ytest, scen[te], n=a.nboot, seed=SEED)
    raw = table.pop("_raw")
    diffs = {}
    for other in ("late_mean", "late_logistic_sorted", "late_max", "early_pca50_logistic", "early_mlp", "prefix_B_action", "text_tfidf_logreg"):
        diffs[f"joint_selected - {other}"] = {"macro_f1": Q.diff_ci(raw, "joint_selected", other, 0), "auroc": Q.diff_ci(raw, "joint_selected", other, 1)}
    per_family = {}
    for f_ in sorted(set(fam[te])):
        m = fam[te] == f_
        per_family[f_] = {k: Q.metrics(ytest[m], v[te][m]) for k, v in scores.items() if not np.isnan(v[te]).any()}

    diag = {}
    for k, v in scores.items():
        if np.isnan(v[te_all]).any():
            continue
        c = lambda n: v[te_all][cell[te_all] == n]
        ord_ = lambda n: np.argsort(scen[te_all][cell[te_all] == n])
        g = lambda n: c(n)[ord_(n)]
        diag[k] = {"mean_score": {n: float(c(n).mean()) for n in ("RX", "RY", "PX", "PY", "PpX")},
                   "flag_rate": {n: float((c(n) > 0.5).mean()) for n in ("RX", "RY", "PX", "PY", "PpX")},
                   "rank_RX_vs_PX": Q.rank_acc(g("RX"), g("PX")), "rank_RX_vs_RY": Q.rank_acc(g("RX"), g("RY")),
                   "rank_RX_vs_PpX": Q.rank_acc(g("RX"), g("PpX")), "rank_PpX_vs_PX": Q.rank_acc(g("PpX"), g("PX"))}
    vtab = {k: dict(zip(("macro_f1", "auroc"), Q.metrics(y[va], v[va]))) for k, v in scores.items() if not np.isnan(v[va]).any()}
    json.dump({"selected": sel, "test": table, "val": vtab, "diffs": diffs, "per_family": per_family, "contrast_diagnostics": diag, "n_test": int(len(te)), "n_test_scenarios": int(len(set(scen[te])))},
              open(f"{a.out}/q2.json", "w"), indent=1)
    with open(f"{a.out}/q2_table.csv", "w") as f:
        f.write("readout,macro_f1,macro_f1_lo,macro_f1_hi,auroc,auroc_lo,auroc_hi\n")
        for k, r in table.items():
            f.write(f"{k},{r['macro_f1']:.4f},{r['macro_f1_ci'][0]:.4f},{r['macro_f1_ci'][1]:.4f},{r['auroc']:.4f},{r['auroc_ci'][0]:.4f},{r['auroc_ci'][1]:.4f}\n")
    for k, r in table.items():
        print("Q2 %-32s f1 %.3f [%.3f,%.3f] auc %.3f [%.3f,%.3f]" % (k, r["macro_f1"], *r["macro_f1_ci"], r["auroc"], *r["auroc_ci"]))
    np.save(f"{a.out}/q2_scores.npy", {"ids": J.ids, "scores": scores}, allow_pickle=True)


if __name__ == "__main__":
    main()
