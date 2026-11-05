import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


class Acts:
    def __init__(self, path):
        d = torch.load(path, weights_only=False)
        self.d = d
        self.ids = list(d["ids"])
        self.index = {u: i for i, u in enumerate(self.ids)}
        self.layers = d["meta"]["layers"]

    def get(self, key, layer, rows=None):
        k = self.layers.index(layer)
        x = self.d[key][:, k, :].float().numpy()
        return x if rows is None else x[rows]

    def feat(self, pool, layer, rows=None):
        if pool.endswith("+"):
            base = pool[:-1]
            parts = [self.get(base, layer, rows), self.get(base.replace("last", "mean"), layer, rows)]
            return np.concatenate(parts, 1)
        return self.get(pool, layer, rows)


def pool_names(prefix=False):
    if prefix:
        return ["span_last", "span_mean", "span_last+"]
    return ["last", "mean", "last+"]


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    p = np.asarray(p, float)
    f1 = f1_score(y, p > thr, average="macro")
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return float(f1), float(auc)


def fit_lr(X, y, C):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000).fit(sc.transform(X), y)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def fit_pca_lr(X, y, C, ncomp=50):
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=min(ncomp, X.shape[0] - 1), random_state=0).fit(sc.transform(X))
    Z = pca.transform(sc.transform(X))
    zs = StandardScaler().fit(Z)
    m = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000).fit(zs.transform(Z), y)
    return lambda W: m.predict_proba(zs.transform(pca.transform(sc.transform(W))))[:, 1]


def fit_mlp(X, y, hidden=64, alpha=1e-2, seed=0):
    from sklearn.neural_network import MLPClassifier
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=min(50, X.shape[0] - 1), random_state=0).fit(sc.transform(X))
    Z = pca.transform(sc.transform(X))
    zs = StandardScaler().fit(Z)
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    reps = np.repeat(np.arange(len(y)), np.maximum(np.round(w).astype(int), 1))
    m = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=800, random_state=seed).fit(zs.transform(Z)[reps], y[reps])
    return lambda W: m.predict_proba(zs.transform(pca.transform(sc.transform(W))))[:, 1]


def cross_fit(fit, X, y, groups, n_splits=5, **kw):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        out[te] = fit(X[tr], y[tr], **kw)(X[te])
    return out


def boot_indices(clusters, n=2000, seed=0):
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.nonzero(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq))
        yield np.concatenate([members[c] for c in pick])


def boot_table(scores, y, clusters, n=2000, seed=0):
    names = list(scores)
    res = {k: [] for k in names}
    for idx in boot_indices(clusters, n, seed):
        for k in names:
            res[k].append(metrics(y[idx], scores[k][idx]))
    out = {}
    for k in names:
        a = np.array(res[k])
        f1, auc = metrics(y, scores[k])
        out[k] = {"macro_f1": f1, "auroc": auc,
                  "macro_f1_ci": [float(np.nanquantile(a[:, 0], 0.025)), float(np.nanquantile(a[:, 0], 0.975))],
                  "auroc_ci": [float(np.nanquantile(a[:, 1], 0.025)), float(np.nanquantile(a[:, 1], 0.975))]}
    out["_raw"] = {k: np.array(v) for k, v in res.items()}
    return out


def diff_ci(raw, a, b, col):
    d = raw[a][:, col] - raw[b][:, col]
    return {"diff": float(np.nanmean(d)), "ci": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]}


def rank_acc(sa, sb):
    v = np.sign(np.asarray(sa, float) - np.asarray(sb, float))
    return float((v > 0).mean() + 0.5 * (v == 0).mean())
