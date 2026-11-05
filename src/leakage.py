import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

rows = [json.loads(l) for l in open("data/k2/joint.jsonl")]
rows = [r for r in rows if r["cell"] != "PpX"]
y = np.array([r["label"] for r in rows])
txt = [r["text"] for r in rows]
fam = np.array([r["family"] for r in rows])
split = np.array([r["split"] for r in rows])
out = {}


def evaluate(name, tr, te):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=4000, sublinear_tf=True)
    Xtr = vec.fit_transform([txt[i] for i in tr])
    Xte = vec.transform([txt[i] for i in te])
    res = {}
    for mname, m in (("logreg", LogisticRegression(C=1.0, max_iter=2000)), ("gbt", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0))):
        m.fit(Xtr if mname == "logreg" else Xtr.toarray(), y[tr])
        p = m.predict_proba(Xte if mname == "logreg" else Xte.toarray())[:, 1]
        res[mname] = {"macro_f1": float(f1_score(y[te], p > 0.5, average="macro")), "auroc": float(roc_auc_score(y[te], p)), "flag_rate": float((p > 0.5).mean())}
    res["n_test"] = len(te)
    res["prevalence"] = float(y[te].mean())
    res["majority_macro_f1"] = float(f1_score(y[te], np.zeros(len(te)), average="macro"))
    out[name] = res
    print(name, json.dumps(res))


tr = np.where(split == "train")[0]
for s in ("val", "test"):
    evaluate(f"family_heldout_{s}", tr, np.where(split == s)[0])
rng = np.random.default_rng(0)
gss = GroupShuffleSplit(1, test_size=0.2, random_state=0)
scen = np.array([r["scen"] for r in rows])
a, b = next(gss.split(txt, y, groups=scen))
evaluate("same_family_scenario_heldout_diagnostic", a, b)
json.dump(out, open("results/step2/leakage.json", "w"), indent=1)
