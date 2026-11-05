import json
import os
import sys

sys.path.insert(0, "repos/Tacit")
sys.modules.setdefault("torch", None)
from transformers import AutoTokenizer
import compo

N_INST = 20
OUT = "data/k2"


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    split_of = {}
    for s, names in compo.split_families().items():
        for n in names:
            split_of[n] = s
    joint, views = [], []
    for fam in compo.FAMILY_NAMES:
        for inst in range(N_INST):
            sc = compo.scenario(fam, inst)
            for cell, (auth, act, label) in compo.CELLS.items():
                text = compo.to_text(tok, sc, compo.turns_joint(sc, auth, act))
                joint.append(dict(id=f"{sc['scen']}|{cell}", scen=sc["scen"], family=fam, split=split_of[fam], cell=cell, auth=auth,
                                  action=act, label=label, text=text, spans=compo.spans_for(text, sc, act, auth),
                                  handoff=sc["handoff"], b_call=sc["b_call"][act], b_result=sc["b_result"][act],
                                  forbid=sc["forbid"], user=sc["auth"][auth], restr=sc["restr"], rclause=sc["rclause"]))
            for auth in ("R", "P", "Pp"):
                views.append(dict(id=f"{sc['scen']}|A|{auth}", scen=sc["scen"], role="A", variant=auth,
                                  text=compo.to_text(tok, sc, compo.turns_a_view(sc, auth)), spans={}))
            for act in ("X", "Y"):
                views.append(dict(id=f"{sc['scen']}|B|{act}", scen=sc["scen"], role="B", variant=act,
                                  text=compo.to_text(tok, sc, compo.turns_b_view(sc, act)), spans={}))
    with open(f"{OUT}/joint.jsonl", "w") as f:
        for r in joint:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUT}/views.jsonl", "w") as f:
        for r in views:
            f.write(json.dumps(r) + "\n")
    json.dump(compo.split_families(), open(f"{OUT}/split.json", "w"), indent=1)
    print(len(joint), "joint traces", len(views), "views")


if __name__ == "__main__":
    main()
