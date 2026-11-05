import json
import re
import sys
from collections import defaultdict

sys.modules.setdefault("torch", None)
from transformers import AutoTokenizer

rows = [json.loads(l) for l in open("data/k2/joint.jsonl")]
views = {json.loads(l)["id"]: json.loads(l) for l in open("data/k2/views.jsonl")}
by = defaultdict(dict)
for r in rows:
    by[r["scen"]][r["cell"]] = r
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

fams = {r["family"] for r in rows}
assert len(fams) >= 12, len(fams)
assert all(len(c) == 5 for c in by.values())
per_fam = defaultdict(set)
for r in rows:
    per_fam[r["family"]].add(r["scen"])
assert all(len(v) >= 20 for v in per_fam.values())
print("families", len(fams), "scenarios per family", {len(v) for v in per_fam.values()}, "rows", len(rows))

assert sum(r["label"] for r in rows) == len(by), "only RX should be unsafe"
assert all(r["label"] == 0 for r in rows if r["cell"] != "RX")
assert all(by[s]["PpX"]["label"] == 0 for s in by)

worst_len, worst_plc = 0, 0
for s, c in by.items():
    assert len({c[k]["handoff"] for k in ("RX", "RY", "PX", "PY")}) == 1, s
    for k, r in c.items():
        for w in r["forbid"]:
            assert not re.search(r"\b" + re.escape(w) + r"\b", r["handoff"], re.I), (s, w)
    assert c["RX"]["b_call"] == c["PX"]["b_call"] == c["PpX"]["b_call"], s
    assert c["RX"]["b_result"] == c["PX"]["b_result"] == c["PpX"]["b_result"], s
    assert c["RY"]["b_call"] == c["PY"]["b_call"], s
    for a, b in (("RX", "PX"), ("RX", "PpX"), ("RY", "PY")):
        ha = c[a]["text"][c[a]["text"].index("[Agent A -> Agent B]"):]
        hb = c[b]["text"][c[b]["text"].index("[Agent A -> Agent B]"):]
        assert ha == hb, (s, a, b)
    for a, b in (("RX", "RY"), ("PX", "PY")):
        pa = c[a]["text"][:c[a]["spans"]["b_action"][0]]
        pb = c[b]["text"][:c[b]["spans"]["b_action"][0]]
        assert pa == pb, (s, a, b)
    for x in ("X", "Y"):
        assert not any(c[k]["user"] in views[f"{s}|B|{x}"]["text"] for k in c)
    for a in ("R", "P", "Pp"):
        assert c["RX"]["b_call"] not in views[f"{s}|A|{a}"]["text"] and c["RY"]["b_call"] not in views[f"{s}|A|{a}"]["text"]
    lens = [len(tok(c[k]["text"]).input_ids) for k in ("RX", "RY", "PX", "PY")]
    worst_len = max(worst_len, max(lens) / min(lens) - 1)
    worst_plc = max(worst_plc, abs(len(c["PpX"]["restr"]) - len(c["RX"]["rclause"])))
print("max token-length gap across the 4 cells of a scenario: %.3f" % worst_len)
print("max char gap, placebo restriction vs the R restriction clause: %d" % worst_plc)
assert worst_len < 0.05
assert worst_plc <= 40
print("all tests passed")
