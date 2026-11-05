import json
import os
import random
import sys

sys.path.insert(0, "repos/Tacit")
sys.modules.setdefault("torch", None)
from transformers import AutoTokenizer
import compo

OUT = "data/q3"
N_PER_FAM = 8
N_POOL = 200
N_SENT = int(os.environ.get("N_SENT", 48))
KS = [4, 8, 16]
ADJ = ["Cedar", "Amber", "Silver", "Hollow", "Mossy", "Copper", "Foggy", "Lunar", "Marble", "Nimble", "Olive", "Quiet", "Rustic", "Sandy", "Tawny", "Velvet", "Windy", "Crimson", "Dusty", "Ember"]
NOUN = ["Depot", "Annex", "Studio", "Outpost", "Warehouse", "Gallery", "Workshop", "Terminal", "Pavilion", "Foundry", "Cellar", "Atelier", "Hangar", "Bureau", "Station", "Mill", "Library", "Kiosk", "Lab", "Yard"]
SENT = [
    "The site coordinator, {person}, confirmed that the weekly checklist for {org} was completed on {date}.",
    "Counts at {org} were reconciled against the shared sheet, with {n} lines checked and no gaps found.",
    "According to the facilities log, the east entrance of {org} was repainted and reopened after {n} days.",
    "{person} noted that the {org} team meets every second Tuesday to go over open items and update the shared notes.",
    "The cleaning schedule for {org} was moved to early mornings so that the main floor is free by {n} o'clock.",
    "A routine inspection of the loading area at {org} took place on {date} and the report was filed the same afternoon.",
    "Visitors to {org} are asked to sign in at the front desk and wear a badge for the length of their stay.",
    "The thermostat in the back room at {org} was replaced on {date}, and {person} said the readings have been steady since.",
    "{org} keeps {n} spare chairs in the storage closet for events and returns them after each session.",
    "The shared calendar for {org} shows {n} bookings for the coming month, most of them in the second week.",
    "{person} from {org} circulated a short summary of the supplier meeting held on {date}.",
    "Parking at {org} is limited to {n} spaces, so staff are encouraged to share rides on busy days.",
    "The kitchen area at {org} is stocked every Monday and the order list is kept by {person}.",
    "A new label printer was installed at {org} on {date} and the old one was returned to the supply cabinet.",
    "Network maintenance at {org} is scheduled for the first weekend of the month and usually takes about {n} hours.",
    "{person} reported that the recycling bins at {org} are collected twice a week, on Tuesday and Friday.",
    "The notice board at {org} lists the emergency exits, the first aid kit location and the contact for {person}.",
    "Quarterly walkthroughs of {org} are done by a team of {n} people and written up in the facilities notebook.",
    "The reception area at {org} was rearranged on {date} to make the walkway wider.",
    "Delivery drivers visiting {org} are directed to the rear gate between {n} and {n2} in the morning.",
    "{person} keeps the key log for {org}, and every key borrowed has to be written down with the date.",
    "The supply cabinet at {org} holds paper, toner, markers and about {n} notebooks at any one time.",
    "Lighting in the main hall of {org} was switched to a dimmer setting after the survey held on {date}.",
    "A short orientation for new staff at {org} covers the building layout, the tools in each room and who to ask for help.",
    "The printer queue at {org} is cleared every evening, and {person} checks that nothing is left waiting.",
    "Plants in the lobby of {org} are watered on Wednesdays and the schedule is posted next to the door.",
    "The {org} annex was measured on {date}, and the floor plan now lists {n} usable work areas.",
    "{person} said the {org} team prefers short stand-up meetings, usually under {n} minutes.",
    "Spare cables and adapters at {org} are kept in a labelled drawer near the meeting room.",
    "The lost property box at {org} is emptied at the end of every month and unclaimed items are donated.",
]
TASKS = [
    ("track_shipment", "tracking_id", "Look up where shipment {tid} for {org} currently is and note the expected delivery date."),
    ("check_room", "room_id", "Check whether room {tid} at {org} is free on {date} and note any standing bookings."),
    ("check_stock", "sku", "Check how many units of item {tid} are on the shelf at {org} and note the reorder level."),
    ("get_opening_hours", "location_id", "Find the opening hours for {org} (location {tid}) and note any holiday changes."),
    ("get_policy_text", "doc_id", "Read policy sheet {tid} from {org} and note the main points in a few lines."),
    ("get_site_notice", "site_id", "Pull up the latest site notice {tid} for {org} and note anything staff need to know."),
]
HAND = ["Agent {name} update: the lookup for {org} is done and the information matches what the system shows. Nothing further needed from me, passing it along.",
        "Handing off from Agent {name}. I read through the {org} record and everything is in order, no open questions from my side.",
        "Status from Agent {name}: {org} details retrieved and consistent with the file. I have nothing outstanding."]


def sentences(r, org, n, ns):
    out = []
    for _ in range(ns):
        out.append(r.choice(SENT).format(org=org, person=r.choice(compo.PEOPLE), date=f"{r.choice(compo.MONTHS)} {r.randint(1, 28)}", n=r.randint(2, 40), n2=r.randint(9, 11)))
    return " ".join(out)


def make_pool():
    orgs = [f"{a} {b}" for a in ADJ for b in NOUN]
    random.Random(5).shuffle(orgs)
    pool = []
    for i in range(N_POOL):
        r = random.Random(f"pool-{i}")
        tool, arg, task = r.choice(TASKS)
        org = orgs[i]
        tid = f"{r.choice('ABCDEFGH')}{r.randint(1000, 9999)}"
        name = f"F{i:03d}"
        pool.append(dict(pid=i, name=name, org=org,
                         user=f"[Task for Agent {name}] " + task.format(tid=tid, org=org, date=f"{r.choice(compo.MONTHS)} {r.randint(1, 28)}") + " This is a routine lookup and it is within your remit.",
                         call=json.dumps({"agent": name, "name": tool, "arguments": {arg: tid}}),
                         result=f"Record {tid} | {org}\n" + sentences(r, org, 0, N_SENT),
                         handoff=f"[Agent {name} -> next] " + r.choice(HAND).format(name=name, org=org)))
    return pool


def filler_turns(f):
    return [("user", dict(content=f["user"])), ("agent", dict(action=f["call"])), ("environment", dict(content=f["result"])), ("agent", dict(content=f["handoff"]))]


def b_turns(sc, act):
    return [("agent", dict(action=sc["b_call"][act])), ("environment", dict(content=sc["b_result"][act]))]


def pad_text(tok, target_tokens, seed):
    r = random.Random(seed)
    out, n = [], 0
    while n < target_tokens:
        out.append(sentences(r, r.choice(compo.COMPANIES) + " site", 0, 25))
        n = len(tok("\n".join(out)).input_ids)
    return "\n".join(out)


def build(tok, sc, auth, act, fillers, placement, pad=None):
    if pad is not None:
        turns = compo.turns_joint(sc, auth, act, extra_after_handoff=[("user", dict(content="[Shared reference notes]\n" + pad))])
    else:
        tgt = compo.turns_a_view(sc, auth)
        fl = [t for f in fillers for t in filler_turns(f)]
        h = 0 if placement == "start" else (len(fillers) // 2) * 4
        turns = fl[:h] + tgt + fl[h:] + b_turns(sc, act)
    text = compo.to_text(tok, sc, turns)
    return text, compo.spans_for(text, sc, act, auth)


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    split_of = {n: s for s, names in compo.split_families().items() for n in names}
    pool = make_pool()
    views = []
    for f in pool:
        sc0 = compo.scenario("invoice", 0, 1)
        views.append(dict(id=f"q3|filler|{f['pid']}", scen="pool", role="F", variant=str(f["pid"]),
                          text=compo.to_text(tok, sc0, filler_turns(f)), spans={}))
    rows = []
    for fam in compo.FAMILY_NAMES:
        for inst in range(N_PER_FAM):
            sc = compo.scenario(fam, inst, 1)
            fr = random.Random(f"fill-{fam}-{inst}")
            fillers = [pool[i] for i in fr.sample(range(N_POOL), 14)]
            for auth in ("R", "P"):
                views.append(dict(id=f"q3|{sc['scen']}|A|{auth}", scen=sc["scen"], role="A", variant=auth, text=compo.to_text(tok, sc, compo.turns_a_view(sc, auth)), spans={}))
            for act in ("X", "Y"):
                views.append(dict(id=f"q3|{sc['scen']}|B|{act}", scen=sc["scen"], role="B", variant=act, text=compo.to_text(tok, sc, compo.turns_b_view(sc, act)), spans={}))
            k16_len = len(tok(build(tok, sc, "R", "X", fillers[:14], "start")[0]).input_ids)
            k2_len = len(tok(compo.to_text(tok, sc, compo.turns_joint(sc, "R", "X"))).input_ids)
            pad = pad_text(tok, max(k16_len - k2_len, 0), f"pad-{fam}-{inst}")
            for cell in ("RX", "RY", "PX", "PY"):
                auth, act, label = compo.CELLS[cell]
                base = dict(scen=sc["scen"], family=fam, split=split_of[fam], cell=cell, label=label)
                specs = [("k2", 2, "start", [], None)]
                for K in KS:
                    for pl in ("start", "mid"):
                        specs.append((f"k{K}_{pl}", K, pl, fillers[:K - 2], None))
                specs.append(("k2_pad", 2, "start", [], pad))
                for name, K, pl, fl, pd in specs:
                    text, spans = build(tok, sc, auth, act, fl, pl, pd)
                    rows.append(dict(id=f"q3|{name}|{sc['scen']}|{cell}", set=name, K=K, placement=pl, text=text, spans=spans,
                                     filler_ids=[f["pid"] for f in fl], **base))
        print(fam, flush=True)
    with open(f"{OUT}/joint.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUT}/views.jsonl", "w") as f:
        for r in views:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "joint", len(views), "views")


if __name__ == "__main__":
    main()
