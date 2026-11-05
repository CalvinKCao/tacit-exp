import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)


def rank_fig():
    s1 = json.load(open("results/step1/replication.json"))
    q1 = json.load(open("results/step2/q1.json"))
    L1 = s1["layers"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURF)
    series = [("tool use (TraceSafe, 180 pairs)", [s1["per_layer"][str(l)]["oof"]["context"] for l in L1], L1),
              ("harmful content (HAICOSYSTEM, 258 pairs)", [s1["per_layer"][str(l)]["oof"]["content_strict"] for l in L1], L1),
              ("placebo vs benign", [s1["per_layer"][str(l)]["placebo"]["placebo_vs_benign"] for l in L1], L1),
              ("composition (R,X) > (P,X), family-grouped", [q1["per_layer"][str(l)]["oof_family"] for l in q1["layers"]], q1["layers"])]
    for i, (name, y, x) in enumerate(series):
        ax.plot(x, y, color=COL[i], lw=2, label=name)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(30, 0.505, "chance", color=INK2, ha="right", va="bottom", fontsize=8)
    ax.set_ylim(0.4, 1.02)
    style(ax, "layer (hidden_states index)", "out-of-fold rank accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/rank_accuracy_vs_layer.png", dpi=160, facecolor=SURF)


def cos_fig():
    s1 = json.load(open("results/step1/replication.json"))
    q1 = json.load(open("results/step2/q1.json"))
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURF)
    P = q1["per_layer"]
    L = q1["layers"]
    ax.plot(s1["layers"], [s1["per_layer"][str(l)]["cos"]["tool|content"] for l in s1["layers"]], color=COL[0], lw=2, label="v_tool vs v_content (Step 1)")
    ax.plot(L, [P[str(l)]["cos"]["comp|tool"] for l in L], color=COL[1], lw=2, label="v_comp vs v_tool")
    ax.plot(L, [P[str(l)]["cos"]["comp|content"] for l in L], color=COL[2], lw=2, label="v_comp vs v_content")
    ax.plot(L, [P[str(l)]["cos"]["comp|auth_on_Y"] for l in L], color=COL[3], lw=2, label="v_comp vs auth-only direction (R,Y)-(P,Y)")
    ax.plot(L, [P[str(l)]["split_half"] for l in L], color=COL[4], lw=2, ls=(0, (4, 2)), label="v_comp split-half (same-direction control)")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylim(-0.7, 1.05)
    style(ax, "layer (hidden_states index)", "cosine")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/cosine_vs_layer.png", dpi=160, facecolor=SURF)


def k_fig():
    fp = "results/step3/q3.json"
    if not os.path.exists(fp):
        return
    q3 = json.load(open(fp))
    readouts = [("joint", "joint trace"), ("prefix_B_action", "shared-context prefix"), ("late_logistic_sorted", "late fusion (logistic)"), ("late_mean", "late fusion (mean)"), ("early_pca50_logistic", "early fusion (PCA+logistic)"), ("early_mlp", "early fusion (MLP)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), facecolor=SURF, sharey=True)
    for ax, pl in zip(axes, ("start", "mid")):
        sets = ["k2"] + [f"k{k}_{pl}" for k in (4, 8, 16)]
        for i, (key, name) in enumerate(readouts):
            t = [q3["sets"][s]["test"][key] for s in sets]
            x = [1, 2, 3, 4]
            y = [r["macro_f1"] for r in t]
            ax.plot(x, y, color=COL[i], lw=2, marker="o", ms=5, label=name)
            ax.fill_between(x, [r["macro_f1_ci"][0] for r in t], [r["macro_f1_ci"][1] for r in t], color=COL[i], alpha=0.08, lw=0)
        pad = q3["sets"]["k2_pad"]["test"]["joint"]["macro_f1"]
        ax.scatter([4], [pad], marker="x", color=INK, zorder=5, label="joint, K=2 padded to K=16 length")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["2", "4", "8", "16"])
        ax.set_title(f"authorization at {pl} of trace", color=INK2, fontsize=10)
        ax.set_ylim(0, 1.02)
        style(ax, "agents K (truncation first applies at K=16)", "macro-F1, test families" if pl == "start" else "")
    h, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h, l_, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig("figures/macro_f1_vs_K.png", dpi=160, facecolor=SURF)


if __name__ == "__main__":
    rank_fig()
    cos_fig()
    k_fig()
    print("figures written")
