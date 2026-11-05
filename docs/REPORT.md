# Compositional harm and TACIT's directions (Qwen3-4B-Instruct-2507)

All numbers below come from code in `src/` and are stored in `results/`. Seeds: analysis seed 2026 (bootstrap and splits), family split seed 7, generator seeds are derived from family name and instantiation index. Model: Qwen/Qwen3-4B-Instruct-2507, bf16, SDPA, 16,384-token head+tail truncation. GPU: Killarney L40S, about 1.2 GPU hours of the 4 hour cap (preflight 1 min, step 1 extraction 11 min, step 2 15 min, Q3 43 min; the first Q3 attempt failed in 0 s on node kn159 and a redundant analysis job was cancelled before it started). Hypotheses are in `PLAN.md`, which was written before any extraction and not edited since.

## What was done

1. Preflight (PREFLIGHT.md): Tacit repo, models, TraceSafe and HAICOSYSTEM all available; Llama models gated.
2. Step 1, replication of Section 3 on 360 TraceSafe traces (180 MissingTypeHint/AmbiguousArg pairs), 180 placebo traces and 1,205 HAICOSYSTEM traces (258 content pairs).
3. Step 2, a 2x2 compositional-harm dataset: 12 families x 20 instantiations x cells RX, RY, PX, PY plus a placebo cell PpX (permissive request plus an unrelated restriction about a second company), 1,200 joint traces and 1,200 per-agent local views. Families split 8 train / 2 validation / 2 test (seed 7). Tests in `src/test_data.py` pass: the handoff is string-identical across the four cells, B's action and result are identical across auth conditions, the text up to B's action is identical between RX and RY, the text from the handoff on is identical between RX and PX, only RX is labelled unsafe, token-length gap across cells at most 4.7%, placebo restriction within 13 characters of the R restriction clause.
4. Q1 on the K=2 data, Q2 with six readout families plus two text guards, Q3 on 3,072 K-agent traces (K in {2,4,8,16}, authorization at start or mid-trace, plus a K=2 trace padded to K=16 length).

## Step 1 replication

| Check (pre-registered window) | Paper | Here | Pass |
|---|---|---|---|
| Tool-use out-of-fold rank accuracy at peak | ~0.98 | 0.986 at layer 15, CI [0.969, 1.000] | yes |
| Peak layer intermediate | intermediate | 15 (layer 22: 0.914) | yes |
| Placebo vs benign at the tool peak (0.49-0.59) | ~0.54 | 0.600, CI [0.528, 0.667]; 0.533 at layer 22 | no, by 0.01 |
| Content peak (0.68-0.79) | 0.73-0.74 | 0.736 at layer 20, CI [0.678, 0.791] | yes |
| abs cosine tool/content < 0.2 | ~0 | 0.001 and -0.004 at the two peaks | yes |
| Same-risk split-half (0.76-0.86) | ~0.81 | 0.81 (tool), 0.83 (content) | yes |

The placebo curve moves between 0.46 and 0.67 across layers (mean 0.58, n=180); I read the miss as sampling noise plus peak-layer selection, but it is a miss against the window I set. `v_tool` and `v_content` per layer are in `results/step1/directions.npz`.

## Q1: where does compositional harm sit?

Direction `v_comp` = out-of-fold mean difference of (R,X) minus (P,X) pairs (240 scenarios), same recipe, same per-dimension scaling as Step 1.

| Layer | OOF rank acc (scenario folds / family folds) | cos(v_comp, v_tool) [CI] | cos(v_comp, v_content) [CI] | cos(v_comp, auth-only (R,Y)-(P,Y)) | split-half |
|---|---|---|---|---|---|
| 10 (first layer at 1.0) | 1.000 / 0.975 | 0.077 [0.059, 0.095] | -0.061 [-0.076, -0.043] | 0.854 | 0.94 |
| 19 | 1.000 / 1.000 | 0.016 [0.012, 0.020] | 0.010 [0.001, 0.018] | 0.138 | 0.99 |
| 22 | 1.000 / 1.000 | 0.014 [0.009, 0.019] | 0.004 [-0.003, 0.012] | -0.264 | 0.99 |

Cross-ranking of (R,X) over (P,X) by the Step 1 directions is unstable across layers: `v_tool` ranges 0.04-0.78 and `v_content` 0.13-0.91, with values well below 0.5 (opposite sign) at several layers. Both exceed 0.75 only at a few layers: `v_content` at layers 13 (0.79), 20 (0.85) and 21 (0.91), `v_tool` at layer 21 (0.78). At layer 21 the placebo is also ranked above (P,X) (0.68-0.70) and (R,X) above the placebo is 0.73-0.86, so these are not clean separations. On the placebo, `v_comp` also scores (P,X)+unrelated restriction above (P,X) (rank 0.78-0.93) but ranks (R,X) above the placebo at 0.99-1.00. Curves: `figures/rank_accuracy_vs_layer.png`, `figures/cosine_vs_layer.png`.

Answer (hedged): in this synthetic setting the direction that separates compositional-harm traces from their permissive twins is nearly orthogonal to both TACIT directions at every layer (|cos| under 0.1) and is estimated stably, so it is consistent with a third direction. The cross-ranking results are less tidy: around layers 20-21 the harmful-content direction does rank (R,X) above (P,X) (0.85-0.91), so a partial alignment with that direction at those layers cannot be excluded even though the cosine is near zero. Two caveats weaken the reading. In layers 8-19 it overlaps strongly with the plain authorization-wording direction (cosine 0.4-0.9), so there it may mostly encode "a restriction was stated"; the overlap falls and turns negative after layer 20. And the rank accuracy saturates at 1.0 for the wording difference alone, so it says nothing about whether the model represents the relation between authorization and action.

## Q2: which readouts recover it (K=2, test families, 160 traces, 40 scenarios, threshold 0.5, 2,000 scenario-clustered bootstrap resamples)

| Readout | macro-F1 [95% CI] | AUROC [95% CI] |
|---|---|---|
| (1) joint trace, chosen on validation (last token, layer 19, C=0.001) | 0.780 [0.728, 0.835] | 1.000 [1.000, 1.000] |
| (1) joint trace, paper settings (layer 22, last+mean, C=0.003) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| (3) agent A alone | 0.680 [0.644, 0.709] | 0.807 [0.784, 0.825] |
| (3) agent B alone | 0.287 [0.218, 0.350] | 0.833 [0.833, 0.833] |
| (3) late fusion, max | 0.271 [0.218, 0.327] | 0.820 [0.808, 0.829] |
| (3) late fusion, mean | 0.659 [0.590, 0.722] | 0.922 [0.908, 0.940] |
| (3) late fusion, logistic on sorted scores | 0.697 [0.656, 0.735] | 0.870 [0.843, 0.909] |
| (4) early fusion, PCA(50)+logistic | 0.898 [0.832, 0.947] | 0.985 [0.974, 0.995] |
| (4) early fusion, one-hidden-layer MLP | 0.877 [0.808, 0.938] | 0.974 [0.948, 0.996] |
| (5) shared-context prefix at B's action tokens | 0.888 [0.820, 0.947] | 0.999 [0.996, 1.000] |
| (6) Qwen3Guard-Gen-4B, joint trace | 0.429 [0.429, 0.429] | 0.990 [0.979, 0.999] |
| (6) AgentDoG-Qwen3-4B, joint trace | 0.590 [0.531, 0.642] | 0.833 [0.787, 0.879] |
| text baseline: TF-IDF logistic | 0.768 [0.720, 0.815] | 0.896 [0.866, 0.923] |
| text baseline: TF-IDF gradient boosting | 0.455 [0.429, 0.503] | 0.891 [0.870, 0.916] |

Readout (2) (TACIT's released readout) is skipped, readout (6) for Llama-Guard-3-8B is SKIPPED-PENDING. Paired differences (macro-F1, joint chosen on validation minus other): late mean +0.122 [0.058, 0.187], late logistic +0.085 [0.027, 0.145], early PCA -0.117 [-0.205, -0.027], early MLP -0.096 [-0.185, 0.003], prefix -0.106 [-0.191, -0.016]. Full diagnostics, per-family numbers and per-cell flag rates are in `results/step2/q2.json`.

Sanity checks on the dataset: agent B alone ranks (R,X) against (P,X) at exactly 0.500, and agent A alone ranks (R,X) against (R,Y) at exactly 0.500, as the construction requires. Qwen3Guard flags nothing at its verdict (0 of 200) yet its unsafe probability ranks (R,X) above (P,X) in every test pair (AUROC 0.99). AgentDoG flags every trace with the risky action regardless of authorization (RX 1.00, PX 1.00, placebo 0.93, RY 0.23, PY 0.40).

Answer (hedged): a joint-trace probe with the paper's settings separates the unsafe cell perfectly on test families, and the prefix readout and isolated early fusion come close (macro-F1 about 0.88-0.90). Isolated late fusion recovers much of the ranking (AUROC 0.87-0.92) but not at the fixed 0.5 threshold. The validation-selected joint probe has perfect AUROC but macro-F1 0.78, because validation tied at 1.0 and the first-best choice had a badly calibrated C; I report it because that was the pre-registered selection rule. Single agents alone are at the 0.83 AUROC ceiling that a perfect detector of one factor gets on an AND label. Because the label is a conjunction of two separable factors, the dataset cannot show that a nonlinear aggregation is needed; the MLP did not beat the linear early-fusion probe.

## Q3: scaling with agent count (test families, 64 traces, 16 scenarios; scenario-clustered bootstrap)

Truncation first applies at K=16 (mean 18.5k tokens, 100% of traces truncated; K=8 mean 8.1k, none). At K=16 with the authorization mid-trace, the authorization message is fully removed in 100% of traces; at K=16 with it at the start, and in the padded K=2 control, it is kept (`results/step3/q3_lengths.json`).

Macro-F1 [95% CI] / AUROC. Settings chosen at K=2 on validation (joint: last, layer 19, C=0.001).

| Readout | K=2 | K=4 start | K=8 start | K=16 start | K=4 mid | K=8 mid | K=16 mid | K=2 padded to 16 |
|---|---|---|---|---|---|---|---|---|
| joint trace | 0.78 [0.72,0.84] / 1.00 | 0.78 / 1.00 | 0.72 [0.68,0.76] / 1.00 | 0.65 [0.59,0.70] / 1.00 | 0.79 / 1.00 | 0.68 [0.62,0.73] / 1.00 | 0.50 [0.37,0.62] / 0.83 | 0.86 [0.79,0.92] / 1.00 |
| prefix at B's action | 0.83 [0.69,0.93] / 0.99 | 0.82 / 0.99 | 0.81 / 0.99 | 0.76 [0.66,0.85] / 0.97 | 0.85 / 0.95 | 0.81 / 0.95 | 0.47 [0.38,0.50] / 0.67 | 1.00 / 1.00 |
| late fusion, logistic | 0.86 [0.73,0.93] / 0.95 | same | same | same | same | same | same | same |
| late fusion, mean | 0.56 / 0.96 | 0.69 / 0.96 | 0.43 / 0.97 | 0.43 / 0.96 | 0.69 / 0.96 | 0.43 / 0.97 | 0.43 / 0.96 | 0.56 / 0.96 |
| early fusion, PCA+logistic | 0.89 [0.81,0.96] / 1.00 | 0.96 / 1.00 | 0.88 / 0.95 | 0.62 [0.53,0.69] / 0.80 | 0.96 / 1.00 | 0.88 / 0.95 | 0.62 / 0.80 | 0.89 / 1.00 |
| early fusion, MLP | 0.84 / 0.97 | 0.93 / 1.00 | 0.81 / 0.97 | 0.63 / 0.74 | 0.93 / 1.00 | 0.81 / 0.97 | 0.63 / 0.74 | 0.84 / 0.97 |

(Per-cell CIs for every entry: `results/step3/q3_table.csv`. Early and late fusion do not depend on placement, only on the chain order of the same per-agent states. Figure: `figures/macro_f1_vs_K.png`. A supplementary evaluation on validation plus test families, 128 traces, is in the same files and shows the same pattern.)

Paired macro-F1 differences for the joint probe (mid placement): K=16 minus K=2 = -0.279 [-0.44, -0.14]; K=8 minus K=2 = -0.102 [-0.19, -0.01]; padded K=2 minus K=16 = +0.356 [0.17, 0.56]. Prefix readout K=16 minus K=2 = -0.361 [-0.55, -0.19], K=8 = -0.013 [-0.18, 0.17]. Early fusion PCA K=16 minus K=2 = -0.270 [-0.38, -0.18]; late logistic 0.000.

Answer (hedged): the joint-trace readout loses macro-F1 as K grows, and the K=2 trace padded to K=16 length does not (0.86), so length alone is not the cause in this setup. The largest drop is the mid-placement K=16 case, where truncation removes the authorization and the readout falls to AUROC 0.83, the level of a detector that sees only the risky action; I read that as information removal and not as a representation failure. In the other cells the joint probe keeps AUROC 1.00 while macro-F1 slides from 0.78 to 0.65-0.72 (start placement, and K=8 mid with no truncation), which looks like a threshold shift under longer context at the validation-chosen C and is not a loss of ranking. The picture for fused per-agent readouts is mixed. Late fusion with a logistic over sorted scores is exactly flat across K, but this is partly built in: the filler agents' views were among the probe's training negatives, so their scores sit near zero and the fusion effectively reads the two target agents only. Late fusion by mean dilutes with K at the 0.5 threshold (0.56 to 0.43) while its AUROC stays near 0.96. Early fusion did not hold at K=16 (0.62, AUROC 0.80 and 0.74; 41k concatenated dimensions against 256 training rows).

## Which hypotheses held (PLAN.md)

- Step 1 acceptance: passed on all windows except the placebo (0.600 against 0.49-0.59).
- H1: supported at the layers I pre-specified. At layer 10 (first layer with out-of-fold accuracy 1.0) and at layer 22, |cos| with v_tool and v_content is below 0.1 and the split-half cosine is 0.94 and 0.99. The cross-ranking part of the criterion is not clean: v_tool reaches 0.78 only at layer 21 (where the placebo is also ranked above (P,X)), and v_content reaches 0.79-0.91 at layers 13, 20 and 21, so H1 is supported on the cosine criterion and only partly on the cross-ranking one. The overlap with the auth-only direction at layer 10 (0.85) is a confound, not covered by the criterion.
- H2 was reformulated before any extraction (the plan notes why: the suggested version, "near chance by construction", does not follow from an AND label). H2a (contrast-level ties) held exactly, 0.500 in both cases. H2b (late fusion recovers most of the signal; falsified if the AUROC lower bound were below 0.75 for both mean and logistic) was not falsified: lower bounds 0.908 and 0.843. Max fusion is worse than mean, as expected (macro-F1 0.27 against 0.66). H2c: the validation-chosen joint probe has higher macro-F1 than late fusion with excluded-zero CIs, but isolated early fusion and the prefix readout are higher than the validation-chosen joint probe on macro-F1, and the paper-settings joint probe is perfect; the answer depends on which joint settings are used.
- H3: the joint trace "works at K=2" on AUROC (1.00 against the 0.9 criterion) but its macro-F1 at the validation-chosen settings is 0.777, below the 0.8 criterion (the paper-settings probe on the Step 2 test families is 1.0). The degradation criterion (macro-F1 at K=16 mid lower than K=2 by more than 0.10 with a CI that excludes 0, and a higher padded control) is met: -0.279 [-0.44, -0.14] and +0.356. The drop is concentrated where the authorization is truncated away; the mid-placement drop at K=8, before any truncation, is -0.102 [-0.19, -0.01].

Against the interpretation guide: the pattern "joint-trace degrades with K while fused readouts hold" appears for late fusion only, and partly by construction; early fusion degraded too. "Linear fusion at chance but MLP succeeds" was not observed. "All local readouts at chance and only joint works" was not observed; in this 2x2 the fused local readouts recover the label.

## Limitations

Synthetic slot-filled templates with mundane content, one small model, no adaptive adversary, no LLM paraphrase. The 2x2's AND label is separable from two features, so it cannot separate a need for nonlinear aggregation from a working linear fusion, and the lexical restriction cue alone ranks the unsafe cell (TF-IDF AUROC 0.89-0.93 on held-out families). Test sets are small: 40 scenarios for Q2 and 16 scenarios (64 traces) for Q3, so intervals are wide and several AUROC intervals collapse to a point; Q3's late-fusion flatness is partly by construction as noted. The relational variant (harm iff the action's entity differs from the authorized entity), which would remove these two issues, was not built because it is not in the specified design.

## Deviations from the prompt and the plan

- `<PDF_PATH>` and `<GPU_INFO>` were empty: used the arXiv PDF and Killarney L40S 48 GB.
- Backbone is Qwen3-4B-Instruct-2507 only (the checkpoint Tacit's config uses); plain Qwen3-4B was not run.
- TraceSafe taken from the official GitHub `data/` folder instead of the gated Hugging Face copy.
- transformers 5.22.1 from the Alliance wheelhouse instead of the repo's pin 5.9.0; torch 2.11.0.
- Pair directions use Tacit's `text` render, not `rich`; per-dimension scaling uses the union of the pair sets actually extracted (360 TraceSafe traces plus 1,205 HAICOSYSTEM traces), not all 2,250 TraceSafe rows.
- Step 1 and 2 save layers 8-30, Q3 saves layers 12-28; the joint and prefix probe selection grid was restricted to layers 12-28 for that reason. C grid {0.001, 0.003, 0.01, 0.1}, pools last, mean, last+mean; class-balanced logistic regression (liblinear); selection by validation macro-F1 then AUROC, which tied at 1.0 and picked C=0.001.
- Readout (2) skipped (Tacit ships no pretrained readouts). Llama-Guard-3-8B and the optional Llama-3.1-8B replication are SKIPPED-PENDING (gated access). Guards were run through transformers generation with Tacit's prompt builders, not vLLM; the scores are the unsafe share at the verdict token.
- Placebo cell exists only for the risky action (PpX). Q3 uses 8 scenarios per family (384 traces per set, not about 400) and only the four main cells; the K=2 start and mid sets are identical. Q3's per-agent probe is one role-agnostic probe trained on the Q3 train-family views with the filler pool as negatives; early-fusion hyperparameters come from Step 2. Joint and prefix probes for Q3 are retrained per set; transfer from K=2 is also stored in `results/step3/q3.json` and degrades sharply.
- Leakage criterion as pre-registered (macro-F1 at 0.5 within 0.10 of the majority baseline) passes (0.43-0.45 against 0.43), but AUROC on held-out families is 0.89-0.93, which I flag as a caveat and use as a baseline row.
- The prompt asked for 20 manually inspected traces; notes are in `results/step2/manual_inspection.md`.
- Analyses ran locally on activations copied from the cluster; extraction and guard scoring ran on the cluster. The first Q3 job (7292531) failed in 0 s on node kn159 and was resubmitted as 7294410 with that node excluded.
- Comment-free code with a casual style, as requested; no secrets were printed or logged.

## Files

`PREFLIGHT.md`, `PLAN.md`, `RESEARCH_LOG.md`, `QUESTIONS.md`; `src/` (generator `compo.py`, `gen_k2.py`, `gen_q3.py`, tests `test_data.py`, `leakage.py`, extractors `extract.py`, `extract2.py`, `guards.py`, analysis `replicate.py`, `analysis_q12.py`, `analysis_q3.py`, `figures.py`); `scripts/` (Slurm jobs); `data/k2`, `data/q3` (traces, seeds are in the code); `results/step1`, `results/step2`, `results/step3` (JSON and CSV for every table); `figures/`.
