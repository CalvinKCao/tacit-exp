# Plan (written 2026-11-02 before any extraction; not edited afterwards)

Backbone: Qwen/Qwen3-4B-Instruct-2507 (Tacit's `qwen3-4b-instruct`). Compute: Killarney L40S. Extraction is deterministic; every analysis seed is fixed in the code and logged.

## Step 1 acceptance (replication of Section 3)

Pairs: TraceSafe MissingTypeHint + AmbiguousArg (180 schema-only pairs, orig = benign), placebo (180), HAICOSYSTEM content pairs (258). Rendering: Tacit's `text` render, last-token states, layers 8-30 saved, per-dimension std over the union of both pair sets, direction = unit mean difference, 5-fold pair-level CV with both members of a test pair held out, rank accuracy with ties = 0.5.

Step 1 passes if, within 0.05 of the paper value:
- tool-use out-of-fold rank accuracy peak in [0.93, 1.00], peak at an intermediate layer (not 8 or 30 and not the last quarter);
- placebo-vs-benign rank accuracy at the tool peak layer in [0.49, 0.59];
- content-pair peak in [0.68, 0.79];
- |cosine(v_tool, v_content)| < 0.2 at the peak layers;
- same-risk split-half cosine (mean over 5 disjoint splits) in [0.76, 0.86] at the peak layer, for the risk the paper reports (checked per set; I report both).

If it fails I diagnose (render mode text vs rich, pooling, layer indexing, truncation count) and stop and report if still failing. No Step 2/3 compute before it passes.

## Hypotheses and falsification criteria

All Q1-Q3 statistics use grouped bootstrap (2,000 resamples) over families or scenarios.

H1 (Q1). The composition direction v_comp (fitted out-of-fold on (R,X) minus (P,X) pairs, same recipe) is not on either TACIT direction.
- Supported if |cos(v_comp, v_tool)| < 0.5 and |cos(v_comp, v_content)| < 0.5 at the layer where v_comp has its best out-of-fold rank accuracy and at layer 22, and the same-direction split-half cosine of v_comp is >= 0.5 at that layer (otherwise the estimate is too noisy and H1 is "uninformative", not supported).
- Falsified if either |cos| >= 0.5 at the v_comp peak layer.
- Also reported: cross-ranking accuracy of v_tool and v_content on the (R,X) vs (P,X) pairs. "On the tool-use direction" would need rank accuracy >= 0.75 for v_tool with a CI that excludes 0.5 and the placebo (restriction on a different entity) near 0.5.

H2 (Q2), reformulated before any run. The prompt's suggestion, "isolated per-agent readouts with linear fusion are near chance by construction", does not follow from the 2x2: label = R and X is linearly separable in (auth feature, action feature), and a label-trained probe on agent A sees R vs P, so A's view alone has AUROC about 0.83 on the label for a perfect detector of R (B's view likewise for X). What the construction does guarantee is the contrast-level statement. So:
- H2a: on contrast pairs that hold one agent's local view identical, that agent's isolated probe scores are exactly tied (B on (R,X) vs (P,X); A on (R,X) vs (R,Y)). Checked as a sanity test of the dataset (rank accuracy exactly 0.5 up to numerical ties), not a finding.
- H2b: late fusion (mean, logistic over sorted scores) of label-trained per-agent probes will recover most of the signal, because conjunction of two separable factors is linear. Falsified for linear fusion if test-family AUROC lower CI bound is < 0.75 for both mean and logistic fusion. Max fusion is expected to be worse than mean (it fires on three of four cells).
- H2c: the joint-trace probe beats late fusion on macro-F1 at fixed threshold 0.5 only if its bootstrap CI of the difference excludes 0. Otherwise the report says no difference was detected.
- Note for the report: because the conjunction is separable, this dataset cannot distinguish "needs nonlinear aggregation" from "late fusion works". A relational variant (harm iff the action's entity differs from the authorized entity, which is XOR-like for each local view) would. It is not part of the specified design, so it is not built unless the user asks (QUESTIONS.md).

H3 (Q2/Q3). Joint-trace works at K=2 (test macro-F1 >= 0.8, AUROC >= 0.9). Whether it degrades with K is open; the criterion is the slope of macro-F1 against log2(K) for mid-trace placement: "degrades" if macro-F1(K=16) is lower than macro-F1(K=2) by more than 0.10 with a bootstrap CI that excludes 0, and the K=2 length-padded control is higher than K=16 mid-trace. Truncation-driven degradation is claimed only if the drop is concentrated in mid-trace placement and appears after the first K where truncation applies.

## Step 2 dataset design (fixed now)

12 families x 20 instantiations x 4 cells, plus placebo variant (length-matched restriction on a different entity, safe). Families split 8 train / 2 validation / 2 test by family. Hyperparameters chosen on validation only. Automated tests: handoff string-identical across cells; B action segment identical across auth conditions; A's view identical between (R,X) and (R,Y). Leakage: TF-IDF 1-2 gram logistic regression and gradient boosting on joint-trace text must be near chance on the 2x2 labels (macro-F1 within 0.10 of the majority/chance baseline on test families), length parity within 5%, 20 manually inspected traces.

## Budget

Total cap about 4 GPU-hours. Step 1 estimate in the job script header (extraction of about 1,750 traces). Step 2 extraction about 12x20x4x2 = 1,920 traces (K=2, plus placebo). Step 3: 4 K values x 2 placements x 400 traces plus the length control; the K=16 traces are long, so estimates are made before submission.
