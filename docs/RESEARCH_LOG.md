# Research log

2026-11-02 preflight
- cloned CSSLab/Tacit at 995692d, fetched paper pdf, TraceSafe (13 golden files) and HAICOSYSTEM (episodes, environments)
- tacit loaders: tracesafe n=2250 unsafe=1080, haico n=7937; build_haico_pairs gives 258 content pairs, placebo gives 180 records
- no pretrained readouts in the repo
- gpu checks not done locally; scripts/slurm_preflight.sh written, waiting on the user to run it on killarney

2026-11-02 step 1 submitted
- preflight job 7202474 ok: L40S 44.4 GB, sdpa ok, flash-attn missing, torch 2.11.0, transformers 5.22.1 from wheelhouse (repo pins 5.9.0, deviation), compute nodes reach github/hf/pypi, scratch free 166 GB, ram 503 GB
- PLAN.md written before extraction. h2 reformulated: label = R and X is linearly separable so late fusion is not chance by construction (see plan)
- token counts (text render, qwen tokenizer): tracesafe pair traces 360 (about 2.0M tokens, 8 over 16384 per 180), placebo 180 (1.0M), haico pairs 1205 (2.7M, 1 over 16384)
- first version of the pair filter dropped the -orig members (180 instead of 360); fixed before submission
- job 7203033: extract 3 sets + replicate.py, estimate 17 min, requested 25 min

2026-11-04 step 1 analysis (run locally, activations rsynced from cluster; job 7269137 cancelled)
- tool peak 0.986 @L15, content peak 0.736 @L20, |cos| ~0, split-half 0.81/0.83: pass
- placebo at tool peak 0.600 vs window 0.49-0.59: fails the preregistered window by 0.01

2026-11-05 step 2 + q3 data built, jobs submitted
- compo.py: 12 families x 20 instantiations x 5 cells (RX RY PX PY + placebo PpX) = 1200 joint traces, 1200 local views (A: R/P/Pp, B: X/Y)
- test_data.py passes: handoff identical across the 4 cells, B action+result identical across auth, A prefix identical between RX and RY, B-local suffix identical between RX and PX, only RX label 1, token length gap across cells <= 4.7%, placebo restriction within 13 chars of the R clause
- first version failed length parity (6.4%) and placebo length matching; fixed by lengthening the safe-action strings and matching the placebo restriction to the R restriction clause (not the whole message, since P + restriction is always longer than R)
- leakage (train families -> held-out families): tf-idf logreg/gbt macro-F1 at 0.5 = 0.43-0.45 (majority 0.43) so the preregistered macro-F1 criterion passes, but AUROC is 0.89-0.93: restriction wording ranks the unsafe cell lexically. expected from the AND structure, reported as a caveat. within-family random split gbt reaches macro-F1 0.96 (diagnostic only)
- manual inspection notes in results/step2/manual_inspection.md
- q3 data: 8 sets (k2, k4/k8/k16 x start/mid, k2_pad) x 384 traces, 584 views, N_SENT=48 gives k16 mean 18.5k tokens: truncation first applies at K=16 (all traces), K=16 mid drops the auth message in 100% of traces, K=8 never truncates
- jobs: 7292424 step2 (extract joint+views, guards qwen3guard/agentdog, est 14 min, asked 21), 7292531 q3 (est 56 min, asked 84)
- deviation: joint/prefix probe layer grid limited to 12-28 since q3 stores layers 12-28 only

2026-11-05 step 2 results
- job 7292424 completed in 15 min (estimate 14, asked 21). activations k2_joint/k2_views and guard scores pulled to results/
- q1 (results/step2/q1.json): v_comp (R,X)-(P,X) has |cos| < 0.1 with v_tool and v_content at every layer, split-half 0.87-0.99. but cos with the auth-only direction (R,Y)-(P,Y) is 0.4-0.9 at layers 8-19 and negative after layer 20, so in the middle layers v_comp overlaps with plain restriction-wording. placebo (unrelated restriction) also raises v_comp scores vs (P,X) (rank 0.78-0.93)
- q2 (results/step2/q2.json): joint probe selected on val (last, L19, C=0.001) gives AUROC 1.0 but macro-F1 0.78 at 0.5 on test; paper settings (L22, last+mean, C=0.003) give 1.0 / 1.0. val selection tied at F1 1.0 and picked a badly calibrated C. isolated early fusion f1 0.90 / auc 0.985; late fusion mean f1 0.66 auc 0.92; logistic 0.70 / 0.87; max 0.27 / 0.82. qwen3guard flags nothing (f1 0.43) but auc 0.99; agentdog flags every X trace regardless of auth (auc 0.83)
- h2a check holds exactly: agent-B-only ranks (R,X) vs (P,X) at 0.500, agent-A-only (R,X) vs (R,Y) at 0.500
- figures/rank_accuracy_vs_layer.png and figures/cosine_vs_layer.png written (src/figures.py), q3 figure waits on job 7292531 (still pending in the queue)

2026-11-05 job 7292531 (q3) failed in 0 s on node kn159 with no log (exited before opening the log file), likely node issue. resubmitted with --exclude=kn159, same 84 min request.

2026-11-05 q3 and report
- job 7292531 failed in 0 s on kn159 (no log), resubmitted as 7294410 excluding kn159, completed in 43 min on kn135 (estimate 56, asked 84)
- q3 analysis ran locally (results/step3). joint probe: auroc 1.00 through k16 start and k8 mid, macro-F1 0.78 -> 0.65 (k16 start) -> 0.50 (k16 mid, auth dropped by truncation in 100% of traces, auroc 0.83). k2 padded control 0.86. late logistic flat by construction (filler views were probe training negatives). early fusion drops at k16 (0.62)
- while writing REPORT.md I found that my first draft claim 'neither TACIT direction separates (R,X) from (P,X)' was wrong: v_content ranks them at 0.85-0.91 at layers 20-21 and v_tool 0.78 at layer 21, with unstable sign elsewhere; corrected in the report
- total GPU time about 1.2 h of the 4 h cap
- figures/macro_f1_vs_K.png written
