# tacit-exp

Quick experiment building on the TACIT paper ("Agent Safety From Within", Jiao, Anderson). They predict if agent trajectory is safe by training linear probes on the hidden states of Qwen model.

*Disclosure: Claude was used to write code for the experiment and draft the below quick writeup. ~60% of the final below text is my own.*

- Tacit finds two kinds of harm, harmful content and unsafe tool use, and they sit in near normal directions
- What about harm only exists between two agents? ex: agent A is told "review the invoice, do NOT pay it" and agent B (who never saw that) pays it. each agent alone looks fine
- Questions I wanted to answer:
  1. Is "compositional harm" on the tool use direction, the content direction, or something else?
  2. What kind of readout can catch it? (one probe on whole trace, per agent probes, etc)
  3. what happens when there are way more agents (2 -> 16) and the trace gets long?

## Steps

- step 1: replicated the paper's direction results on Tracesafe + HAICOSYSTEM (mostly worked, see below)
- step 2: made a synthetic 2x2 dataset. 12 task families (pay invoice, share file, send vs draft email, delete vs archive ...) x 20 versions each. auth is restrictive or permissive, action is risky or safe. only (restrictive, risky) is unsafe. handoff text from A is identical in all 4 setups.
- step 3: filler agents so K = 2, 4, 8, 16, with the auth at the start or in the middle of the trace.
- split by family (8 train/2 val/2 test)

## results

- Tool use direction peaks at 0.986 rank acc (paper ~0.98), content ~0.74 (paper 0.73), cosine between them ~0, all good
- Placebo came out 0.60 and the window I set before was 0.49-0.59.

![rank acc](figures/rank_accuracy_vs_layer.png)

**1 - which direction**

- the compositional direction is almost orthogonal to both tacit directions (cos < 0.1 in every layer) and its stable (split half 0.94-0.99)
- BUT in early/mid layers it overlaps a lot with just "a restriction was written" direction (cosine up to 0.85), so it might not be really about the relation
- content direction does separate the pairs a bit around layers 20-21 (0.85-0.91) so not 100% clean

![cosine](figures/cosine_vs_layer.png)

**2) which readout**

- One probe on the whole trace (like with original TACIT paper) gets 1.0 macro-F1 on the test families. The one picked on validation only got 0.78 (AUROC still 1.0) (picked bad C)
- Reading the states at B's action tokens, or concatenating each agent's own states, gets ~0.89
- Per agent probes + late fusion: ranks ok (AUROC .87-.92)
- text guards: Qwen3Guard says safe to everything, AgentDoG flags every risky action no matter if it was authorized
- one agent alone is stuck at exactly 0.5 on the pairs where its view is the same, so the dataset does what its supposed to


**Q3 - more agents**

- joint probe AUROC stays 1.0 up to 16 agents but macro-F1 slides (0.78 -> 0.65), looks like a threshold thing
- at 16 agents with auth in the middle, truncation deletes the auth message in 100% of traces and it drops to AUROC 0.83 (= only seeing the risky action). 
- same K=2 trace padded with junk text to 16 agent length is fine (0.86), so length alone isnt the problem
- late fusion looks flat but kind of by accident (the filler agents were in the probe training set as negatives). early fusion drops at 16 agents too

![macro f1 vs K](figures/macro_f1_vs_K.png)

*Caveats*
Of course this is a cheap initial experiment with only fake templated data, one small model, no adversary. The "unsafe iff A and B" label is easy for linear stuff to get, so this can't really tell if you need a nonlinear readout. A version where the entity has to match would be better. Text alone can already rank the unsafe cell (tf-idf AUROC ~0.9) because restriction words are a giveaway. Small test sets were used (40 scenarios for Q2, only 16 for Q3).
