# EXP-001: Similarity threshold calibration

**Status:** ✅ adopted (drove ADR-0002)

## Hypothesis

Raising the dense-retrieval similarity floor from the common ~0.20 default to a
stricter value would sharply reduce irrelevant chunks reaching the LLM (and thus
confabulation) without a proportional loss in recall, because BM25 covers the
exact-term queries where dense retrieval is weakest.

## Setup

- Retrieval: this pipeline (BM25 + dense + RRF)
- Variable: `similarityThreshold` on the dense retriever
- TODO: Document the query set (size, how queries were selected/labeled) used
  for calibration.
- TODO: Document the corpus (size, domain) the calibration ran against.

## Method

Swept the dense threshold and compared retrieval quality between the permissive
default and the stricter floor, holding everything else constant.

- TODO: Add the exact thresholds swept and the metric used (precision / recall /
  confabulation rate — define how confabulation was measured).

## Results

- TODO: Add the results table (threshold vs. metric) from the calibration run.
  The headline finding to document: threshold choice dominated other tuning —
  moving the floor mattered more than prompt-level changes.

## Conclusion

Adopted a default floor of 0.60 (ADR-0002). The floor is exposed as config
because the optimal value is corpus- and embedding-model-dependent; 0.60 is the
default, not a universal constant.

## What surprised me

TODO: Note what was unexpected — e.g. how large the confabulation swing was
relative to how small the parameter change looked.
