# EXP-001 — Threshold Calibration

## Question

How does the dense-retrieval similarity threshold affect precision and recall
for Portuguese clinical queries?

## Why this matters

Threshold selection directly affects retrieval quality and, downstream, the
grounding of generated answers. A floor set too low admits weakly-related
chunks that invite confabulation; set too high, it discards useful context.
This single parameter tends to dominate other tuning.

## Status

Re-running the experiment to publish reproducible metrics.

The original threshold calibration was performed during iterative development
and was not captured in a reproducible format. Rather than reconstructing
historical numbers from memory, this experiment is being repeated with a
documented methodology and a versioned dataset.

## Current observation

The production configuration uses the calibrated threshold derived during
development (the constructor default, 0.60). The purpose of this experiment is
to publish the methodology and measurements behind that decision — not to
justify it after the fact.

## Next step

Publish, as a self-contained and reproducible artifact:

- the evaluation dataset (versioned)
- the evaluation protocol
- the metrics (precision / recall across thresholds)
- the analysis

## Related

- ADR-0002 records the decision this experiment documents.
