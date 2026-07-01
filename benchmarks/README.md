# Benchmarks

## Purpose

Reproducible benchmark suites for the retrieval pipeline. Every published
number should regenerate from scratch with a documented method and environment.

## Status

The benchmark harness is being standardized so that every result can be
reproduced independently.

The initial measurements were taken during active product development, coupled
to production-specific assumptions. Before publishing performance numbers, the
harness is being isolated into a standalone engineering artifact with a fixed
dataset and a documented environment.

## Planned suites

- Retrieval latency (percentiles, not means)
- BM25 vs. dense vs. hybrid recall by query type
- Fusion strategy comparison
- Threshold calibration (see experiments/EXP-001)

## Principle

A benchmark without its methodology is marketing. Numbers are published here
only alongside the environment and command that reproduce them.
