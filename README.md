# Adaptive Transformer Inference (V1)

A research-grade ML systems experiment investigating compute-adaptive Transformer inference using intermediate exit heads and a lightweight multi-signal controller.

## Core Research Question

> Can a lightweight controller using intermediate representations, predictions, confidence scores, and prediction trajectories allocate Transformer computation more efficiently than simple confidence thresholds while maintaining task quality?

The goal is to solve the optimization problem:
$$\min \mathbb{E}[\text{exit\_depth}] \quad \text{subject to} \quad \text{Macro-F1} \ge \text{Macro-F1}_{\text{full}} - \epsilon$$

---

## Locked V1 Specification

- **Task**: Intent classification on CLINC150 / OOS-EVAL (150 in-scope classes + out-of-scope evaluation).
- **Backbone**: `distilbert-base-uncased` (6 Transformer layers).
- **Candidate Exits**: Layer 2 (L2), Layer 4 (L4), and Layer 6 (L6).
- **Controller**: Lightweight MLP evaluating:
  1. Compact projection of intermediate hidden representation $z(h_i)$
  2. Normalized class logits
  3. Maximum softmax confidence
  4. Prediction trajectory $\Delta$ (change from prior exit)
- **Baseline Models**:
  1. Full-Depth L6 (upper quality reference)
  2. Fixed L2 (minimum compute reference)
  3. Fixed L4 (intermediate static reference)
  4. Confidence-based early exit (validation-calibrated threshold)
  5. Proposed Multi-Signal Controller
- **Target Hardware**: RTX 3050-class GPU (real batch-1 wall-clock latency measurement).

---

## Documentation Index

Detailed specifications and protocols are organized into five documentation modules in [`docs/`](docs/):

1. **[docs/problem.md](docs/problem.md)**: Research motivation, formal hypotheses, literature positioning (LayerSkip comparison), and core objectives.
2. **[docs/constraints.md](docs/constraints.md)**: Hardware constraints, strict scope boundaries, non-goals, anti-leakage rules, and timing protocols.
3. **[docs/architecture.md](docs/architecture.md)**: Complete system design, DistilBERT layer mapping, auxiliary multi-exit loss formulation, and controller architecture.
4. **[docs/experiments.md](docs/experiments.md)**: Baseline definitions, signal ablations, evaluation metrics, latency benchmarking protocol, and experiment registry.
5. **[docs/implementation.md](docs/implementation.md)**: Dataset pipeline, step-by-step implementation roadmap (Stages 0–8), and pre-completion checklist.
