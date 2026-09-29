# Problem Statement & Research Context

## 1. Motivation

Standard Transformer inference evaluates all layers uniformly regardless of sample difficulty. A fixed-depth architecture expends unnecessary computation on straightforward examples, while complex queries may genuinely require deeper contextual representations.

Early-exit mechanisms address this inefficiency by enabling intermediate representations to yield predictions. However, conventional early-exit strategies rely almost exclusively on output confidence (such as maximum softmax probability or entropy). This presents key limitations:
- **Confidence is not accuracy**: Models can be overconfident on misclassified or ambiguous tokens.
- **Snapshot blindness**: Single-layer confidence metrics lack visibility into how the prediction trajectory evolved across prior layers.

## 2. Research Question & Hypothesis

### Core Question
> Can a lightweight controller utilizing intermediate representations, current predictions, confidence scores, and prediction trajectories allocate Transformer computation more efficiently than simple confidence thresholds while maintaining task quality?

### Hypothesis
A controller integrating multi-layer signals—specifically the trajectory of predictions between layers—achieves a superior quality-versus-compute Pareto frontier compared to confidence-only early exiting. 

This hypothesis is falsifiable: empirical evidence showing that trajectory and intermediate representations add negligible routing value over calibrated confidence is a valid and acceptable scientific outcome.

## 3. Scope & Task

- **Task**: Intent classification over the 150 in-scope intent classes of the **CLINC150 (OOS-EVAL)** benchmark, plus evaluation on Out-Of-Scope (OOS) utterances.
- **Backbone**: `distilbert-base-uncased` (6 Transformer layers).
- **Candidate Exits**: Exits located strictly at Layer 2 (L2), Layer 4 (L4), and Layer 6 (L6).

## 4. Literature Context & Prior Work

This study is an ML systems evaluation, not a claim of a novel Transformer backbone or the general concept of early exit.

### Relationship to LayerSkip
- [LayerSkip (Elhoushi et al., 2024)](https://aclanthology.org/2024.acl-long.681/) demonstrates that standard pretrained Transformer layers are often poorly suited for immediate early exit without dedicated training (e.g., layer dropout and intermediate auxiliary losses), and emphasizes accurate wall-clock latency benchmarking.
- **Distinct Focus of this Project**: Rather than focusing solely on training regularization or speculative decoding, this project examines the **runtime routing controller**: specifically evaluating whether trajectory and representation features justify their computational overhead over standard confidence-threshold routing in small-scale models.

### Novelty Discipline
Any performance or novelty claim must be grounded in direct comparison with existing adaptive-depth mechanisms. Controller inputs, training targets, and evaluation protocols must be contrasted explicitly with standard confidence baselines.

## 5. Objectives

### Primary Objective
Minimize expected computational depth subject to an explicit task-quality constraint:
$$\min \mathbb{E}[\text{exit\_depth}] \quad \text{subject to} \quad \text{Macro-F1} \ge \text{Macro-F1}_{\text{full}} - \epsilon$$

### Secondary Objectives
1. **Routing Mechanics**: Characterize when and why early exits succeed or fail.
2. **Signal Attribution**: Isolate whether prediction trajectory ($\Delta$) provides genuine signal beyond output confidence.
3. **Controller Overhead**: Measure wall-clock latency of the controller to determine if theoretical FLOP reductions yield real inference speedups on consumer hardware.
4. **OOS Behavior**: Analyze whether out-of-scope samples naturally defer to deeper layers or trigger premature exits.
