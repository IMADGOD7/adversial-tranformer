# Experiments, Baselines & Evaluation Protocol

## 1. Baseline Systems

To assess the value of the adaptive controller, five core systems are compared:

- **B0 — Full-Depth L6**: Standard non-adaptive inference executing all 6 layers. Serves as the primary quality reference ceiling.
- **B1 — Fixed L2**: All inputs exit unconditionally at Layer 2. Defines the minimum compute / maximum speedup bound.
- **B2 — Fixed L4**: All inputs exit unconditionally at Layer 4. Intermediate static reference point.
- **B3 — Confidence-Based Exit**: Dynamic exit where Layer 2 or Layer 4 exits if $\max_c \text{Softmax}(\text{logits}_i)_c \ge \tau_{\text{conf}}$. Thresholds are calibrated on the validation set.
- **B4 — Proposed Controller**: Dynamic exit guided by the lightweight MLP controller evaluating representation $z(h_i)$, logits, confidence, and trajectory $\Delta$.

## 2. Required Signal Ablations

To isolate the marginal value of each controller feature, four variants must be evaluated under identical backbone weights:

1. **Ablation 1 (Confidence Only)**: Controller receives only the scalar confidence value.
2. **Ablation 2 (Confidence + Trajectory)**: Controller receives confidence and $\Delta\text{prediction}$.
3. **Ablation 3 (Trajectory + Representation)**: Controller receives $\Delta\text{prediction}$ and compact representation $z(h_i)$, without explicit scalar confidence.
4. **Ablation 4 (Full Controller)**: All features combined ($z(h_i)$, normalized logits, confidence, $\Delta\text{prediction}$).

## 3. Evaluation Metrics

### Task Quality
- **Macro F1**: Primary classification metric across the 150 in-scope intent classes.
- **Top-1 Accuracy**: Overall accuracy on in-scope data.
- **Out-of-Scope (OOS) Performance**: Detection accuracy / AUROC on out-of-scope examples.

### Compute & Efficiency
- **Average Exit Depth**: $\mathbb{E}[\text{exit}] = \frac{2 N_2 + 4 N_4 + 6 N_6}{N_{\text{total}}}$
- **Exit Distribution**: Percentage of queries exiting at L2, L4, and L6.

### Systems Performance
- **Wall-Clock Latency**: Measured in milliseconds at **batch size 1** on the target GPU. Reported as **p50 (median)** and **p95 (tail)**.
- **Throughput**: Sequences processed per second under batched inference.
- **Peak VRAM**: Maximum allocated memory during inference.
- **Controller Overhead**: Wall-clock time spent inside the controller module relative to Transformer block execution.

### Primary Comparison Table

| System | Macro F1 | Avg Exit Depth | p50 Latency (ms) | p95 Latency (ms) | Throughput (seq/s) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Full L6 (B0) | | 6.0 | | | | |
| Fixed L2 (B1) | | 2.0 | | | | |
| Fixed L4 (B2) | | 4.0 | | | | |
| Confidence (B3) | | | | | | |
| Controller (B4) | | | | | | |

## 4. Diagnostic & Stratified Analyses

### Exit Depth × Correctness Contingency
Every evaluation run must tabulate sample counts and percentages across the $2 \times 3$ matrix:
- L2 Correct / L2 Incorrect
- L4 Correct / L4 Incorrect
- L6 Correct / L6 Incorrect

### Stratified Sub-Analyses
Exit distributions and error rates must be analyzed across:
1. **In-scope vs. OOS**: Determine whether unfamiliar utterances naturally propagate deeper or exit prematurely.
2. **Sequence Length Buckets**: Short ($\le 8$ tokens), Medium (9–16 tokens), Long ($> 16$ tokens).
3. **Confidence Buckets**: Binned confidence intervals vs. actual exit decisions.

### Quality-Compute Frontier Curve
By sweeping the exit threshold $\tau \in [0.0, 1.0]$ on the validation set, plot:
1. **Macro F1 vs. Average Exit Depth**
2. **Macro F1 vs. Measured Wall-Clock Latency (ms)**

The central comparison is whether the proposed controller dominates the confidence baseline along the Pareto frontier.

## 5. Latency Benchmarking Protocol

To prevent noisy or misleading runtime measurements:
1. Warm up the GPU with 50 dummy passes before recording.
2. Place explicit `torch.cuda.synchronize()` barriers around timed code blocks.
3. Perform a minimum of 500 repetitions for batch size 1.
4. Isolate controller forward time from backbone layer forward time.

## 6. Experiment Registry & Tracking

Each experiment is assigned a canonical identifier:
- `E001_full_l6`: Baseline L6 fine-tuning and evaluation.
- `E002_multiexit_backbone`: Multi-exit backbone fine-tuning ($L_{2,4,6}$ losses).
- `E003_fixed_exits`: Static L2, L4, and L6 evaluation.
- `E004_confidence_exit`: Confidence threshold tuning and evaluation.
- `E005_controller_training`: Training the proposed MLP controller.
- `E006_signal_ablations`: Signal ablation evaluations (A1–A4).
- `E007_threshold_sweep`: Pareto frontier generation via threshold sweeps.
- `E008_latency_benchmark`: Dedicated systems benchmarking on hardware.

Every run must log: random seed, checkpoint path, git commit / code version, hyperparameters, exit distribution, validation/test metrics, and hardware specs.

### Stop Criteria & Negative Results
If empirical testing reveals that:
- The controller fails to outperform simple confidence thresholding on the Pareto frontier, or
- Controller computational overhead negates layer-skipping latency savings,
the findings must be reported honestly. Negative or neutral outcomes regarding the controller hypothesis are completely valid results.
