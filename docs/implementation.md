# Implementation Guide & Roadmap

## 1. Dataset Specification & Pipeline

### Dataset
- **Source**: CLINC150 / OOS-EVAL ([official repository](https://github.com/clinc/oos-eval)).
- **Splits**: Standard train, validation, and test splits (150 intent classes, 100 train / 20 val / 30 test examples per class; optional out-of-scope splits for robustness evaluation).

### Preprocessing & Integrity Checks
- **Tokenizer**: Standard `distilbert-base-uncased` tokenizer.
- **Sequence Length**: Configurable max sequence length (e.g., 32 or 64; document truncation statistics during loading).
- **Mandatory Integrity Checks**:
  1. Verify total sample counts and class balance across splits.
  2. Confirm 1-to-1 label-to-ID mapping consistency across all exit heads.
  3. Ensure zero overlap/leakage between train, validation, and test sets.
  4. Record utterance length distributions before and after tokenization.

## 2. Step-by-Step Implementation Roadmap

The project proceeds in structured stages. Each stage verifies prerequisites before advancing:

```text
Stage 0: Environment Verification
   │
   ▼
Stage 1: Dataset Loading & Sanity Checks
   │
   ▼
Stage 2: Full-Depth Baseline (L6)
   │
   ▼
Stage 3: Multi-Exit Backbone Fine-Tuning (L2, L4, L6)
   │
   ▼
Stage 4: Fixed-Exit Evaluation (L2, L4, L6 standalone)
   │
   ▼
Stage 5: Confidence-Based Dynamic Exit
   │
   ▼
Stage 6: Controller Feature Extraction & Training
   │
   ▼
Stage 7: Threshold Sweeping & Frontier Curves
   │
   ▼
Stage 8: Dedicated Latency & Systems Benchmarking
```

### Stage Details

- **Stage 0 — Environment & Scaffold**:
  Verify Python, PyTorch, CUDA, Hugging Face `transformers`, `datasets`, and `scikit-learn`. Verify GPU memory and device detection.
- **Stage 1 — Data Sanity**:
  Build dataset loaders; compute sequence statistics; verify label mappings; confirm train/val/test split counts.
- **Stage 2 — Full-Depth L6 Baseline (`E001`)**:
  Fine-tune standard DistilBERT on L6 to establish the reference quality ceiling (Macro F1, accuracy).
- **Stage 3 — Multi-Exit Backbone (`E002`)**:
  Attach lightweight linear heads to Layer 2, Layer 4, and Layer 6. Train with joint auxiliary loss ($w_2=0.3, w_4=0.3, w_6=1.0$).
- **Stage 4 — Fixed-Exit Profiling (`E003`)**:
  Evaluate the trained multi-exit backbone at fixed L2, fixed L4, and L6 without dynamic routing to measure intermediate capability.
- **Stage 5 — Confidence Early Exit (`E004`)**:
  Implement confidence routing based on maximum softmax probability. Calibrate thresholds $\tau_2, \tau_4$ on the validation set.
- **Stage 6 — Controller Implementation & Training (`E005`)**:
  Extract offline states $[z(h_i), \text{logits}_i, \text{conf}_i, \Delta\text{pred}_i]$ and supervisory exit labels from the frozen backbone. Train the lightweight controller MLP.
- **Stage 7 — Threshold Sweeps & Trade-off Analysis (`E006`, `E007`)**:
  Sweep decision thresholds on validation data. Plot quality vs. average exit depth and quality vs. wall-clock latency. Evaluate feature ablations.
- **Stage 8 — Hardware Latency Benchmarking (`E008`)**:
  Execute isolated latency benchmarks on GPU (batch size 1, 500 repetitions, CUDA sync, warmup) measuring both backbone layers and controller overhead.

## 3. Engineering & Reproducibility Guidelines

- **Configurability**: Hyperparameters (learning rates, loss weights, batch sizes, sequence lengths, thresholds) must be specified via configuration files.
- **Seed Management**: Global random seeds set across Python, NumPy, and PyTorch.
- **Checkpointing**: Checkpoint the best model based exclusively on validation metrics. Never overwrite best checkpoints without versioning.
- **Verification Tests**:
  1. Forward-pass smoke test on dummy batch.
  2. Overfit-small-batch test (verify model converges on 10 examples).
  3. Layer indexing assertion (confirm exact layers corresponding to L2, L4, L6).

## 4. Implementation Checklist

### Repository & Setup
- [ ] Clean directory structure established
- [ ] Environment and dependencies documented
- [ ] Centralized configuration system implemented
- [ ] Seed utility and deterministic execution verified
- [ ] Logging and checkpoint management implemented

### Dataset
- [ ] CLINC150 dataset downloaded and parsed
- [ ] Split integrity and class counts verified
- [ ] Label mapping constructed and frozen
- [ ] Tokenization sequence-length statistics reported
- [ ] Train/validation/test isolation confirmed

### Model & Backbone
- [ ] DistilBERT loaded and layer indexing verified
- [ ] L2, L4, and L6 classification heads implemented
- [ ] Forward smoke test completed
- [ ] Overfit-small-batch test completed
- [ ] Joint auxiliary loss training implemented

### Controller & Routing
- [ ] Feature representation builder implemented ($z(h)$, logits, confidence, trajectory $\Delta$)
- [ ] Controller MLP architecture defined
- [ ] Offline supervisory target generator built
- [ ] Controller trained and evaluated on validation split
- [ ] Threshold selection logic calibrated on validation set

### Evaluation & Systems
- [ ] Full-depth L6 baseline measured
- [ ] Fixed L2 and Fixed L4 baselines measured
- [ ] Confidence-based routing baseline evaluated
- [ ] Controller routing evaluated across threshold sweeps
- [ ] Signal ablation experiments completed
- [ ] Wall-clock latency benchmarked (batch size 1, warmup, CUDA sync)
- [ ] Controller inference overhead quantified
- [ ] Final experiment comparison table and Pareto frontier plotted
