# System Constraints & Experimental Guards

## 1. Hardware & Resource Constraints

- **Hardware Target**: RTX 3050-class GPU (~4GB–8GB VRAM).
- **Execution Limits**: All training and inference benchmarks must fit comfortably within local GPU memory without aggressive gradient checkpointing (as checkpointing alters the runtime compute profile).
- **Batching**: Inference latency benchmarking begins strictly at **batch size 1** to measure genuine per-sample responsiveness before evaluating batched throughput.

## 2. Locked Scope & Architectural Boundaries

The V1 project is strictly bounded to maintain scientific rigor and clear attribution:

- **Backbone**: Exclusively `distilbert-base-uncased` (6 Transformer layers).
- **Exit Points**: Strictly Layer 2 (L2), Layer 4 (L4), and Layer 6 (L6). Do not introduce intermediate exits at L1, L3, or L5.
- **Task**: 150-class intent classification (+ OOS evaluation). No generative text modeling.
- **Non-Goals (Out of Scope for V1)**:
  - No Reinforcement Learning (RL) or policy gradient methods.
  - No quantization (INT8/FP4).
  - No Mixture of Experts (MoE) or speculative decoding.
  - No custom CUDA kernel optimization or compilation frameworks (e.g., TensorRT, ONNX) until baseline PyTorch behavior is fully characterized.

## 3. Data Integrity & Leakage Prevention

Strict experimental discipline is required to ensure valid findings:

1. **Test Set Isolation**:
   - The test set is evaluated **once** on frozen models.
   - **Never** tune exit thresholds on test data.
   - **Never** perform checkpoint selection or hyperparameter search against test set metrics.
   - **Never** compute data normalization or feature statistics over the test split.
2. **Causal Horizon (No Future Peeking)**:
   - At runtime, the controller at exit $i$ has access only to current and past representations ($h_{\le i}$).
   - Under no circumstances may future layer representations ($h_{>i}$) or future exit predictions be fed into the controller during inference.
   - Future layer information may only be leveraged during offline training as supervisory signal (e.g., whether deeper layers would have changed or corrected the prediction).

## 4. Benchmarking & Measurement Discipline

1. **Empirical Latency vs. FLOPs**:
   - Theoretical FLOP reductions often fail to yield real-world speedups due to memory bandwidth, kernel launch overhead, and controller inference cost.
   - Latency claims must be supported by empirical wall-clock measurements, not theoretical FLOP calculations alone.
2. **Timing Protocol**:
   - Always warm up the GPU prior to timing runs.
   - Ensure explicit `torch.cuda.synchronize()` calls before and after timed sections.
   - Report median (**p50**) and tail (**p95**) latencies rather than mean alone.
   - Explicitly benchmark and report **controller overhead** separately from backbone execution.
3. **Decoupled Timing**:
   - Never benchmark inference latency during training loops.
   - Separate one-time model initialization and weight loading from pure forward inference.
