# Research Notes & Future Improvements Log

## 1. Context & Motivation
During Stage 6 evaluation, we observed that while both the **Confidence Router (B3)** and **Controller MLP (B4)** successfully reduced compute on standard test data (preserving ~94% Macro-F1 at ~2.7–2.8 average depth), their behaviors diverged significantly when exposed to **1,000 Out-of-Scope (OOS) / Trick Questions**.

### The Problem Identified
- **The "Overconfident Assistant" Trap**: The Controller was trained solely on clean In-Scope training examples where the model had a high accuracy (~93.6% at Layer 2). As a result, the Controller developed an optimistic bias: it learned that inputs reaching Layer 2 are usually valid.
- **Consequence on OOS Data**: When presented with completely unfamiliar utterances, the Controller mistakenly allowed **17.6%** to exit early at Layer 2, whereas the simple Confidence Router only allowed **5.3%** to exit early.
- **Key Insight**: Softmax confidence naturally diffuses across 150 classes when given nonsense or alien inputs, naturally falling below the exit threshold ($<0.56$). A neural controller, however, can overfit to internal representation norms and fail to recognize out-of-distribution signals unless explicitly regularized.

---

## 2. Experimental Benchmark Summary

| Metric | B3 Confidence Router | B4 Controller MLP | Note |
|---|:---:|:---:|---|
| **In-Scope Avg Depth** | 2.74 layers | 2.83 layers | Both save ~53% of transformer compute |
| **In-Scope Macro-F1** | 94.19% | 93.62% | Within ~1% of full 6-layer model |
| **OOS Avg Depth** | **5.60 layers** | **3.97 layers** | B3 pushes trick queries much deeper |
| **OOS L2 Trap Rate** | **5.3%** | **17.6%** | B4 exits prematurely 3.3× more often |
| **OOS Detection AUROC** | **93.70%** | **80.41%** | B3 is significantly better at spotting aliens |

---

## 3. Concrete Action Plan for Future Iterations

To make the adaptive system truly robust beyond clean benchmark conditions, we will incorporate the following upgrades:

### Improvement 1: OOS-Aware Negative Sampling during Controller Training
- **Current Setup**: Controller trained only on $(x, y) \in \text{Train}_{\text{in-scope}}$.
- **Upgrade**: Incorporate `oos_train` (available in CLINC150) into the Controller's training dataset.
- **Supervision Rule**: For any OOS sample, enforce the target label $y_{\text{stop}} = 0$ with strong loss weighting. This directly teaches the Controller that ambiguous or alien inputs must never exit early.

### Improvement 2: Multi-Faceted Uncertainty Signals
- Instead of relying on raw hidden vectors and softmax probabilities alone, feed explicit uncertainty metrics into the Controller state vector:
  1. **Normalized Shannon Entropy**: $H(p) = -\frac{1}{\log C} \sum_{c=1}^C p_c \log p_c$ (detects flat/diffuse distributions).
  2. **Top-1 vs. Top-2 Margin**: $m = p_{(1)} - p_{(2)}$ (sharp confidence vs. near tie).
  3. **Free Energy Score**: $E(x) = -T \cdot \log \sum_{c} \exp(z_c / T)$ (provably superior to softmax max probability for OOD separation).

### Improvement 3: Trajectory Velocity & Layer Cosine Drift
- Track how the `[CLS]` representation itself evolves between Layer 0, Layer 2, and Layer 4 using cosine similarity:
  $$\text{drift}_{2 \to 4} = 1 - \frac{h_2 \cdot h_4}{\|h_2\| \|h_4\|}$$
- On normal queries, representations stabilize quickly. On tricky or ambiguous queries, representations keep shifting substantially across layers.

### Improvement 4: Adversarial & Perturbation Stress Testing
- Test the system against common voice-command degradations:
  - Typos and character noise (e.g. keyboard distance swaps).
  - Paraphrased and synonymous perturbations.
  - Prefix/suffix distractors (e.g. *"Actually wait, I meant..."*).
