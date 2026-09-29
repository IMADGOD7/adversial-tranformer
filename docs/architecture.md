# System Architecture Specification

## 1. High-Level Architecture

The system augments a 6-layer DistilBERT backbone with intermediate classification heads at Layer 2 (L2), Layer 4 (L4), and Layer 6 (L6), paired with lightweight routing controllers at intermediate exits:

```text
INPUT (Tokenized sequence)
  │
  ▼
[DistilBERT Embeddings]
  │
  ▼
[Transformer Layer 1]
  │
  ▼
[Transformer Layer 2] ────► [L2 Exit Head] ────► [Controller L2]
  │                                                    │
  │                                               STOP / CONTINUE
  │                                                    │
  │ (CONTINUE)                                    STOP ──► Yield L2 Prediction
  ▼
[Transformer Layer 3]
  │
  ▼
[Transformer Layer 4] ────► [L4 Exit Head] ────► [Controller L4]
  │                                                    │
  │                                               STOP / CONTINUE
  │                                                    │
  │ (CONTINUE)                                    STOP ──► Yield L4 Prediction
  ▼
[Transformer Layer 5]
  │
  ▼
[Transformer Layer 6] ────► [L6 Final Head] ───► Yield L6 Prediction
```

## 2. Backbone & Layer Indexing

- **Model Backbone**: `distilbert-base-uncased` (6 Transformer layers).
- **Layer Mapping**:
  In Hugging Face DistilBERT (`transformer.layer`), module indexing is 0-indexed:
  - Conceptual **L2** $\rightarrow$ `transformer.layer[1]`
  - Conceptual **L4** $\rightarrow$ `transformer.layer[3]`
  - Conceptual **L6** $\rightarrow$ `transformer.layer[5]`
  *Note*: This mapping must be verified programmatically via runtime inspection before training.

## 3. Exit Representations & Classification Heads

- **Pooling Strategy**:
  At each exit layer $i \in \{2, 4, 6\}$, let $H_i \in \mathbb{R}^{B \times L \times D}$ be the hidden states. The exit representation $h_i$ uses the standard DistilBERT classification pooling convention (extracting the first token `[CLS]` representation: $h_i = H_i[:, 0, :]$). This pooling must remain identical across all exit heads.
- **Classification Heads**:
  Each candidate exit features a lightweight linear classification head:
  $$\text{logits}_i = W_i h_i + b_i \quad (W_i \in \mathbb{R}^{C \times D})$$
  where $C = 150$ is the shared intent label space.

## 4. Multi-Exit Backbone Fine-Tuning

To ensure intermediate representations at L2 and L4 are capable of reliable predictions, the backbone is fine-tuned with auxiliary exit losses:

$$\mathcal{L}_{\text{backbone}} = w_2 \mathcal{L}_{\text{CE}}(y, \text{logits}_2) + w_4 \mathcal{L}_{\text{CE}}(y, \text{logits}_4) + w_6 \mathcal{L}_{\text{CE}}(y, \text{logits}_6)$$

- Loss weights $w_2, w_4, w_6$ are fully configurable (default baseline: $w_2=0.3, w_4=0.3, w_6=1.0$).
- Weights are balanced to maintain L6 terminal quality while adapting intermediate features.

## 5. Controller Design

At exits L2 and L4, a lightweight controller decides whether to halt execution ($\text{STOP}$) or proceed to the next block ($\text{CONTINUE}$).

### State Input Representation
For exit $i \in \{2, 4\}$, the controller receives a feature vector:
$$\text{state}_i = \left[ z(h_i), \, \text{norm\_logits}_i, \, \text{confidence}_i, \, \Delta\text{prediction}_i \right]$$

1. **Compact Representation $z(h_i)$**: A small linear projection or dimension-reduced summary of the pooled representation $h_i$ (e.g., $D \to d_{\text{proj}}$).
2. **Logit Features $\text{norm\_logits}_i$**: Softmax distribution or temperature-normalized logits over class predictions.
3. **Confidence $\text{confidence}_i$**: A scalar confidence metric, defined as the Maximum Softmax Probability:
   $$\text{conf}_i = \max_c \, \text{Softmax}(\text{logits}_i)_c$$
4. **Prediction Trajectory $\Delta\text{prediction}_i$**:
   - **At L2**: No preceding exit exists. $\Delta\text{prediction}_2$ is defined as a fixed zero-vector of matching dimension.
   - **At L4**: Captures prediction shift between L2 and L4:
     $$\Delta\text{prediction}_4 = \text{logits}_4 - \text{logits}_2$$
     (or probability distribution shift $\|P_4 - P_2\|$, depending on configured ablation).

### Controller Architecture
- A small multi-layer perceptron (MLP):
  $$\text{Input} \to \text{Linear}(d_{\text{in}}, d_{\text{hidden}}) \to \text{ReLU} \to \text{Linear}(d_{\text{hidden}}, 1) \to \text{Sigmoid}$$
- Generates probability $P(\text{STOP} \mid \text{state}_i) \in [0, 1]$.
- Hidden dimension $d_{\text{hidden}}$ is kept small (e.g., 64) to minimize inference latency overhead.

### Inference Decision Rule
$$\text{Action}_i = \begin{cases} \text{STOP} & \text{if } P(\text{STOP} \mid \text{state}_i) \ge \tau_i \\ \text{CONTINUE} & \text{otherwise} \end{cases}$$
Thresholds $\tau_i$ are tuned strictly on validation data to achieve target quality-compute points.

## 6. Controller Training Strategy

- **Decoupled Training**: Backbone fine-tuning and controller training are decoupled. Intermediate features and logits are cached or passed from the frozen multi-exit backbone.
- **Supervision Target**: Controller targets are constructed offline using ground truth and downstream exit outcomes:
  - Label $y_{\text{stop}} = 1$ if exit $i$ is correct and deeper computation provides negligible utility or error correction.
  - Label $y_{\text{stop}} = 0$ if exit $i$ is incorrect and a subsequent exit (L4 or L6) corrects the prediction.
- Binary Cross-Entropy loss is used to train the controller MLP.
