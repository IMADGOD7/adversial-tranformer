import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitControllerMLP(nn.Module):
    """Lightweight Multi-Layer Perceptron (MLP) routing controller.

    Evaluates whether an intermediate exit is reliable enough to STOP execution.
    Input features:
      - z(h): projected hidden representation [CLS] (e.g. 32-dim)
      - norm_logits: softmax probabilities over classes (150-dim)
      - confidence: scalar maximum softmax probability (1-dim)
      - delta_prediction: change in logits from preceding exit (150-dim)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = 150,
        proj_dim: int = 32,
        mlp_hidden: int = 64,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)

        # Feature dimension: proj_dim + num_classes (probs) + 1 (conf) + num_classes (delta)
        in_features = proj_dim + num_classes + 1 + num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden, 1),
        )

    def extract_state_vector(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        prev_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """Constructs the combined state vector for the controller.

        Args:
            h: (B, hidden_dim) pooled [CLS] hidden state.
            logits: (B, num_classes) current exit raw logits.
            prev_logits: (B, num_classes) previous exit logits (optional, for delta).

        Returns:
            state: (B, in_features) concatenated feature vector.
        """
        # 1. Compact projection z(h)
        z = F.relu(self.proj(h))

        # 2. Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # 3. Confidence scalar (max prob)
        conf = torch.max(probs, dim=-1, keepdim=True)[0]

        # 4. Trajectory delta
        if prev_logits is not None:
            delta = logits - prev_logits
        else:
            delta = torch.zeros_like(logits)

        return torch.cat([z, probs, conf, delta], dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        prev_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns stop probability P(STOP | state) in [0, 1]."""
        state = self.extract_state_vector(h, logits, prev_logits)
        logits_out = self.mlp(state).squeeze(-1)
        return torch.sigmoid(logits_out)
