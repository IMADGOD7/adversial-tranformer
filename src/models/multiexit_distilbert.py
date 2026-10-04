from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel
from transformers.models.distilbert.modeling_distilbert import create_bidirectional_mask


class MultiExitDistilBERT(nn.Module):
    def __init__(
        self,
        backbone_name: str = "distilbert-base-uncased",
        num_classes: int = 150,
        loss_weights: Optional[Dict[str, float]] = None,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.config = DistilBertConfig.from_pretrained(backbone_name)
        self.distilbert = DistilBertModel.from_pretrained(backbone_name, config=self.config)
        self.loss_weights = loss_weights or {"exit_2": 0.3, "exit_4": 0.3, "exit_6": 1.0}
        
        self.dropout = nn.Dropout(dropout_prob)
        self.heads = nn.ModuleDict({
            "exit_2": nn.Linear(self.config.dim, num_classes),
            "exit_4": nn.Linear(self.config.dim, num_classes),
            "exit_6": nn.Linear(self.config.dim, num_classes),
        })
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states
        reps = {
            "exit_2": hidden[2][:, 0, :],
            "exit_4": hidden[4][:, 0, :],
            "exit_6": hidden[6][:, 0, :],
        }
        logits = {k: self.heads[k](self.dropout(v)) for k, v in reps.items()}

        out = {"logits": logits, "representations": reps}
        if labels is not None:
            valid = labels >= 0
            if valid.any():
                v_labels = labels[valid]
                exit_losses = {k: self.loss_fn(logits[k][valid], v_labels) for k in logits}
                out["loss"] = sum(self.loss_weights[k] * exit_losses[k] for k in exit_losses)
                out["exit_losses"] = exit_losses

        return out

    def _step_layer(self, layer_idx: int, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.distilbert.transformer.layer[layer_idx](hidden, mask)
        return out[0] if isinstance(out, tuple) else out

    def forward_early_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stop_at_exit: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if stop_at_exit not in (2, 4, 6):
            raise ValueError(f"Invalid stop_at_exit: {stop_at_exit}")

        hidden = self.distilbert.embeddings(input_ids)
        bi_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
        )

        # Stage 1: Layers 0-1 (Exit 2)
        for i in range(2):
            hidden = self._step_layer(i, hidden, bi_mask)
        if stop_at_exit == 2:
            h2 = hidden[:, 0, :]
            return self.heads["exit_2"](h2), h2

        # Stage 2: Layers 2-3 (Exit 4)
        for i in range(2, 4):
            hidden = self._step_layer(i, hidden, bi_mask)
        if stop_at_exit == 4:
            h4 = hidden[:, 0, :]
            return self.heads["exit_4"](h4), h4

        # Stage 3: Layers 4-5 (Exit 6)
        for i in range(4, 6):
            hidden = self._step_layer(i, hidden, bi_mask)
        h6 = hidden[:, 0, :]
        return self.heads["exit_6"](h6), h6
