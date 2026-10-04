from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_macro_f1 = -1.0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(input_ids, attention_mask, labels=labels)
            loss = out["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = {"exit_2": [], "exit_4": [], "exit_6": []}
        all_targets = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].cpu().numpy()

            out = self.model(input_ids, attention_mask)
            for exit_name in all_preds:
                preds = out["logits"][exit_name].argmax(dim=-1).cpu().numpy()
                all_preds[exit_name].extend(preds)
            all_targets.extend(labels)

        targets = np.array(all_targets)
        metrics = {}
        for exit_name, preds in all_preds.items():
            p = np.array(preds)
            metrics[f"{exit_name}_acc"] = float(accuracy_score(targets, p))
            metrics[f"{exit_name}_macro_f1"] = float(f1_score(targets, p, average="macro", zero_division=0))

        return metrics

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def fit(self, num_epochs: int) -> Dict[str, float]:
        best_metrics = {}
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            l6_f1 = val_metrics["exit_6_macro_f1"]
            print(
                f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
                f"L2: F1={val_metrics['exit_2_macro_f1']:.4f}, Acc={val_metrics['exit_2_acc']:.4f} | "
                f"L4: F1={val_metrics['exit_4_macro_f1']:.4f}, Acc={val_metrics['exit_4_acc']:.4f} | "
                f"L6: F1={l6_f1:.4f}, Acc={val_metrics['exit_6_acc']:.4f}"
            )

            if l6_f1 > self.best_macro_f1:
                self.best_macro_f1 = l6_f1
                best_metrics = val_metrics
                save_path = self.save_dir / "multiexit_distilbert_best.pt"
                self.save_checkpoint(save_path, epoch, val_metrics)
                print(f"  -> Saved new best checkpoint to {save_path} (L6 F1: {l6_f1:.4f})")

        return best_metrics
