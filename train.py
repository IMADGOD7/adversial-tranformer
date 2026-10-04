import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# pyrefly: ignore [missing-import]
from src.data.clinc150 import get_dataloaders
# pyrefly: ignore [missing-import]
from src.models.multiexit_distilbert import MultiExitDistilBERT
# pyrefly: ignore [missing-import]
from src.training.trainer import Trainer
# pyrefly: ignore [missing-import]
from src.utils.config import load_config
# pyrefly: ignore [missing-import]
from src.utils.seed import set_seed


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() and config["experiment"]["device"] == "cuda" else "cpu"
    print(f"Executing on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        raw_path=config["data"]["raw_path"],
        processed_dir=config["data"]["processed_dir"],
        backbone_name=config["model"]["backbone_name"],
        max_seq_len=config["data"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        eval_batch_size=config["training"]["eval_batch_size"],
    )

    model = MultiExitDistilBERT(
        backbone_name=config["model"]["backbone_name"],
        num_classes=config["data"]["num_classes"],
        loss_weights=config["training"]["loss_weights"],
    )

    epochs = config["training"]["num_epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config["paths"]["checkpoints_dir"],
    )

    print(f"Starting Multi-Exit DistilBERT training for {epochs} epochs...")
    best_metrics = trainer.fit(num_epochs=epochs)

    print("\n--- Final Validation Metrics for Best Model ---")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
