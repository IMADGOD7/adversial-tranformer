import json
from pathlib import Path

import torch

from src.data.clinc150 import get_dataloaders
from src.models.multiexit_distilbert import MultiExitDistilBERT
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(config["paths"]["checkpoints_dir"]) / "multiexit_distilbert_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    _, _, test_loader, oos_loaders, _ = get_dataloaders(
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
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Dummy optimizer needed only for Trainer interface — not used during eval
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = Trainer(
        model=model,
        train_loader=test_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        device=device,
        save_dir=config["paths"]["checkpoints_dir"],
    )

    print(f"Evaluating on test set (frozen checkpoint from epoch {ckpt['epoch']})...\n")
    test_metrics = trainer.evaluate(test_loader)

    baselines = {
        "B0_full_L6": {
            "avg_exit_depth": 6.0,
            "macro_f1": test_metrics["exit_6_macro_f1"],
            "accuracy": test_metrics["exit_6_acc"],
        },
        "B1_fixed_L2": {
            "avg_exit_depth": 2.0,
            "macro_f1": test_metrics["exit_2_macro_f1"],
            "accuracy": test_metrics["exit_2_acc"],
        },
        "B2_fixed_L4": {
            "avg_exit_depth": 4.0,
            "macro_f1": test_metrics["exit_4_macro_f1"],
            "accuracy": test_metrics["exit_4_acc"],
        },
    }

    print("=" * 65)
    print(f"{'System':<20} {'Avg Depth':>10} {'Macro-F1':>12} {'Accuracy':>12}")
    print("-" * 65)
    for name, m in baselines.items():
        print(f"{name:<20} {m['avg_exit_depth']:>10.1f} {m['macro_f1']:>12.4f} {m['accuracy']:>12.4f}")
    print("=" * 65)

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "stage4_fixed_exit_baselines.json"
    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
