import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.clinc150 import get_dataloaders
from src.models.multiexit_distilbert import MultiExitDistilBERT
from src.routing.confidence_router import ConfidenceRouter
from src.utils.config import load_config
from src.utils.seed import set_seed


@torch.no_grad()
def extract_exit_predictions(
    model: MultiExitDistilBERT,
    dataloader: DataLoader,
    device: str,
    desc: str = "Extracting predictions",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference across all multi-exit heads and extracts softmax confidences and predictions."""
    model.eval()

    all_conf_2, all_conf_4 = [], []
    all_preds_2, all_preds_4, all_preds_6 = [], [], []
    all_targets = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()

        out = model(input_ids, attention_mask)
        logits = out["logits"]

        probs_2 = F.softmax(logits["exit_2"], dim=-1)
        conf_2, pred_2 = torch.max(probs_2, dim=-1)

        probs_4 = F.softmax(logits["exit_4"], dim=-1)
        conf_4, pred_4 = torch.max(probs_4, dim=-1)

        pred_6 = torch.argmax(logits["exit_6"], dim=-1)

        all_conf_2.extend(conf_2.cpu().numpy())
        all_conf_4.extend(conf_4.cpu().numpy())
        all_preds_2.extend(pred_2.cpu().numpy())
        all_preds_4.extend(pred_4.cpu().numpy())
        all_preds_6.extend(pred_6.cpu().numpy())
        all_targets.extend(labels)

    return (
        np.array(all_conf_2),
        np.array(all_conf_4),
        np.array(all_preds_2),
        np.array(all_preds_4),
        np.array(all_preds_6),
        np.array(all_targets),
    )


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(config["paths"]["checkpoints_dir"]) / "multiexit_distilbert_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load validation and test loaders
    _, val_loader, test_loader, _, _ = get_dataloaders(
        raw_path=config["data"]["raw_path"],
        processed_dir=config["data"]["processed_dir"],
        backbone_name=config["model"]["backbone_name"],
        max_seq_len=config["data"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        eval_batch_size=config["training"]["eval_batch_size"],
    )

    # Load frozen model from checkpoint
    model = MultiExitDistilBERT(
        backbone_name=config["model"]["backbone_name"],
        num_classes=config["data"]["num_classes"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded frozen multi-exit backbone from {checkpoint_path} (epoch {ckpt['epoch']}).")

    # Step 1: Extract predictions on VALIDATION set
    print("\n--- Step 1: Extracting predictions on Validation Set (3,000 samples) ---")
    val_c2, val_c4, val_p2, val_p4, val_p6, val_y = extract_exit_predictions(
        model, val_loader, device, desc="Val Extraction"
    )

    # Step 2: Calibrate thresholds on VALIDATION set only
    print("\n--- Step 2: Calibrating Confidence Thresholds on Validation Set ---")
    routing_cfg = config.get("routing", {}).get("confidence", {})
    min_tau = routing_cfg.get("min_tau", 0.50)
    max_tau = routing_cfg.get("max_tau", 0.99)
    grid_steps = routing_cfg.get("grid_steps", 40)
    target_f1_ratio = routing_cfg.get("target_f1_ratio", 0.99)

    # We evaluate 3 operating profiles: Balanced (default, 99%), Aggressive (97%), Conservative (99.8%)
    profiles = {
        "B3_conf_aggressive": {"target_f1_ratio": 0.97, "label": "Aggressive (Max Speedup)"},
        "B3_conf_balanced": {"target_f1_ratio": target_f1_ratio, "label": "Balanced (Standard, 99% F1)"},
        "B3_conf_conservative": {"target_f1_ratio": 0.998, "label": "Conservative (Near-lossless)"},
    }

    calibrated_profiles = {}
    for prof_key, prof_info in profiles.items():
        best_cfg, _ = ConfidenceRouter.calibrate(
            val_c2,
            val_c4,
            val_p2,
            val_p4,
            val_p6,
            val_y,
            grid_steps=grid_steps,
            min_tau=min_tau,
            max_tau=max_tau,
            target_f1_ratio=prof_info["target_f1_ratio"],
        )
        calibrated_profiles[prof_key] = {
            "label": prof_info["label"],
            "tau_2": best_cfg["tau_2"],
            "tau_4": best_cfg["tau_4"],
            "val_metrics": best_cfg,
        }
        print(
            f"  [{prof_info['label']}] Calibrated: tau_2 = {best_cfg['tau_2']:.4f}, "
            f"tau_4 = {best_cfg['tau_4']:.4f} -> Val F1: {best_cfg['macro_f1']:.4f}, "
            f"Avg Depth: {best_cfg['avg_exit_depth']:.2f} (L2: {best_cfg['exit_2_ratio']*100:.1f}%, "
            f"L4: {best_cfg['exit_4_ratio']*100:.1f}%, L6: {best_cfg['exit_6_ratio']*100:.1f}%)"
        )

    # Step 3: Extract predictions on TEST set
    print("\n--- Step 3: Extracting predictions on Test Set (4,500 samples) ---")
    test_c2, test_c4, test_p2, test_p4, test_p6, test_y = extract_exit_predictions(
        model, test_loader, device, desc="Test Extraction"
    )

    # Step 4: Evaluate calibrated routers on TEST set
    print("\n--- Step 4: Evaluating on Test Set with Frozen Thresholds ---")
    stage5_results = {}
    for prof_key, prof_info in calibrated_profiles.items():
        t2 = prof_info["tau_2"]
        t4 = prof_info["tau_4"]
        test_eval = ConfidenceRouter.evaluate_thresholds(
            test_c2, test_c4, test_p2, test_p4, test_p6, test_y, t2, t4
        )
        stage5_results[prof_key] = {
            "label": prof_info["label"],
            "tau_2": t2,
            "tau_4": t4,
            "val_avg_depth": prof_info["val_metrics"]["avg_exit_depth"],
            "val_macro_f1": prof_info["val_metrics"]["macro_f1"],
            "test_avg_depth": test_eval["avg_exit_depth"],
            "test_macro_f1": test_eval["macro_f1"],
            "test_accuracy": test_eval["accuracy"],
            "exit_distribution": {
                "exit_2_pct": round(test_eval["exit_2_ratio"] * 100, 2),
                "exit_4_pct": round(test_eval["exit_4_ratio"] * 100, 2),
                "exit_6_pct": round(test_eval["exit_6_ratio"] * 100, 2),
            },
        }

    # Step 5: Load Stage 4 Baselines and Print Unified Comparison Table
    results_dir = Path(config["paths"]["results_dir"])
    stage4_file = results_dir / "stage4_fixed_exit_baselines.json"
    stage4_baselines = {}
    if stage4_file.is_file():
        with open(stage4_file, "r") as f:
            stage4_baselines = json.load(f)

    print("\n" + "=" * 80)
    print(f"{'System / Profile':<26} {'Tau (L2/L4)':<14} {'Depth':>7} {'Macro-F1':>10} {'Accuracy':>10} {'Exit Split (2/4/6)':>20}")
    print("-" * 80)

    # Print Stage 4 baselines
    for b_name, b_val in stage4_baselines.items():
        tau_str = "Static"
        split_str = "100% / 0% / 0%" if "L2" in b_name else ("0% / 100% / 0%" if "L4" in b_name else "0% / 0% / 100%")
        print(
            f"{b_name:<26} {tau_str:<14} {b_val['avg_exit_depth']:>7.1f} "
            f"{b_val['macro_f1']:>10.4f} {b_val['accuracy']:>10.4f} {split_str:>20}"
        )

    print("-" * 80)

    # Print Stage 5 confidence routers
    for prof_key, res in stage5_results.items():
        tau_str = f"{res['tau_2']:.2f}/{res['tau_4']:.2f}"
        dist = res["exit_distribution"]
        split_str = f"{dist['exit_2_pct']:.0f}% / {dist['exit_4_pct']:.0f}% / {dist['exit_6_pct']:.0f}%"
        print(
            f"{prof_key:<26} {tau_str:<14} {res['test_avg_depth']:>7.2f} "
            f"{res['test_macro_f1']:>10.4f} {res['test_accuracy']:>10.4f} {split_str:>20}"
        )

    print("=" * 80)

    # Save to results/stage5_confidence_early_exit.json
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "stage5_confidence_early_exit.json"
    with open(out_file, "w") as f:
        json.dump(stage5_results, f, indent=2)
    print(f"\nResults saved successfully to {out_file}")


if __name__ == "__main__":
    main()
