import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.clinc150 import get_dataloaders
from src.models.multiexit_distilbert import MultiExitDistilBERT
from src.routing.controller import ExitControllerMLP
from src.utils.config import load_config
from src.utils.seed import set_seed


@torch.no_grad()
def extract_representations_and_logits(
    model: MultiExitDistilBERT,
    dataloader: DataLoader,
    device: str,
    desc: str = "Extracting features",
) -> Dict[str, torch.Tensor]:
    """Runs frozen backbone inference once to cache representations, logits, and labels."""
    model.eval()

    all_h2, all_h4, all_h6 = [], [], []
    all_logits2, all_logits4, all_logits6 = [], [], []
    all_labels = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        out = model(input_ids, attention_mask)
        reps = out["representations"]
        logits = out["logits"]

        all_h2.append(reps["exit_2"].cpu())
        all_h4.append(reps["exit_4"].cpu())
        all_h6.append(reps["exit_6"].cpu())

        all_logits2.append(logits["exit_2"].cpu())
        all_logits4.append(logits["exit_4"].cpu())
        all_logits6.append(logits["exit_6"].cpu())

        all_labels.append(labels.cpu())

    return {
        "h2": torch.cat(all_h2, dim=0),
        "h4": torch.cat(all_h4, dim=0),
        "h6": torch.cat(all_h6, dim=0),
        "logits2": torch.cat(all_logits2, dim=0),
        "logits4": torch.cat(all_logits4, dim=0),
        "logits6": torch.cat(all_logits6, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


def train_single_controller(
    controller: ExitControllerMLP,
    h: torch.Tensor,
    logits: torch.Tensor,
    prev_logits: torch.Tensor,
    labels: torch.Tensor,
    val_h: torch.Tensor,
    val_logits: torch.Tensor,
    val_prev_logits: torch.Tensor,
    val_labels: torch.Tensor,
    device: str,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    exit_name: str = "L2",
) -> ExitControllerMLP:
    """Trains a controller MLP to predict whether stopping at the current exit is correct."""
    controller = controller.to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    # Supervisory target: 1 if prediction matches true label, else 0
    preds = torch.argmax(logits, dim=-1)
    targets = (preds == labels).float().to(device)

    val_preds = torch.argmax(val_logits, dim=-1)
    val_targets = (val_preds == val_labels).float().to(device)

    # Tensor dataset
    train_data = TensorDataset(h, logits, prev_logits if prev_logits is not None else torch.zeros_like(logits), targets)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print(f"Training Controller for {exit_name} (Target positive rate: {targets.mean().item()*100:.1f}%)...")
    for epoch in range(1, epochs + 1):
        controller.train()
        total_loss = 0.0
        for batch_h, batch_l, batch_prev_l, batch_y in loader:
            batch_h = batch_h.to(device)
            batch_l = batch_l.to(device)
            batch_prev_l = batch_prev_l.to(device) if prev_logits is not None else None
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            p_stop = controller(batch_h, batch_l, batch_prev_l)
            loss = loss_fn(p_stop, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loss check
        controller.eval()
        with torch.no_grad():
            v_p_stop = controller(
                val_h.to(device),
                val_logits.to(device),
                val_prev_logits.to(device) if val_prev_logits is not None else None,
            )
            val_loss = loss_fn(v_p_stop, val_targets).item()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {total_loss / len(loader):.4f} | Val Loss: {val_loss:.4f}")

    return controller


def simulate_controller_routing(
    p_stop_2: np.ndarray,
    p_stop_4: np.ndarray,
    preds_2: np.ndarray,
    preds_4: np.ndarray,
    preds_6: np.ndarray,
    tau_2: float,
    tau_4: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Vectorized simulation of controller routing decisions."""
    exit_2 = p_stop_2 >= tau_2
    exit_4 = (~exit_2) & (p_stop_4 >= tau_4)
    exit_6 = (~exit_2) & (~exit_4)

    final_preds = np.where(exit_2, preds_2, np.where(exit_4, preds_4, preds_6))
    depths = np.where(exit_2, 2.0, np.where(exit_4, 4.0, 6.0))

    stats = {
        "exit_2_ratio": float(np.mean(exit_2)),
        "exit_4_ratio": float(np.mean(exit_4)),
        "exit_6_ratio": float(np.mean(exit_6)),
        "avg_exit_depth": float(np.mean(depths)),
    }
    return final_preds, depths, stats


def calibrate_controller_thresholds(
    p_stop_2: np.ndarray,
    p_stop_4: np.ndarray,
    preds_2: np.ndarray,
    preds_4: np.ndarray,
    preds_6: np.ndarray,
    targets: np.ndarray,
    target_f1_ratio: float = 0.99,
    grid_steps: int = 40,
) -> Dict[str, float]:
    """Sweeps thresholds on validation set to minimize average exit depth subject to F1 constraint."""
    l6_f1 = float(f1_score(targets, preds_6, average="macro", zero_division=0))
    min_acceptable_f1 = l6_f1 * target_f1_ratio

    tau_grid = np.linspace(0.40, 0.99, grid_steps)
    valid_candidates = []
    all_results = []

    for t2 in tau_grid:
        for t4 in tau_grid:
            final_p, _, stats = simulate_controller_routing(
                p_stop_2, p_stop_4, preds_2, preds_4, preds_6, float(t2), float(t4)
            )
            f1 = float(f1_score(targets, final_p, average="macro", zero_division=0))
            acc = float(accuracy_score(targets, final_p))
            res = {
                "tau_2": float(t2),
                "tau_4": float(t4),
                "macro_f1": f1,
                "accuracy": acc,
                "avg_exit_depth": stats["avg_exit_depth"],
                "exit_2_ratio": stats["exit_2_ratio"],
                "exit_4_ratio": stats["exit_4_ratio"],
                "exit_6_ratio": stats["exit_6_ratio"],
            }
            all_results.append(res)
            if f1 >= min_acceptable_f1:
                valid_candidates.append(res)

    if valid_candidates:
        best = min(valid_candidates, key=lambda m: (m["avg_exit_depth"], -m["macro_f1"]))
    else:
        best = max(all_results, key=lambda m: m["macro_f1"])

    return best


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(config["paths"]["checkpoints_dir"]) / "multiexit_distilbert_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load Train, Val, and Test dataloaders
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        raw_path=config["data"]["raw_path"],
        processed_dir=config["data"]["processed_dir"],
        backbone_name=config["model"]["backbone_name"],
        max_seq_len=config["data"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        eval_batch_size=config["training"]["eval_batch_size"],
    )

    # Load frozen multi-exit backbone
    model = MultiExitDistilBERT(
        backbone_name=config["model"]["backbone_name"],
        num_classes=config["data"]["num_classes"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded frozen multi-exit backbone from {checkpoint_path} (epoch {ckpt['epoch']}).")

    # Step 1: Feature Extraction across splits
    print("\n--- Step 1: Extracting Representations and Logits across Splits ---")
    train_cache = extract_representations_and_logits(model, train_loader, device, desc="Train Features")
    val_cache = extract_representations_and_logits(model, val_loader, device, desc="Val Features")
    test_cache = extract_representations_and_logits(model, test_loader, device, desc="Test Features")

    # Step 2: Initialize and Train Controllers for L2 and L4
    print("\n--- Step 2: Training Lightweight Controller MLPs ---")
    controller_l2 = ExitControllerMLP(
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["data"]["num_classes"],
        mlp_hidden=config["model"]["controller_hidden_dim"],
    )
    controller_l4 = ExitControllerMLP(
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["data"]["num_classes"],
        mlp_hidden=config["model"]["controller_hidden_dim"],
    )

    # Train L2 Controller
    controller_l2 = train_single_controller(
        controller=controller_l2,
        h=train_cache["h2"],
        logits=train_cache["logits2"],
        prev_logits=None,
        labels=train_cache["labels"],
        val_h=val_cache["h2"],
        val_logits=val_cache["logits2"],
        val_prev_logits=None,
        val_labels=val_cache["labels"],
        device=device,
        epochs=15,
        exit_name="Layer 2",
    )

    # Train L4 Controller (takes L4 hidden state, L4 logits, and delta from L2 logits)
    controller_l4 = train_single_controller(
        controller=controller_l4,
        h=train_cache["h4"],
        logits=train_cache["logits4"],
        prev_logits=train_cache["logits2"],
        labels=train_cache["labels"],
        val_h=val_cache["h4"],
        val_logits=val_cache["logits4"],
        val_prev_logits=val_cache["logits2"],
        val_labels=val_cache["labels"],
        device=device,
        epochs=15,
        exit_name="Layer 4",
    )

    # Save trained controllers
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    torch.save(controller_l2.state_dict(), checkpoints_dir / "controller_l2.pt")
    torch.save(controller_l4.state_dict(), checkpoints_dir / "controller_l4.pt")
    print("Controllers saved to checkpoints/controller_l2.pt and controller_l4.pt")

    # Step 3: Compute Stop Probabilities on Validation and Test
    print("\n--- Step 3: Evaluating Stop Probabilities ---")
    controller_l2.eval()
    controller_l4.eval()
    with torch.no_grad():
        val_p_stop_2 = controller_l2(val_cache["h2"].to(device), val_cache["logits2"].to(device)).cpu().numpy()
        val_p_stop_4 = controller_l4(
            val_cache["h4"].to(device), val_cache["logits4"].to(device), val_cache["logits2"].to(device)
        ).cpu().numpy()

        test_p_stop_2 = controller_l2(test_cache["h2"].to(device), test_cache["logits2"].to(device)).cpu().numpy()
        test_p_stop_4 = controller_l4(
            test_cache["h4"].to(device), test_cache["logits4"].to(device), test_cache["logits2"].to(device)
        ).cpu().numpy()

    # Class predictions
    val_preds_2 = torch.argmax(val_cache["logits2"], dim=-1).numpy()
    val_preds_4 = torch.argmax(val_cache["logits4"], dim=-1).numpy()
    val_preds_6 = torch.argmax(val_cache["logits6"], dim=-1).numpy()
    val_y = val_cache["labels"].numpy()

    test_preds_2 = torch.argmax(test_cache["logits2"], dim=-1).numpy()
    test_preds_4 = torch.argmax(test_cache["logits4"], dim=-1).numpy()
    test_preds_6 = torch.argmax(test_cache["logits6"], dim=-1).numpy()
    test_y = test_cache["labels"].numpy()

    # Step 4: Calibrate Thresholds on Validation Split
    print("\n--- Step 4: Calibrating Controller Thresholds on Validation Split ---")
    profiles = {
        "B4_controller_aggressive": {"target_f1_ratio": 0.97, "label": "Aggressive (Max Speedup)"},
        "B4_controller_balanced": {"target_f1_ratio": 0.99, "label": "Balanced (Standard, 99% F1)"},
        "B4_controller_conservative": {"target_f1_ratio": 0.998, "label": "Conservative (Near-lossless)"},
    }

    calibrated_results = {}
    for prof_key, prof_info in profiles.items():
        best_cfg = calibrate_controller_thresholds(
            val_p_stop_2,
            val_p_stop_4,
            val_preds_2,
            val_preds_4,
            val_preds_6,
            val_y,
            target_f1_ratio=prof_info["target_f1_ratio"],
        )

        # Evaluate on Test Set
        test_preds, test_depths, stats = simulate_controller_routing(
            test_p_stop_2,
            test_p_stop_4,
            test_preds_2,
            test_preds_4,
            test_preds_6,
            best_cfg["tau_2"],
            best_cfg["tau_4"],
        )
        test_f1 = float(f1_score(test_y, test_preds, average="macro", zero_division=0))
        test_acc = float(accuracy_score(test_y, test_preds))

        calibrated_results[prof_key] = {
            "label": prof_info["label"],
            "tau_2": best_cfg["tau_2"],
            "tau_4": best_cfg["tau_4"],
            "val_avg_depth": best_cfg["avg_exit_depth"],
            "val_macro_f1": best_cfg["macro_f1"],
            "test_avg_depth": stats["avg_exit_depth"],
            "test_macro_f1": test_f1,
            "test_accuracy": test_acc,
            "exit_distribution": {
                "exit_2_pct": round(stats["exit_2_ratio"] * 100, 2),
                "exit_4_pct": round(stats["exit_4_ratio"] * 100, 2),
                "exit_6_pct": round(stats["exit_6_ratio"] * 100, 2),
            },
        }

        print(
            f"  [{prof_info['label']}] Calibrated tau: ({best_cfg['tau_2']:.2f}, {best_cfg['tau_4']:.2f}) -> "
            f"Test F1: {test_f1:.4f}, Depth: {stats['avg_exit_depth']:.2f} "
            f"(L2: {stats['exit_2_ratio']*100:.1f}%, L4: {stats['exit_4_ratio']*100:.1f}%, L6: {stats['exit_6_ratio']*100:.1f}%)"
        )

    # Step 5: Print Unified Comparison Table (Stage 4 vs Stage 5 vs Stage 6)
    results_dir = Path(config["paths"]["results_dir"])
    stage4_data = {}
    stage5_data = {}
    if (results_dir / "stage4_fixed_exit_baselines.json").is_file():
        with open(results_dir / "stage4_fixed_exit_baselines.json", "r") as f:
            stage4_data = json.load(f)
    if (results_dir / "stage5_confidence_early_exit.json").is_file():
        with open(results_dir / "stage5_confidence_early_exit.json", "r") as f:
            stage5_data = json.load(f)

    print("\n" + "=" * 84)
    print(f"{'System / Routing Policy':<28} {'Mechanism':<14} {'Depth':>7} {'Macro-F1':>10} {'Accuracy':>10} {'Exit Split (2/4/6)':>14}")
    print("-" * 84)

    # Stage 4 baselines
    for b_name, b_val in stage4_data.items():
        print(f"{b_name:<28} {'Static':<14} {b_val['avg_exit_depth']:>7.1f} {b_val['macro_f1']:>10.4f} {b_val['accuracy']:>10.4f} {'-':>14}")

    print("-" * 84)

    # Stage 5 confidence baselines
    for c_name, c_val in stage5_data.items():
        dist = c_val["exit_distribution"]
        split_s = f"{dist['exit_2_pct']:.0f}%/{dist['exit_4_pct']:.0f}%/{dist['exit_6_pct']:.0f}%"
        print(f"{c_name:<28} {'Confidence':<14} {c_val['test_avg_depth']:>7.2f} {c_val['test_macro_f1']:>10.4f} {c_val['test_accuracy']:>10.4f} {split_s:>14}")

    print("-" * 84)

    # Stage 6 controller results
    for ctrl_name, ctrl_val in calibrated_results.items():
        dist = ctrl_val["exit_distribution"]
        split_s = f"{dist['exit_2_pct']:.0f}%/{dist['exit_4_pct']:.0f}%/{dist['exit_6_pct']:.0f}%"
        print(f"{ctrl_name:<28} {'Controller MLP':<14} {ctrl_val['test_avg_depth']:>7.2f} {ctrl_val['test_macro_f1']:>10.4f} {ctrl_val['test_accuracy']:>10.4f} {split_s:>14}")

    print("=" * 84)

    # Save to file
    out_file = results_dir / "stage6_controller_results.json"
    with open(out_file, "w") as f:
        json.dump(calibrated_results, f, indent=2)
    print(f"\nResults saved successfully to {out_file}")


if __name__ == "__main__":
    main()
