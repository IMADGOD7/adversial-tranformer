import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.clinc150 import get_dataloaders
from src.models.multiexit_distilbert import MultiExitDistilBERT
from src.routing.confidence_router import ConfidenceRouter
from src.routing.controller import ExitControllerMLP
from src.utils.config import load_config
from src.utils.seed import set_seed


@torch.no_grad()
def extract_all_eval_data(
    model: MultiExitDistilBERT,
    dataloader: DataLoader,
    device: str,
    desc: str = "Extracting features",
) -> Dict[str, np.ndarray]:
    """Runs frozen model and extracts confidences, predictions, and representations."""
    model.eval()

    all_c2, all_c4, all_c6 = [], [], []
    all_p2, all_p4, all_p6 = [], [], []
    all_h2, all_h4 = [], []
    all_l2, all_l4 = [], []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids, attention_mask)
        logits = out["logits"]
        reps = out["representations"]

        probs_2 = F.softmax(logits["exit_2"], dim=-1)
        probs_4 = F.softmax(logits["exit_4"], dim=-1)
        probs_6 = F.softmax(logits["exit_6"], dim=-1)

        c2, p2 = torch.max(probs_2, dim=-1)
        c4, p4 = torch.max(probs_4, dim=-1)
        c6, p6 = torch.max(probs_6, dim=-1)

        all_c2.extend(c2.cpu().numpy())
        all_c4.extend(c4.cpu().numpy())
        all_c6.extend(c6.cpu().numpy())

        all_p2.extend(p2.cpu().numpy())
        all_p4.extend(p4.cpu().numpy())
        all_p6.extend(p6.cpu().numpy())

        all_h2.append(reps["exit_2"].cpu())
        all_h4.append(reps["exit_4"].cpu())
        all_l2.append(logits["exit_2"].cpu())
        all_l4.append(logits["exit_4"].cpu())

    return {
        "c2": np.array(all_c2),
        "c4": np.array(all_c4),
        "c6": np.array(all_c6),
        "p2": np.array(all_p2),
        "p4": np.array(all_p4),
        "p6": np.array(all_p6),
        "h2": torch.cat(all_h2, dim=0),
        "h4": torch.cat(all_h4, dim=0),
        "logits2": torch.cat(all_l2, dim=0),
        "logits4": torch.cat(all_l4, dim=0),
    }


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(config["paths"]["results_dir"])

    # Load previously calibrated configurations
    stage5_file = results_dir / "stage5_confidence_early_exit.json"
    stage6_file = results_dir / "stage6_controller_results.json"
    if not stage5_file.is_file() or not stage6_file.is_file():
        raise FileNotFoundError("Missing Stage 5 or Stage 6 results. Please run those stages first.")

    with open(stage5_file, "r") as f:
        stage5_data = json.load(f)
    with open(stage6_file, "r") as f:
        stage6_data = json.load(f)

    # Load dataloaders
    _, _, test_loader, oos_loaders, _ = get_dataloaders(
        raw_path=config["data"]["raw_path"],
        processed_dir=config["data"]["processed_dir"],
        backbone_name=config["model"]["backbone_name"],
        max_seq_len=config["data"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        eval_batch_size=config["training"]["eval_batch_size"],
    )

    if "oos_test" not in oos_loaders:
        raise ValueError("oos_test split not found in dataset!")
    oos_test_loader = oos_loaders["oos_test"]

    # Load frozen backbone model
    model = MultiExitDistilBERT(
        backbone_name=config["model"]["backbone_name"],
        num_classes=config["data"]["num_classes"],
    )
    checkpoint_path = Path(config["paths"]["checkpoints_dir"]) / "multiexit_distilbert_best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Load trained Controller MLPs
    controller_l2 = ExitControllerMLP(
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["data"]["num_classes"],
        mlp_hidden=config["model"]["controller_hidden_dim"],
    ).to(device)
    controller_l4 = ExitControllerMLP(
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["data"]["num_classes"],
        mlp_hidden=config["model"]["controller_hidden_dim"],
    ).to(device)

    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    controller_l2.load_state_dict(torch.load(ckpt_dir / "controller_l2.pt", map_location=device))
    controller_l4.load_state_dict(torch.load(ckpt_dir / "controller_l4.pt", map_location=device))
    controller_l2.eval()
    controller_l4.eval()

    print(f"Loaded models successfully. Evaluating In-Scope Test (4,500) vs. OOS Test (1,000)...")

    # Step 1: Feature Extraction
    print("\n--- Step 1: Extracting In-Scope Test Features ---")
    in_scope = extract_all_eval_data(model, test_loader, device, desc="In-Scope Test")

    print("\n--- Step 2: Extracting Out-of-Scope (OOS) Test Features ---")
    oos_data = extract_all_eval_data(model, oos_test_loader, device, desc="OOS Test")

    # Compute Controller Stop Probabilities
    with torch.no_grad():
        in_p_stop_2 = controller_l2(in_scope["h2"].to(device), in_scope["logits2"].to(device)).cpu().numpy()
        in_p_stop_4 = controller_l4(
            in_scope["h4"].to(device), in_scope["logits4"].to(device), in_scope["logits2"].to(device)
        ).cpu().numpy()

        oos_p_stop_2 = controller_l2(oos_data["h2"].to(device), oos_data["logits2"].to(device)).cpu().numpy()
        oos_p_stop_4 = controller_l4(
            oos_data["h4"].to(device), oos_data["logits4"].to(device), oos_data["logits2"].to(device)
        ).cpu().numpy()

    # Evaluate routing behaviors on OOS
    print("\n" + "=" * 90)
    print(f"{'System / Routing Policy':<28} {'In-Scope Depth':>16} {'OOS Depth':>12} {'OOS L2 Trap':>14} {'OOS Depth Shift':>16}")
    print("-" * 90)

    comparison_results = {}

    # 1. Confidence Router (Balanced)
    conf_cfg = stage5_data["B3_conf_balanced"]
    t2_c, t4_c = conf_cfg["tau_2"], conf_cfg["tau_4"]
    c_router = ConfidenceRouter(t2_c, t4_c)

    _, in_c_depths, in_c_stats = c_router.route(
        in_scope["c2"], in_scope["c4"], in_scope["p2"], in_scope["p4"], in_scope["p6"]
    )
    _, oos_c_depths, oos_c_stats = c_router.route(
        oos_data["c2"], oos_data["c4"], oos_data["p2"], oos_data["p4"], oos_data["p6"]
    )

    c_shift = oos_c_stats["avg_exit_depth"] - in_c_stats["avg_exit_depth"]
    print(
        f"{'B3 Confidence (Balanced)':<28} {in_c_stats['avg_exit_depth']:>16.2f} "
        f"{oos_c_stats['avg_exit_depth']:>12.2f} {oos_c_stats['exit_2_ratio']*100:>13.1f}% "
        f"{('+' if c_shift>=0 else '') + f'{c_shift:.2f} layers':>16}"
    )

    comparison_results["B3_conf_balanced"] = {
        "in_scope_avg_depth": in_c_stats["avg_exit_depth"],
        "oos_avg_depth": oos_c_stats["avg_exit_depth"],
        "oos_exit_2_pct": round(oos_c_stats["exit_2_ratio"] * 100, 2),
        "oos_exit_4_pct": round(oos_c_stats["exit_4_ratio"] * 100, 2),
        "oos_exit_6_pct": round(oos_c_stats["exit_6_ratio"] * 100, 2),
        "depth_increase": round(c_shift, 2),
    }

    # 2. Controller MLP (Balanced)
    ctrl_cfg = stage6_data["B4_controller_balanced"]
    t2_m, t4_m = ctrl_cfg["tau_2"], ctrl_cfg["tau_4"]

    # In-Scope routing
    in_exit_2 = in_p_stop_2 >= t2_m
    in_exit_4 = (~in_exit_2) & (in_p_stop_4 >= t4_m)
    in_depths_ctrl = np.where(in_exit_2, 2.0, np.where(in_exit_4, 4.0, 6.0))

    # OOS routing
    oos_exit_2 = oos_p_stop_2 >= t2_m
    oos_exit_4 = (~oos_exit_2) & (oos_p_stop_4 >= t4_m)
    oos_depths_ctrl = np.where(oos_exit_2, 2.0, np.where(oos_exit_4, 4.0, 6.0))

    in_ctrl_avg = float(np.mean(in_depths_ctrl))
    oos_ctrl_avg = float(np.mean(oos_depths_ctrl))
    ctrl_shift = oos_ctrl_avg - in_ctrl_avg
    oos_l2_ctrl_pct = float(np.mean(oos_exit_2)) * 100

    print(
        f"{'B4 Controller (Balanced)':<28} {in_ctrl_avg:>16.2f} "
        f"{oos_ctrl_avg:>12.2f} {oos_l2_ctrl_pct:>13.1f}% "
        f"{('+' if ctrl_shift>=0 else '') + f'{ctrl_shift:.2f} layers':>16}"
    )

    comparison_results["B4_controller_balanced"] = {
        "in_scope_avg_depth": in_ctrl_avg,
        "oos_avg_depth": oos_ctrl_avg,
        "oos_exit_2_pct": round(oos_l2_ctrl_pct, 2),
        "oos_exit_4_pct": round(float(np.mean(oos_exit_4)) * 100, 2),
        "oos_exit_6_pct": round(float(np.mean((~oos_exit_2) & (~oos_exit_4))) * 100, 2),
        "depth_increase": round(ctrl_shift, 2),
    }

    # Also calculate AUROC for OOS Detection
    # Ground truth: 0 for In-Scope, 1 for OOS
    y_true_oos = np.concatenate([np.zeros(len(in_scope["c2"])), np.ones(len(oos_data["c2"]))])

    # Score: Higher means more likely OOS (uncertainty score)
    # For Confidence: 1.0 - conf_2
    score_conf = np.concatenate([1.0 - in_scope["c2"], 1.0 - oos_data["c2"]])
    auroc_conf = float(roc_auc_score(y_true_oos, score_conf))

    # For Controller: 1.0 - p_stop_2
    score_ctrl = np.concatenate([1.0 - in_p_stop_2, 1.0 - oos_p_stop_2])
    auroc_ctrl = float(roc_auc_score(y_true_oos, score_ctrl))

    print("-" * 90)
    print(f"OOS Detection AUROC at Layer 2:")
    print(f"  Confidence-Based (1 - Conf):     {auroc_conf * 100:.2f}%")
    print(f"  Controller-Based (1 - P(Stop)):  {auroc_ctrl * 100:.2f}%")
    print("=" * 90)

    comparison_results["auroc_layer2"] = {
        "confidence_auroc": round(auroc_conf, 4),
        "controller_auroc": round(auroc_ctrl, 4),
    }

    out_file = results_dir / "stage6_oos_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nOOS Evaluation saved successfully to {out_file}")


if __name__ == "__main__":
    main()
