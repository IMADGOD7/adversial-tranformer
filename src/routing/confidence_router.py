from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class ConfidenceRouter:
    """Early exit router based on Maximum Softmax Probability (MSP).

    Decides exit point sequentially:
      - Exit at Layer 2 if max_prob_2 >= tau_2
      - Else exit at Layer 4 if max_prob_4 >= tau_4
      - Else exit at Layer 6
    """

    def __init__(self, tau_2: float, tau_4: float):
        self.tau_2 = float(tau_2)
        self.tau_4 = float(tau_4)

    def route(
        self,
        conf_2: np.ndarray,
        conf_4: np.ndarray,
        preds_2: np.ndarray,
        preds_4: np.ndarray,
        preds_6: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Vectorized routing simulation.

        Args:
            conf_2: (N,) Maximum softmax probability at exit 2.
            conf_4: (N,) Maximum softmax probability at exit 4.
            preds_2: (N,) Predicted class indices at exit 2.
            preds_4: (N,) Predicted class indices at exit 4.
            preds_6: (N,) Predicted class indices at exit 6.

        Returns:
            final_preds: (N,) Combined predictions.
            depths: (N,) Exit depth for each sample (2.0, 4.0, or 6.0).
            stats: Summary dictionary with fraction exiting at each layer and average depth.
        """
        n_samples = len(conf_2)
        exit_2 = conf_2 >= self.tau_2
        exit_4 = (~exit_2) & (conf_4 >= self.tau_4)
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

    @staticmethod
    def evaluate_thresholds(
        conf_2: np.ndarray,
        conf_4: np.ndarray,
        preds_2: np.ndarray,
        preds_4: np.ndarray,
        preds_6: np.ndarray,
        targets: np.ndarray,
        tau_2: float,
        tau_4: float,
    ) -> Dict[str, float]:
        """Compute metrics for a specific (tau_2, tau_4) configuration."""
        router = ConfidenceRouter(tau_2, tau_4)
        final_preds, depths, stats = router.route(conf_2, conf_4, preds_2, preds_4, preds_6)
        acc = float(accuracy_score(targets, final_preds))
        macro_f1 = float(f1_score(targets, final_preds, average="macro", zero_division=0))
        return {
            "tau_2": tau_2,
            "tau_4": tau_4,
            "macro_f1": macro_f1,
            "accuracy": acc,
            "avg_exit_depth": stats["avg_exit_depth"],
            "exit_2_ratio": stats["exit_2_ratio"],
            "exit_4_ratio": stats["exit_4_ratio"],
            "exit_6_ratio": stats["exit_6_ratio"],
        }

    @classmethod
    def calibrate(
        cls,
        conf_2: np.ndarray,
        conf_4: np.ndarray,
        preds_2: np.ndarray,
        preds_4: np.ndarray,
        preds_6: np.ndarray,
        targets: np.ndarray,
        grid_steps: int = 40,
        min_tau: float = 0.50,
        max_tau: float = 0.99,
        target_f1_ratio: float = 0.99,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Calibrate thresholds on validation data by grid search.

        Finds the (tau_2, tau_4) pair that minimizes average exit depth while
        retaining at least `target_f1_ratio` of the full-depth (L6) Macro-F1.

        Returns:
            best_config: The optimal threshold configuration.
            all_results: All evaluated points on the grid.
        """
        l6_f1 = float(f1_score(targets, preds_6, average="macro", zero_division=0))
        min_acceptable_f1 = l6_f1 * target_f1_ratio

        tau_grid = np.linspace(min_tau, max_tau, grid_steps)
        all_results = []
        valid_candidates = []

        for t2 in tau_grid:
            for t4 in tau_grid:
                metrics = cls.evaluate_thresholds(
                    conf_2, conf_4, preds_2, preds_4, preds_6, targets, float(t2), float(t4)
                )
                all_results.append(metrics)
                if metrics["macro_f1"] >= min_acceptable_f1:
                    valid_candidates.append(metrics)

        if valid_candidates:
            # Pick candidate that minimizes avg_exit_depth (highest speedup).
            # Tie-break by highest macro_f1.
            best_config = min(valid_candidates, key=lambda m: (m["avg_exit_depth"], -m["macro_f1"]))
        else:
            # Fallback to the point with maximum macro_f1
            best_config = max(all_results, key=lambda m: m["macro_f1"])

        return best_config, all_results
