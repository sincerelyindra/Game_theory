from __future__ import annotations

"""
Part C: Adaptive confidence-interval calibration for robust quantile pricing.

This script is designed as the *novel empirical extension* for the project.
It compares:
- Fixed 90% CI
- Fixed 95% CI
- Fixed 99% CI
- Adaptive-Feasible: choose the narrowest CI level that yields a feasible ambiguity set
- Adaptive-Validated: choose the feasible CI level with the best validation revenue

It reuses an existing robust-pricing solver file (for example
`robust_pricing_solver_fixed.py`) and produces summary CSVs and plots.
"""

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class MethodResult:
    method: str
    selected_conf: float
    feasible: bool
    price: float
    robust_value: float
    mean_interval_width: float
    expected_ratio: float
    expected_revenue: float
    oracle_revenue: float


FIXED_LEVELS: Dict[str, float] = {
    "Fixed90": 0.90,
    "Fixed95": 0.95,
    "Fixed99": 0.99,
}

ADAPTIVE_METHODS = ["AdaptiveFeasible", "AdaptiveValidated"]
ALL_METHODS = list(FIXED_LEVELS.keys()) + ADAPTIVE_METHODS


# -----------------------------------------------------------------------------
# Dynamic import of the user's solver module
# -----------------------------------------------------------------------------


def load_solver_module(path: str):
    spec = importlib.util.spec_from_file_location("rp_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import solver module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rp_module"] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# CI construction and policy selection
# -----------------------------------------------------------------------------


def build_intervals_for_level(rp, counts: Sequence[int], n: int, conf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qhat, ql, qh = [], [], []
    for k in counts:
        phat = k / n
        lo, hi = rp.wilson_interval(int(k), int(n), conf)
        qhat.append(phat)
        ql.append(lo)
        qh.append(hi)
    return np.asarray(qhat, dtype=float), np.asarray(ql, dtype=float), np.asarray(qh, dtype=float)


def average_width(ql: np.ndarray, qh: np.ndarray) -> float:
    return float(np.mean(np.maximum(0.0, qh - ql)))


def choose_fixed_method(
    rp,
    model,
    obs_prices: np.ndarray,
    counts: Sequence[int],
    n_train: int,
    alpha: float,
    conf: float,
    r_grid_size: int,
) -> MethodResult:
    _, ql, qh = build_intervals_for_level(rp, counts, n_train, conf)
    amb = rp.AmbiguitySet(obs_prices, ql, qh, alpha, model.support_max)

    if amb.is_feasible:
        robust_value, price = amb.robust_value(obs_prices, r_grid_size=r_grid_size)
    else:
        robust_value, price = 0.0, float(obs_prices[0])

    _, oracle_revenue = model.oracle_price(4001)
    expected_revenue = float(price * model.demand(price))
    expected_ratio = expected_revenue / max(oracle_revenue, 1e-12)

    return MethodResult(
        method=f"Fixed{int(round(conf * 100))}",
        selected_conf=float(conf),
        feasible=bool(amb.is_feasible),
        price=float(price),
        robust_value=float(robust_value),
        mean_interval_width=average_width(ql, qh),
        expected_ratio=float(expected_ratio),
        expected_revenue=float(expected_revenue),
        oracle_revenue=float(oracle_revenue),
    )


def choose_adaptive_feasible(
    rp,
    model,
    obs_prices: np.ndarray,
    counts: Sequence[int],
    n_train: int,
    alpha: float,
    candidate_confs: Sequence[float],
    r_grid_size: int,
) -> MethodResult:
    chosen = None
    for conf in sorted(candidate_confs):
        _, ql, qh = build_intervals_for_level(rp, counts, n_train, conf)
        amb = rp.AmbiguitySet(obs_prices, ql, qh, alpha, model.support_max)
        if amb.is_feasible:
            chosen = (conf, amb, ql, qh)
            break

    if chosen is None:
        conf = max(candidate_confs)
        _, ql, qh = build_intervals_for_level(rp, counts, n_train, conf)
        amb = rp.AmbiguitySet(obs_prices, ql, qh, alpha, model.support_max)
    else:
        conf, amb, ql, qh = chosen

    if amb.is_feasible:
        robust_value, price = amb.robust_value(obs_prices, r_grid_size=r_grid_size)
    else:
        robust_value, price = 0.0, float(obs_prices[0])

    _, oracle_revenue = model.oracle_price(4001)
    expected_revenue = float(price * model.demand(price))
    expected_ratio = expected_revenue / max(oracle_revenue, 1e-12)

    return MethodResult(
        method="AdaptiveFeasible",
        selected_conf=float(conf),
        feasible=bool(amb.is_feasible),
        price=float(price),
        robust_value=float(robust_value),
        mean_interval_width=average_width(ql, qh),
        expected_ratio=float(expected_ratio),
        expected_revenue=float(expected_revenue),
        oracle_revenue=float(oracle_revenue),
    )


def choose_adaptive_validated(
    rp,
    model,
    obs_prices: np.ndarray,
    counts: Sequence[int],
    n_train: int,
    alpha: float,
    candidate_confs: Sequence[float],
    n_val: int,
    rng: np.random.Generator,
    r_grid_size: int,
) -> MethodResult:
    candidates = []
    for conf in sorted(candidate_confs):
        _, ql, qh = build_intervals_for_level(rp, counts, n_train, conf)
        amb = rp.AmbiguitySet(obs_prices, ql, qh, alpha, model.support_max)
        if not amb.is_feasible:
            continue
        robust_value, price = amb.robust_value(obs_prices, r_grid_size=r_grid_size)
        val_draws = model.sample_bernoulli(float(price), n_val, rng)
        val_revenue = float(price * np.mean(val_draws))
        candidates.append((val_revenue, -average_width(ql, qh), conf, amb, ql, qh, robust_value, price))

    if not candidates:
        # Fall back to the widest interval.
        conf = max(candidate_confs)
        _, ql, qh = build_intervals_for_level(rp, counts, n_train, conf)
        amb = rp.AmbiguitySet(obs_prices, ql, qh, alpha, model.support_max)
        if amb.is_feasible:
            robust_value, price = amb.robust_value(obs_prices, r_grid_size=r_grid_size)
        else:
            robust_value, price = 0.0, float(obs_prices[0])
    else:
        candidates.sort(reverse=True)
        _, _, conf, amb, ql, qh, robust_value, price = candidates[0]

    _, oracle_revenue = model.oracle_price(4001)
    expected_revenue = float(price * model.demand(price))
    expected_ratio = expected_revenue / max(oracle_revenue, 1e-12)

    return MethodResult(
        method="AdaptiveValidated",
        selected_conf=float(conf),
        feasible=bool(amb.is_feasible),
        price=float(price),
        robust_value=float(robust_value),
        mean_interval_width=average_width(ql, qh),
        expected_ratio=float(expected_ratio),
        expected_revenue=float(expected_revenue),
        oracle_revenue=float(oracle_revenue),
    )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def run_part_c_simulation(
    rp,
    out_dir: str,
    model_names: Sequence[str],
    alphas: Sequence[float],
    Ks: Sequence[int],
    sample_sizes: Sequence[int],
    n_trials: int,
    candidate_confs: Sequence[float],
    n_val: int,
    r_grid_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    trial_rows: List[dict] = []

    for model_name in model_names:
        model = rp.MODELS[model_name]
        for alpha in alphas:
            for K in Ks:
                # Use a common interior grid of observed prices.
                obs_prices = np.linspace(0.1 * model.support_max, 0.9 * model.support_max, K)
                for n_train in sample_sizes:
                    for trial in range(n_trials):
                        counts = [int(model.sample_bernoulli(float(p), n_train, rng).sum()) for p in obs_prices]

                        results: List[MethodResult] = []
                        for _, conf in FIXED_LEVELS.items():
                            results.append(
                                choose_fixed_method(
                                    rp, model, obs_prices, counts, n_train, alpha, conf, r_grid_size
                                )
                            )

                        results.append(
                            choose_adaptive_feasible(
                                rp, model, obs_prices, counts, n_train, alpha, candidate_confs, r_grid_size
                            )
                        )
                        results.append(
                            choose_adaptive_validated(
                                rp,
                                model,
                                obs_prices,
                                counts,
                                n_train,
                                alpha,
                                candidate_confs,
                                n_val,
                                rng,
                                r_grid_size,
                            )
                        )

                        for res in results:
                            trial_rows.append(
                                {
                                    "model": model_name,
                                    "alpha": alpha,
                                    "K": K,
                                    "n_train": n_train,
                                    "trial": trial,
                                    "method": res.method,
                                    "selected_conf": res.selected_conf,
                                    "feasible": int(res.feasible),
                                    "price": res.price,
                                    "robust_value": res.robust_value,
                                    "mean_interval_width": res.mean_interval_width,
                                    "expected_ratio": res.expected_ratio,
                                    "expected_revenue": res.expected_revenue,
                                    "oracle_revenue": res.oracle_revenue,
                                }
                            )

    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_csv(os.path.join(out_dir, "adaptive_ci_trials.csv"), index=False)

    summary_df = (
        trials_df.groupby(["model", "alpha", "K", "n_train", "method"], as_index=False)
        .agg(
            feasibility_rate=("feasible", "mean"),
            mean_ratio=("expected_ratio", "mean"),
            median_ratio=("expected_ratio", "median"),
            std_ratio=("expected_ratio", "std"),
            worst_decile=("expected_ratio", lambda x: float(np.quantile(x, 0.1))),
            mean_width=("mean_interval_width", "mean"),
            mean_selected_conf=("selected_conf", "mean"),
            mean_robust_value=("robust_value", "mean"),
        )
    )
    summary_df.to_csv(os.path.join(out_dir, "adaptive_ci_summary.csv"), index=False)

    make_plots(out_dir, summary_df, model_names, alphas, Ks, sample_sizes)
    return trials_df, summary_df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def make_plots(
    out_dir: str,
    summary_df: pd.DataFrame,
    model_names: Sequence[str],
    alphas: Sequence[float],
    Ks: Sequence[int],
    sample_sizes: Sequence[int],
) -> None:
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    # Fix the first alpha and K for headline figures if multiple are passed.
    alpha0 = alphas[0]
    K0 = Ks[0]

    for metric, ylabel, filename in [
        ("mean_ratio", "Mean Revenue Ratio", "mean_ratio.png"),
        ("feasibility_rate", "Feasibility Rate", "feasibility_rate.png"),
        ("mean_width", "Average Interval Width", "mean_width.png"),
        ("mean_selected_conf", "Average Selected CI Level", "selected_conf.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, model_name in zip(axes.ravel(), model_names):
            sub = summary_df[
                (summary_df["model"] == model_name)
                & (summary_df["alpha"] == alpha0)
                & (summary_df["K"] == K0)
            ]
            for method in ALL_METHODS:
                vals = sub[sub["method"] == method].sort_values("n_train")
                if len(vals) == 0:
                    continue
                ax.plot(vals["n_train"], vals[metric], marker="o", label=method)
            ax.set_title(model_name.replace("_", " ").title())
            ax.set_xlabel("Sample Size per Price")
            ax.set_ylabel(ylabel)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, filename), dpi=180)
        plt.close(fig)

    # Appendix-style table for the first alpha and K.
    appendix_df = summary_df[(summary_df["alpha"] == alpha0) & (summary_df["K"] == K0)].copy()
    appendix_df.to_csv(os.path.join(out_dir, "appendix_table_part_c.csv"), index=False)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        default="/mnt/data/robust_pricing_solver_fixed.py",
        help="Path to the solver module to import.",
    )
    parser.add_argument("--out", default="part_c_results", help="Output directory.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["linear", "logit", "pareto", "piecewise_exponential"],
        help="Demand models to evaluate.",
    )
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 1.0])
    parser.add_argument("--Ks", nargs="*", type=int, default=[3])
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--candidate-confs", nargs="*", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--r-grid-size", type=int, default=121)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rp = load_solver_module(args.solver)
    ensure_dir(args.out)
    run_part_c_simulation(
        rp=rp,
        out_dir=args.out,
        model_names=args.models,
        alphas=args.alphas,
        Ks=args.Ks,
        sample_sizes=args.sample_sizes,
        n_trials=args.n_trials,
        candidate_confs=args.candidate_confs,
        n_val=args.n_val,
        r_grid_size=args.r_grid_size,
        seed=args.seed,
    )
    print(f"Saved Part C outputs to: {args.out}")


if __name__ == "__main__":
    main()
