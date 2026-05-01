from __future__ import annotations

import importlib.util
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm


def load_solver_module(path: str):
    spec = importlib.util.spec_from_file_location("robust_solver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import solver module from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["robust_solver"] = module
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def z_value(conf_level: float) -> float:
    alpha = 1.0 - conf_level
    return float(norm.ppf(1.0 - alpha / 2.0))


def normal_mean_ci(values: np.ndarray, conf_level: float = 0.99) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (0.0, 1.0)
    if len(values) == 1:
        v = float(np.clip(values[0], 0.0, 1.0))
        return (v, v)
    mu = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    z = z_value(conf_level)
    lo = max(0.0, mu - z * se)
    hi = min(1.0, mu + z * se)
    return (lo, hi)


def fit_linear_proxy(prices: np.ndarray, q_proxy: np.ndarray):
    A = np.vstack([np.ones_like(prices), prices]).T
    coef, *_ = np.linalg.lstsq(A, q_proxy, rcond=None)
    a, b = coef
    return lambda p: clamp01(a + b * np.asarray(p))


def fit_quadratic_proxy(prices: np.ndarray, q_proxy: np.ndarray):
    A = np.vstack([np.ones_like(prices), prices, prices**2]).T
    coef, *_ = np.linalg.lstsq(A, q_proxy, rcond=None)
    a, b, c = coef
    return lambda p: clamp01(a + b * np.asarray(p) + c * np.asarray(p) ** 2)


def _logit_curve(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(a * p + b))


def fit_logit_proxy(prices: np.ndarray, q_proxy: np.ndarray):
    y = np.clip(q_proxy, 1e-4, 1 - 1e-4)
    try:
        params, _ = curve_fit(_logit_curve, prices, y, p0=[1.0, 0.0], maxfev=20000)
        a, b = params
    except Exception:
        a, b = 1.0, 0.0
    return lambda p: clamp01(_logit_curve(np.asarray(p), a, b))


def load_tuna_data(wtna_csv: str = "tuna_data/wtna.csv", upctna_csv: str = "upctna.csv") -> pd.DataFrame:
    move = pd.read_csv(wtna_csv)
    upc = pd.read_csv(upctna_csv)

    move.columns = [c.strip().upper() for c in move.columns]
    upc.columns = [c.strip().upper() for c in upc.columns]

    if "OK" in move.columns:
        move = move[move["OK"] == 1].copy()

    move["QTY"] = pd.to_numeric(move["QTY"], errors="coerce").fillna(1.0)
    move["QTY"] = move["QTY"].where(move["QTY"] > 0, 1.0)

    move["PRICE"] = pd.to_numeric(move["PRICE"], errors="coerce")
    move["MOVE"] = pd.to_numeric(move["MOVE"], errors="coerce")
    move["WEEK"] = pd.to_numeric(move["WEEK"], errors="coerce")
    move["STORE"] = move["STORE"].astype(str)
    move["UPC"] = move["UPC"].astype(str)

    move["UNIT_PRICE"] = move["PRICE"] / move["QTY"]
    move["REVENUE"] = move["PRICE"] * move["MOVE"] / move["QTY"]

    move = move.replace([np.inf, -np.inf], np.nan)
    move = move.dropna(subset=["STORE", "UPC", "WEEK", "MOVE", "UNIT_PRICE"])
    move = move[(move["UNIT_PRICE"] > 0) & (move["MOVE"] >= 0)].copy()
    move["WEEK"] = move["WEEK"].astype(int)

    upc["UPC"] = upc["UPC"].astype(str)

    df = move.merge(
        upc[["UPC", "DESCRIP", "SIZE", "CASE", "NITEM", "COM_CODE"]],
        on="UPC",
        how="left"
    )
    return df


def filter_pairs(df: pd.DataFrame, min_weeks: int = 40, min_distinct_prices: int = 3, max_pairs: int = 40):
    stats = (
        df.groupby(["STORE", "UPC", "DESCRIP"], as_index=False)
        .agg(
            n_weeks=("WEEK", "nunique"),
            n_prices=("UNIT_PRICE", lambda x: x.round(4).nunique()),
            total_revenue=("REVENUE", "sum"),
            total_units=("MOVE", "sum"),
        )
    )
    stats = stats[(stats["n_weeks"] >= min_weeks) & (stats["n_prices"] >= min_distinct_prices)].copy()
    stats = stats.sort_values(["n_weeks", "total_revenue"], ascending=[False, False]).head(max_pairs)
    return stats


def rolling_windows(weeks, train_weeks=26, test_weeks=6, step_weeks=4):
    uniq = sorted(pd.unique(list(weeks)))
    out = []
    i = 0
    while i + train_weeks + test_weeks <= len(uniq):
        tr = uniq[i : i + train_weeks]
        te = uniq[i + train_weeks : i + train_weeks + test_weeks]
        out.append((tr, te))
        i += step_weeks
    return out


def choose_anchor_prices(train: pd.DataFrame) -> np.ndarray:
    distinct = np.sort(train["UNIT_PRICE"].round(4).unique())
    if len(distinct) < 3:
        raise ValueError("Need at least 3 distinct prices.")
    anchors = np.quantile(distinct, [0.25, 0.50, 0.75])
    mapped = []
    for a in anchors:
        mapped.append(float(distinct[np.argmin(np.abs(distinct - a))]))
    mapped = np.unique(np.round(mapped, 4))
    if len(mapped) == 3:
        return mapped.astype(float)
    idxs = np.linspace(0, len(distinct) - 1, 3).round().astype(int)
    return distinct[idxs].astype(float)


def choose_candidate_prices(train: pd.DataFrame, max_candidates: int = 25) -> np.ndarray:
    prices = np.sort(train["UNIT_PRICE"].round(4).unique())
    if len(prices) <= max_candidates:
        return prices.astype(float)
    qs = np.linspace(0, 1, max_candidates)
    chosen = np.quantile(prices, qs)
    return np.unique(np.round(chosen, 4)).astype(float)


def attach_proxy(train: pd.DataFrame, scale_quantile: float = 0.95) -> Tuple[pd.DataFrame, float]:
    tr = train.copy()
    n_hat = float(np.quantile(tr["MOVE"], scale_quantile))
    if n_hat <= 0:
        n_hat = max(float(tr["MOVE"].max()), 1.0)
    tr["Q_PROXY"] = np.clip(tr["MOVE"] / n_hat, 0.0, 1.0)
    return tr, n_hat


def summarize_anchor_quantiles(train: pd.DataFrame, anchor_prices: np.ndarray, ci_level: float = 0.99):
    tr = train.copy()
    diffs = np.abs(tr["UNIT_PRICE"].to_numpy()[:, None] - anchor_prices[None, :])
    tr["ANCHOR_ID"] = np.argmin(diffs, axis=1)

    rows = []
    q_lo, q_hi, realized = [], [], []

    for i in range(len(anchor_prices)):
        grp = tr[tr["ANCHOR_ID"] == i].copy()
        if grp.empty:
            idx = np.argsort(np.abs(tr["UNIT_PRICE"].to_numpy() - anchor_prices[i]))[: min(3, len(tr))]
            grp = tr.iloc[idx].copy()

        vals = grp["Q_PROXY"].to_numpy(dtype=float)
        lo, hi = normal_mean_ci(vals, conf_level=ci_level)
        rp = float(np.median(grp["UNIT_PRICE"].to_numpy(dtype=float)))

        realized.append(rp)
        q_lo.append(lo)
        q_hi.append(hi)

        rows.append(
            {
                "anchor_id": i,
                "anchor_requested": float(anchor_prices[i]),
                "anchor_used": rp,
                "n_obs": int(len(grp)),
                "q_mean": float(np.mean(vals)),
                "q_lo": float(lo),
                "q_hi": float(hi),
            }
        )

    realized = np.asarray(realized, dtype=float)
    order = np.argsort(realized)

    debug = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    return realized[order], np.asarray(q_lo)[order], np.asarray(q_hi)[order], debug


def price_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("UNIT_PRICE", as_index=False)
        .agg(
            mean_units=("MOVE", "mean"),
            mean_revenue=("REVENUE", "mean"),
            n_obs=("WEEK", "count")
        )
        .sort_values("UNIT_PRICE")
    )


def nearest_test_price(pred_price: float, test_table: pd.DataFrame):
    arr = test_table["UNIT_PRICE"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(arr - pred_price)))
    row = test_table.iloc[idx]
    return float(row["UNIT_PRICE"]), float(row["mean_revenue"])


def one_window_backtest(solver_mod, train: pd.DataFrame, test: pd.DataFrame, alpha: float = 1.0, ci_level: float = 0.99):
    train, n_hat = attach_proxy(train, scale_quantile=0.95)
    candidate_prices = choose_candidate_prices(train, max_candidates=25)
    anchor_prices = choose_anchor_prices(train)
    anchor_prices, q_lo, q_hi, anchor_debug = summarize_anchor_quantiles(train, anchor_prices, ci_level=ci_level)

    support_max = float(max(train["UNIT_PRICE"].max(), test["UNIT_PRICE"].max()) * 1.25)

    preds = {}

    tr_table = price_table(train)
    preds["ERM"] = float(tr_table.loc[tr_table["mean_revenue"].idxmax(), "UNIT_PRICE"])

    x = train["UNIT_PRICE"].to_numpy(dtype=float)
    y = train["Q_PROXY"].to_numpy(dtype=float)

    lin = fit_linear_proxy(x, y)
    log = fit_logit_proxy(x, y)
    quad = fit_quadratic_proxy(x, y)

    preds["Linear"] = float(candidate_prices[np.argmax(candidate_prices * lin(candidate_prices))])
    preds["Logit"] = float(candidate_prices[np.argmax(candidate_prices * log(candidate_prices))])
    preds["Quadratic"] = float(candidate_prices[np.argmax(candidate_prices * quad(candidate_prices))])

    amb = solver_mod.AmbiguitySet(anchor_prices, q_lo, q_hi, alpha, support_max)
    _, robust_price = amb.robust_value(candidate_prices, r_grid_size=201)
    preds["Robust"] = float(robust_price)

    te_table = price_table(test)
    oracle_rev = float(te_table["mean_revenue"].max())

    rows = []
    for method, p_hat in preds.items():
        eval_price, test_rev = nearest_test_price(p_hat, te_table)
        ratio = test_rev / oracle_rev if oracle_rev > 0 else np.nan
        rows.append(
            {
                "method": method,
                "predicted_price": float(p_hat),
                "evaluated_test_price": float(eval_price),
                "test_revenue": float(test_rev),
                "oracle_test_revenue": float(oracle_rev),
                "revenue_ratio": float(ratio),
                "n_hat": float(n_hat),
                "ci_level": float(ci_level),
            }
        )

    return pd.DataFrame(rows), anchor_debug


def run_backtest(
    solver_path="robust_pricing_rewritten_best_effort.py",
    wtna_csv="tuna_data/wtna.csv",
    upctna_csv="upctna.csv",
    out_dir="tuna_backtest_results",
    min_weeks=40,
    min_distinct_prices=3,
    max_pairs=40,
    train_weeks=26,
    test_weeks=6,
    step_weeks=4,
    alpha=1.0,
    ci_level=0.99,
):
    ensure_dir(out_dir)
    solver_mod = load_solver_module(solver_path)
    df = load_tuna_data(wtna_csv, upctna_csv)

    pair_stats = filter_pairs(df, min_weeks=min_weeks, min_distinct_prices=min_distinct_prices, max_pairs=max_pairs)
    pair_stats.to_csv(f"{out_dir}/eligible_pairs.csv", index=False)

    all_rows = []
    anchor_rows = []

    for _, pair in pair_stats.iterrows():
        store = str(pair["STORE"])
        upc = str(pair["UPC"])
        desc = str(pair["DESCRIP"])

        sub = df[(df["STORE"].astype(str) == store) & (df["UPC"].astype(str) == upc)].copy()
        sub = sub.sort_values("WEEK")

        windows = rolling_windows(sub["WEEK"].unique(), train_weeks=train_weeks, test_weeks=test_weeks, step_weeks=step_weeks)
        if not windows:
            continue

        for win_id, (train_ws, test_ws) in enumerate(windows):
            train = sub[sub["WEEK"].isin(train_ws)].copy()
            test = sub[sub["WEEK"].isin(test_ws)].copy()

            if train["UNIT_PRICE"].round(4).nunique() < 3 or test.empty:
                continue

            rows, dbg = one_window_backtest(solver_mod, train, test, alpha=alpha, ci_level=ci_level)

            rows["STORE"] = store
            rows["UPC"] = upc
            rows["DESCRIP"] = desc
            rows["window_id"] = win_id
            rows["train_start_week"] = min(train_ws)
            rows["train_end_week"] = max(train_ws)
            rows["test_start_week"] = min(test_ws)
            rows["test_end_week"] = max(test_ws)
            all_rows.append(rows)

            dbg["STORE"] = store
            dbg["UPC"] = upc
            dbg["DESCRIP"] = desc
            dbg["window_id"] = win_id
            anchor_rows.append(dbg)

    long_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    anchor_df = pd.concat(anchor_rows, ignore_index=True) if anchor_rows else pd.DataFrame()

    long_df.to_csv(f"{out_dir}/backtest_long.csv", index=False)
    anchor_df.to_csv(f"{out_dir}/anchor_debug.csv", index=False)

    summary = (
        long_df.groupby("method", as_index=False)
        .agg(
            mean_ratio=("revenue_ratio", "mean"),
            median_ratio=("revenue_ratio", "median"),
            std_ratio=("revenue_ratio", "std"),
            worst_decile=("revenue_ratio", lambda x: float(np.nanquantile(x, 0.10))),
            n_windows=("revenue_ratio", "count"),
        )
        .sort_values("mean_ratio", ascending=False)
    )
    summary.to_csv(f"{out_dir}/backtest_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary["method"], summary["mean_ratio"])
    plt.ylabel("Mean revenue ratio")
    plt.title("Canned Tuna rolling backtest")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mean_ratio_by_method.png", dpi=180)
    plt.close()

    methods = [m for m in ["Robust", "ERM", "Linear", "Logit", "Quadratic"] if m in long_df["method"].unique()]
    box_data = [long_df.loc[long_df["method"] == m, "revenue_ratio"].dropna().to_numpy() for m in methods]
    plt.figure(figsize=(8, 5))
    plt.boxplot(box_data, labels=methods, showfliers=False)
    plt.ylabel("Revenue ratio")
    plt.title("Canned Tuna revenue-ratio distribution")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ratio_boxplot.png", dpi=180)
    plt.close()

    return long_df, summary, anchor_df


if __name__ == "__main__":
    run_backtest()
