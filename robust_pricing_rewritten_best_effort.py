from __future__ import annotations

"""
Best-effort full rewrite for the course project based on
"Optimal Robust Pricing with Minimal Information".

What this file is
-----------------
A cleaner rewrite of the previous script with:
- a rewritten robust core,
- explicit upper-envelope construction J_alpha(r),
- equation (7)-style interval tightening,
- cached interval pieces for speed,
- experiment presets for Colab.

What this file is not
---------------------
The paper states exact closed-form O(K^2) / cubic-equation subroutines in Proposition 3 and Theorem 2.
Those appendix-level formulas are not fully machine-readable from the uploaded paper. So this script
keeps the paper's structure but evaluates the Nature problem numerically on dense grids. This is the
closest honest executable rewrite I can provide from the uploaded paper alone.
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, truncnorm, gamma as gamma_dist, norm

EPS = 1e-12


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def z_value(conf_level: float) -> float:
    return float(norm.ppf(1 - (1.0 - conf_level) / 2.0))


def wilson_interval(k: int, n: int, conf_level: float = 0.99) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = z_value(conf_level)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    rad = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - rad), min(1.0, center + rad)


def gaussian_ci(samples: np.ndarray, conf_level: float = 0.99) -> Tuple[float, float]:
    z = z_value(conf_level)
    mu = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    rad = z * sd / math.sqrt(max(1, len(samples)))
    return max(0.0, mu - rad), min(1.0, mu + rad)


def safe_exp(x: np.ndarray | float):
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, -700.0, 700.0)
    out = np.exp(arr)
    return out if out.ndim else float(out)


# -----------------------------------------------------------------------------
# Paper core objects: \bar\Gamma_alpha and \bar G_alpha
# -----------------------------------------------------------------------------

def gamma_bar(alpha: float, v):
    arr = np.asarray(v, dtype=float)
    if abs(alpha - 1.0) < 1e-12:
        out = safe_exp(-arr)
    else:
        base = np.maximum(1.0 + (1.0 - alpha) * arr, EPS)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            out = np.power(base, -1.0 / (1.0 - alpha))
    out = np.clip(out, 0.0, np.inf)
    return out if np.ndim(out) else float(out)


def gamma_bar_inv(alpha: float, y):
    arr = np.asarray(y, dtype=float)
    arr = np.maximum(arr, EPS)
    if abs(alpha - 1.0) < 1e-12:
        out = -np.log(arr)
    else:
        out = (np.power(arr, -(1.0 - alpha)) - 1.0) / (1.0 - alpha)
    return out if np.ndim(out) else float(out)


def segment_beta(alpha: float, qs: float, qt: float) -> float:
    ratio = np.clip(qt / max(qs, EPS), EPS, 1.0)
    return float(gamma_bar_inv(alpha, ratio))


def gbar(alpha: float, v, s: float, qs: float, t: float, qt: float):
    """Equation (6) generalized Pareto interpolation."""
    if not (t > s + EPS):
        raise ValueError(f"Need t>s in gbar, got s={s}, t={t}")

    qs = float(np.clip(qs, EPS, 1.0))
    qt = float(np.clip(qt, 0.0, qs))
    beta = segment_beta(alpha, qs, qt)
    varr = np.asarray(v, dtype=float)

    if abs(beta) < 1e-14:
        out = np.where(varr < s, 1.0, qs)
        return out if out.ndim else float(out)

    threshold = s + (t - s) * float(gamma_bar_inv(alpha, 1.0 / qs)) / beta
    out = np.ones_like(varr, dtype=float)
    mask = varr >= threshold
    if np.any(mask):
        scaled = beta * (varr[mask] - s) / (t - s)
        out[mask] = qs * gamma_bar(alpha, scaled)
    out = clamp01(out)
    return out if out.ndim else float(out)


def psi_segment(alpha: float, s: float, qs: float, t: float, qt: float) -> float:
    """Constant alpha-virtual value along a Gbar segment."""
    beta = segment_beta(alpha, qs, qt)
    if abs(alpha - 1.0) < 1e-12:
        return -(t - s) / max(beta, EPS)
    return (1.0 - alpha) * s - (t - s) / max(beta, EPS)


def root_intersection(f_left: Callable[[float], float], f_right: Callable[[float], float], a: float, b: float, grid: int = 200) -> float:
    diff = lambda x: f_left(x) - f_right(x)
    fa, fb = diff(a), diff(b)
    if abs(fa) < 1e-12:
        return a
    if abs(fb) < 1e-12:
        return b
    if np.sign(fa) != np.sign(fb):
        return float(brentq(diff, a, b))
    xs = np.linspace(a, b, grid)
    vals = np.array([abs(diff(x)) for x in xs])
    return float(xs[int(np.argmin(vals))])


# -----------------------------------------------------------------------------
# Demand models used in the uploaded paper
# -----------------------------------------------------------------------------

class DemandModel:
    def __init__(self, name: str, support_max: float, demand_fn: Callable[[np.ndarray], np.ndarray]):
        self.name = name
        self.support_max = float(support_max)
        self._demand_fn = demand_fn

    def demand(self, p):
        arr = np.asarray(p, dtype=float)
        out = np.clip(self._demand_fn(arr), 0.0, 1.0)
        return out if out.ndim else float(out)

    def oracle_price(self, grid_size: int = 4001) -> Tuple[float, float]:
        grid = np.linspace(max(1e-6, 0.0), self.support_max - 1e-6, grid_size)
        rev = grid * self.demand(grid)
        idx = int(np.argmax(rev))
        return float(grid[idx]), float(rev[idx])

    def sample_bernoulli(self, p: float, n: int, rng: np.random.Generator) -> np.ndarray:
        q = float(self.demand(np.array([p]))[0])
        return rng.binomial(1, q, size=n)

    def sample_noisy_demand(self, p: float, n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
        mu = float(self.demand(np.array([p]))[0])
        return mu + rng.normal(0.0, sigma, size=n)


def build_models() -> Dict[str, DemandModel]:
    models: Dict[str, DemandModel] = {}

    models["linear"] = DemandModel("linear", 1.0, lambda p: 1.0 - p)
    models["logit"] = DemandModel("logit", 1.0, lambda p: 1.0 / (1.0 + np.exp(4.0 * p + 4.0)))
    models["pareto"] = DemandModel("pareto", 1.0, lambda p: 1.0 / (4.0 * p + 1.0))
    models["piecewise_exponential"] = DemandModel(
        "piecewise_exponential",
        1.0,
        lambda p: np.where(p < 0.4, gbar(1.0, p, 0.0, 1.0, 0.4, 0.8), gbar(1.0, p, 0.4, 0.8, 1.0, 0.1)),
    )

    models["beta22"] = DemandModel("beta22", 1.0, lambda p: 1.0 - beta_dist.cdf(p, 2, 2))
    models["uniform01"] = DemandModel("uniform01", 1.0, lambda p: 1.0 - np.clip(p, 0.0, 1.0))

    c2 = chi2.cdf(2.0, df=5)
    models["chisq5_trunc02"] = DemandModel(
        "chisq5_trunc02", 2.0, lambda p: (c2 - chi2.cdf(np.clip(p, 0.0, 2.0), df=5)) / max(c2, EPS)
    )

    a, b = (0.0 - 5.0) / 2.5, (10.0 - 5.0) / 2.5
    tr = truncnorm(a=a, b=b, loc=5.0, scale=2.5)
    models["truncnorm_5_2p5_010"] = DemandModel(
        "truncnorm_5_2p5_010", 10.0, lambda p: 1.0 - tr.cdf(np.clip(p, 0.0, 10.0))
    )

    models["uniform010"] = DemandModel("uniform010", 10.0, lambda p: 1.0 - np.clip(p / 10.0, 0.0, 1.0))

    c10 = chi2.cdf(10.0, df=5)
    models["chisq5_trunc010"] = DemandModel(
        "chisq5_trunc010", 10.0, lambda p: (c10 - chi2.cdf(np.clip(p, 0.0, 10.0), df=5)) / max(c10, EPS)
    )

    a2, b2 = (0.0 - 1.0) / 0.5, (1.0 - 1.0) / 0.5
    tr_small = truncnorm(a=a2, b=b2, loc=1.0, scale=0.5)
    models["truncnorm_1_0p5_01"] = DemandModel(
        "truncnorm_1_0p5_01", 1.0, lambda p: 1.0 - tr_small.cdf(np.clip(p, 0.0, 1.0))
    )

    models["beta25"] = DemandModel("beta25", 1.0, lambda p: 1.0 - beta_dist.cdf(p, 2, 5))

    c_g = gamma_dist.cdf(10.0, a=4.0, scale=1.0)
    models["gamma4_trunc010"] = DemandModel(
        "gamma4_trunc010", 10.0,
        lambda p: (c_g - gamma_dist.cdf(np.clip(p, 0.0, 10.0), a=4.0, scale=1.0)) / max(c_g, EPS),
    )

    return models


MODELS = build_models()


# -----------------------------------------------------------------------------
# Robust ambiguity set and rewritten solver core
# -----------------------------------------------------------------------------

@dataclass
class IntervalPiece:
    a: float
    b: float
    kind: str
    params_left: Optional[Tuple[float, float, float, float]] = None
    params_right: Optional[Tuple[float, float, float, float]] = None
    split: Optional[float] = None


@dataclass
class AmbiguitySet:
    w_obs: np.ndarray
    ql_obs: np.ndarray
    qh_obs: np.ndarray
    alpha: float
    support_max: float
    _pieces: List[IntervalPiece] = field(default_factory=list, init=False)
    _J_cache_grid: Optional[np.ndarray] = field(default=None, init=False)
    _J_cache_vals: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self) -> None:
        order = np.argsort(self.w_obs)
        self.w_obs = np.asarray(self.w_obs, dtype=float)[order]
        self.ql_obs = clamp01(np.asarray(self.ql_obs, dtype=float)[order])
        self.qh_obs = clamp01(np.asarray(self.qh_obs, dtype=float)[order])
        if len(self.w_obs) == 0:
            self.ql_tight = np.array([], dtype=float)
            self.qh_tight = np.array([], dtype=float)
            self.is_feasible = True
            self._pieces = []
            return
        self.w_obs = np.clip(self.w_obs, 1e-6, self.support_max - 1e-6)
        if np.any(np.diff(self.w_obs) <= 0):
            raise ValueError("Observed prices must be strictly increasing.")
        self.ql_tight, self.qh_tight = self._tighten_bounds()
        self.is_feasible = bool(np.all(self.ql_tight <= self.qh_tight + 1e-9))
        if self.is_feasible:
            self._build_upper_pieces()
            self._build_J_cache()

    @property
    def w_ext(self) -> np.ndarray:
        return np.concatenate(([0.0], self.w_obs, [self.support_max]))

    @property
    def ql_ext(self) -> np.ndarray:
        return np.concatenate(([1.0], self.ql_tight, [0.0]))

    @property
    def qh_ext(self) -> np.ndarray:
        return np.concatenate(([1.0], self.qh_tight, [0.0]))

    # ---------- equation (7) tightening ----------

    def _tighten_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.concatenate(([0.0], self.w_obs))
        ql = np.concatenate(([1.0], self.ql_obs))
        qh = np.concatenate(([1.0], self.qh_obs))
        K = len(self.w_obs)

        ql_new = ql.copy()
        qh_new = qh.copy()
        for j in range(1, K + 1):
            upper_candidates = [qh[j]]
            for s in range(j + 1, K + 1):
                for t in range(s + 1, K + 1):
                    upper_candidates.append(gbar(self.alpha, w[j], w[s], qh[s], w[t], ql[t]))
            for s in range(0, j - 1):
                for t in range(s + 1, j):
                    upper_candidates.append(gbar(self.alpha, w[j], w[s], ql[s], w[t], qh[t]))

            lower_candidates = [ql[j]]
            for s in range(0, j):
                for t in range(j + 1, K + 1):
                    lower_candidates.append(gbar(self.alpha, w[j], w[s], ql[s], w[t], ql[t]))

            qh_new[j] = min(upper_candidates)
            ql_new[j] = max(lower_candidates)

        return clamp01(ql_new[1:]), clamp01(qh_new[1:])

    # ---------- lower envelope used in proposition 6 / maximin revenue ----------

    def lower_envelope(self, p):
        parr = np.asarray(p, dtype=float)
        w, ql = self.w_ext, self.ql_ext
        out = np.zeros_like(parr, dtype=float)
        for j in range(len(w) - 1):
            mask = (parr >= w[j]) & (parr <= w[j + 1] + 1e-12)
            if np.any(mask):
                out[mask] = gbar(self.alpha, parr[mask], w[j], ql[j], w[j + 1], ql[j + 1])
        return out if out.ndim else float(out)

    # ---------- L and R maps from section 4.1 ----------

    def _left_choice(self, x: float, qx: float) -> int:
        w, ql = self.w_ext, self.ql_ext
        cand = [i for i in range(len(w)) if w[i] < x - 1e-12]
        if not cand:
            return 0
        vals = [psi_segment(self.alpha, w[i], ql[i], x, qx) for i in cand]
        return int(cand[int(np.argmax(vals))])

    def _right_choice(self, x: float, qx: float) -> int:
        w, ql = self.w_ext, self.ql_ext
        cand = [i for i in range(len(w)) if w[i] > x + 1e-12]
        if not cand:
            return len(w) - 1
        vals = [psi_segment(self.alpha, x, qx, w[i], ql[i]) for i in cand]
        return int(cand[int(np.argmin(vals))])

    def _left_path(self, x: float, qx: float) -> List[Tuple[float, float]]:
        path = [(x, qx)]
        cx, cq = x, qx
        while cx > 1e-12:
            idx = self._left_choice(cx, cq)
            nx, nq = float(self.w_ext[idx]), float(self.ql_ext[idx])
            if nx >= cx - 1e-12:
                break
            path.append((nx, nq))
            cx, cq = nx, nq
            if idx == 0:
                break
        return path

    def _right_path(self, x: float, qx: float) -> List[Tuple[float, float]]:
        path = [(x, qx)]
        cx, cq = x, qx
        while cx < self.support_max - 1e-12:
            idx = self._right_choice(cx, cq)
            nx, nq = float(self.w_ext[idx]), float(self.ql_ext[idx])
            if nx <= cx + 1e-12:
                break
            path.append((nx, nq))
            cx, cq = nx, nq
            if idx == len(self.w_ext) - 1:
                break
        return path

    def F_left(self, p: float, x: float, qx: float) -> float:
        if p >= x:
            return float(qx)
        path = self._left_path(x, qx)
        for i in range(1, len(path)):
            rx, rq = path[i - 1]
            lx, lq = path[i]
            if lx - 1e-12 <= p <= rx + 1e-12:
                return float(gbar(self.alpha, p, lx, lq, rx, rq))
        return 1.0

    def F_right(self, p: float, x: float, qx: float) -> float:
        if p <= x:
            return float(qx)
        path = self._right_path(x, qx)
        for i in range(1, len(path)):
            lx, lq = path[i - 1]
            rx, rq = path[i]
            if lx - 1e-12 <= p <= rx + 1e-12:
                return float(gbar(self.alpha, p, lx, lq, rx, rq))
        return 0.0

    # ---------- upper envelope J_alpha(r) ----------

    def _build_upper_pieces(self) -> None:
        """
        Build interval pieces following the section 4.1 structure more closely than the old script:
        - first interval [0, w1]: right extremal curve from (w1, qh1)
        - interior interval [wi, w{i+1}]: min of left/right extremal curves split at m_i
        - last interval [wK, support]: left extremal curve into (wK, qhK)
        """
        w, ql, qh = self.w_ext, self.ql_ext, self.qh_ext
        n = len(self.w_obs)
        pieces: List[IntervalPiece] = []
        if n == 0:
            self._pieces = pieces
            return

        # first interval [0, w1]
        r1 = self._right_choice(w[1], qh[1])
        pieces.append(IntervalPiece(a=w[0], b=w[1], kind="right_only", params_right=(w[1], qh[1], w[r1], ql[r1])))

        # interior intervals [w_i, w_{i+1}], i=1..n-1 in extended indexing
        for i in range(1, n):
            Li = self._left_choice(w[i], qh[i])
            Ri1 = self._right_choice(w[i + 1], qh[i + 1])
            lf = lambda x, Li=Li, i=i: float(gbar(self.alpha, x, w[Li], ql[Li], w[i], qh[i]))
            rf = lambda x, i=i, Ri1=Ri1: float(gbar(self.alpha, x, w[i + 1], qh[i + 1], w[Ri1], ql[Ri1]))
            m = root_intersection(lf, rf, w[i], w[i + 1])
            pieces.append(
                IntervalPiece(
                    a=w[i], b=w[i + 1], kind="split",
                    params_left=(w[Li], ql[Li], w[i], qh[i]),
                    params_right=(w[i + 1], qh[i + 1], w[Ri1], ql[Ri1]),
                    split=float(np.clip(m, w[i], w[i + 1])),
                )
            )

        # last interval [wK, support]
        Lk = self._left_choice(w[n], qh[n])
        pieces.append(IntervalPiece(a=w[n], b=w[n + 1], kind="left_only", params_left=(w[Lk], ql[Lk], w[n], qh[n])))
        self._pieces = pieces

    def J_exact_piecewise(self, r: float) -> float:
        if len(self._pieces) == 0:
            return 1.0 if r <= 0 else 0.0
        rr = float(np.clip(r, 0.0, self.support_max))
        for piece in self._pieces:
            if piece.a - 1e-12 <= rr <= piece.b + 1e-12:
                if piece.kind == "right_only":
                    s, qs, t, qt = piece.params_right
                    return float(gbar(self.alpha, rr, s, qs, t, qt))
                if piece.kind == "left_only":
                    s, qs, t, qt = piece.params_left
                    return float(gbar(self.alpha, rr, s, qs, t, qt))
                if piece.kind == "split":
                    if rr < (piece.split or piece.a):
                        s, qs, t, qt = piece.params_left
                        return float(gbar(self.alpha, rr, s, qs, t, qt))
                    s, qs, t, qt = piece.params_right
                    return float(gbar(self.alpha, rr, s, qs, t, qt))
        # fallback
        return float(self.feasible_interval_at(rr)[1])

    def _build_J_cache(self, size: int = 2001) -> None:
        xs = np.linspace(0.0, self.support_max, size)
        vals = np.array([self.J_exact_piecewise(float(x)) for x in xs])
        self._J_cache_grid = xs
        self._J_cache_vals = clamp01(vals)

    def J(self, r: float) -> float:
        if self._J_cache_grid is None or self._J_cache_vals is None:
            return self.J_exact_piecewise(r)
        rr = float(np.clip(r, 0.0, self.support_max))
        return float(np.interp(rr, self._J_cache_grid, self._J_cache_vals))

    # ---------- generic feasible interval at continuous x ----------

    def feasible_interval_at(self, x: float) -> Tuple[float, float]:
        x = float(np.clip(x, 0.0, self.support_max))
        w, ql, qh = self.w_ext, self.ql_ext, self.qh_ext
        lower, upper = 0.0, 1.0

        for wi, lbi, ubi in zip(self.w_obs, self.ql_tight, self.qh_tight):
            if abs(x - wi) < 1e-10:
                lower = max(lower, float(lbi))
                upper = min(upper, float(ubi))

        for s in range(len(w) - 1):
            if w[s] >= x:
                break
            for t in range(s + 1, len(w)):
                if w[t] <= x:
                    continue
                lower = max(lower, gbar(self.alpha, x, w[s], ql[s], w[t], ql[t]))

        for s in range(len(w)):
            if w[s] <= x:
                continue
            for t in range(s + 1, len(w)):
                upper = min(upper, gbar(self.alpha, x, w[s], qh[s], w[t], ql[t]))

        for s in range(len(w) - 1):
            if w[s] >= x:
                break
            for t in range(s + 1, len(w)):
                if w[t] >= x:
                    break
                upper = min(upper, gbar(self.alpha, x, w[s], ql[s], w[t], qh[t]))

        lower = float(np.clip(lower, 0.0, 1.0))
        upper = float(np.clip(upper, lower, 1.0))
        return lower, upper

    # ---------- theorem 1 numerical solver ----------

    def nature_ratio(self, p: float, r: float) -> float:
        q_r = self.J(r)
        denom = r * q_r
        if denom <= EPS:
            return np.inf
        num = p * (self.F_left(p, r, q_r) if p < r else self.F_right(p, r, q_r))
        return float(num / denom)

    def robust_value(self, price_grid: np.ndarray, r_grid_size: int = 201) -> Tuple[float, float]:
        if not self.is_feasible:
            return 0.0, float(price_grid[0])
        r_grid = np.linspace(max(1e-6, self.support_max / (5 * r_grid_size)), self.support_max - 1e-6, r_grid_size)
        J_vals = np.array([self.J(float(r)) for r in r_grid])
        best_val, best_p = -np.inf, float(price_grid[0])
        for p in price_grid:
            vals = []
            for r, q_r in zip(r_grid, J_vals):
                denom = r * q_r
                if denom <= EPS:
                    continue
                num = p * (self.F_left(float(p), float(r), float(q_r)) if p < r else self.F_right(float(p), float(r), float(q_r)))
                vals.append(num / denom)
            cur = float(np.min(vals)) if vals else 0.0
            if cur > best_val:
                best_val, best_p = cur, float(p)
        return best_val, best_p

    def maximin_revenue_price(self, price_grid: np.ndarray) -> Tuple[float, float]:
        lb = self.lower_envelope(price_grid)
        rev = price_grid * lb
        idx = int(np.argmax(rev))
        return float(rev[idx]), float(price_grid[idx])


# -----------------------------------------------------------------------------
# Parametric baseline fits used in section 6.1
# -----------------------------------------------------------------------------

def fit_linear(prices: np.ndarray, obs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    A = np.vstack([np.ones_like(prices), prices]).T
    coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
    a, b = coef
    return lambda p: clamp01(a + b * np.asarray(p))


def fit_quadratic(prices: np.ndarray, obs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    A = np.vstack([np.ones_like(prices), prices, prices * prices]).T
    coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
    a, b, c = coef
    return lambda p: clamp01(a + b * np.asarray(p) + c * np.asarray(p) ** 2)


def _logit_fn(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(a * p + b))


def fit_logit(prices: np.ndarray, obs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    y = np.clip(obs, 1e-4, 1 - 1e-4)
    try:
        params, _ = curve_fit(_logit_fn, prices, y, p0=[4.0, 4.0], maxfev=20000)
        a, b = params
    except Exception:
        a, b = 4.0, 4.0
    return lambda p: clamp01(_logit_fn(np.asarray(p), a, b))


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def experiment_table2_feasibility(out_dir: str, n_trials: int = 10_000, n: int = 10, ci_level: float = 0.99, seed: int = 7) -> pd.DataFrame:
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    configs = [
        ("Beta(2,2)", MODELS["beta22"], 1.0),
        ("Uniform(0,1)", MODELS["uniform01"], 1.0),
        ("Chi-Squared(k=5)", MODELS["chisq5_trunc02"], 2.0),
        ("Truncated Normal(5,2.5)", MODELS["truncnorm_5_2p5_010"], 10.0),
    ]
    rows = []
    for label, model, support in configs:
        prices = np.array([0.25 * support, 0.50 * support, 0.75 * support])
        for alpha in [1.0, 0.0]:
            exact_ok, uncertain_ok = 0, 0
            for _ in range(n_trials):
                qhat, ql, qh = [], [], []
                for p in prices:
                    draws = model.sample_bernoulli(float(p), n, rng)
                    k = int(draws.sum())
                    qhat.append(k / n)
                    lo, hi = wilson_interval(k, n, ci_level)
                    ql.append(lo)
                    qh.append(hi)
                exact = AmbiguitySet(prices, np.array(qhat), np.array(qhat), alpha, support)
                uncertain = AmbiguitySet(prices, np.array(ql), np.array(qh), alpha, support)
                exact_ok += int(exact.is_feasible)
                uncertain_ok += int(uncertain.is_feasible)
            rows.append({
                "distribution": label,
                "alpha": alpha,
                "exact_feasibility": exact_ok / n_trials,
                "uncertain_feasibility": uncertain_ok / n_trials,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table2_feasibility.csv"), index=False)
    return df


def experiment_61_parametric_benchmarks(
    out_dir: str,
    n_trials: int = 100,
    sigma: float = 0.25,
    n_samples_per_price: int = 100,
    ci_level: float = 0.99,
    alpha: float = 1.0,
    r_grid_size: int = 161,
    seed: int = 11,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    names = ["linear", "logit", "pareto", "piecewise_exponential"]
    grid_sizes = [5, 10, 20]
    rows = []

    for model_name in names:
        model = MODELS[model_name]
        _, oracle_r = model.oracle_price(4001)
        for grid_n in grid_sizes:
            price_grid = np.linspace(0.01, 0.99, grid_n)
            ratios_by_method = {"Linear": [], "Logit": [], "Quadratic": [], "Ours": []}
            for _ in range(n_trials):
                noisy_means, ql, qh = [], [], []
                for p in price_grid:
                    draws = model.sample_noisy_demand(float(p), n_samples_per_price, sigma, rng)
                    noisy_means.append(float(np.mean(draws)))
                    lo, hi = gaussian_ci(draws, ci_level)
                    ql.append(lo)
                    qh.append(hi)
                noisy_means = np.array(noisy_means)
                ql = np.array(ql)
                qh = np.array(qh)

                f_lin = fit_linear(price_grid, noisy_means)
                f_log = fit_logit(price_grid, noisy_means)
                f_quad = fit_quadratic(price_grid, noisy_means)
                methods = {"Linear": f_lin, "Logit": f_log, "Quadratic": f_quad}
                for mname, fn in methods.items():
                    preds = np.clip(fn(price_grid), 0.0, 1.0)
                    p_hat = float(price_grid[int(np.argmax(price_grid * preds))])
                    ratios_by_method[mname].append((p_hat * float(model.demand(p_hat))) / max(oracle_r, EPS))

                amb = AmbiguitySet(price_grid, ql, qh, alpha, 1.0)
                _, p_rob = amb.robust_value(price_grid, r_grid_size=r_grid_size)
                ratios_by_method["Ours"].append((p_rob * float(model.demand(p_rob))) / max(oracle_r, EPS))

            for method, vals in ratios_by_method.items():
                vals = np.asarray(vals, dtype=float)
                rows.append({
                    "demand_model": model_name,
                    "grid_size": grid_n,
                    "method": method,
                    "mean_ratio": float(np.mean(vals)),
                    "median_ratio": float(np.median(vals)),
                    "std_ratio": float(np.std(vals)),
                    "worst_decile": float(np.quantile(vals, 0.1)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "section61_parametric_benchmarks.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    method_order = ["Linear", "Logit", "Quadratic", "Ours"]
    for ax, model_name in zip(axes.ravel(), names):
        sub = df[df["demand_model"] == model_name]
        for method in method_order:
            vals = sub[sub["method"] == method].sort_values("grid_size")
            ax.plot(vals["grid_size"], vals["median_ratio"], marker="o", label=method)
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Discretization Size")
        ax.set_ylabel("Competitive Ratio")
        ax.set_ylim(0.0, 1.01)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure9_parametric_benchmarks.png"), dpi=180)
    plt.close(fig)
    return df


def inverse_survival_price(model: DemandModel, q: float, grid_size: int = 5001) -> float:
    grid = np.linspace(1e-6, model.support_max - 1e-6, grid_size)
    demand = model.demand(grid)
    idx = int(np.argmin(np.abs(demand - q)))
    return float(grid[idx])


def experiment_63_sequential(
    out_dir: str,
    alpha: float = 1.0,
    candidate_grid_size: int = 21,
    q_grid_size: int = 5,
    r_grid_size: int = 121,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    configs = [
        ("Uniform(0,10)", MODELS["uniform010"]),
        ("Chi-squared(k=5)", MODELS["chisq5_trunc010"]),
        ("Truncated Normal(5,2.5)", MODELS["truncnorm_5_2p5_010"]),
        ("Gamma(4)", MODELS["gamma4_trunc010"]),
    ]
    rows = []
    for init_q in [0.01, 0.99]:
        for label, model in configs:
            w_hist = [inverse_survival_price(model, init_q)]
            q_hist = [float(model.demand(w_hist[0]))]
            for round_id in range(1, 6):
                amb_current = AmbiguitySet(np.array(w_hist), np.array(q_hist), np.array(q_hist), alpha, model.support_max)
                grid = np.linspace(1e-6, model.support_max - 1e-6, candidate_grid_size)
                grid = np.array([g for g in grid if min(abs(g - w) for w in w_hist) > 1e-8])
                if round_id > 1:
                    value_now, _ = amb_current.robust_value(np.linspace(1e-6, model.support_max - 1e-6, 61), r_grid_size=r_grid_size)
                    rows.append({
                        "distribution": label,
                        "initial_q1": init_q,
                        "round": round_id - 1,
                        "competitive_ratio": value_now,
                        "last_price": w_hist[-1],
                    })
                best_w, best_val = float(grid[0]), -np.inf
                for w in grid:
                    ql_w, qh_w = amb_current.feasible_interval_at(float(w))
                    q_candidates = np.array([ql_w]) if qh_w - ql_w < 1e-10 else np.linspace(ql_w, qh_w, q_grid_size)
                    worst_after = np.inf
                    for q in q_candidates:
                        w_aug = np.unique(np.round(np.array(sorted(w_hist + [float(w)]), dtype=float), 10))
                        q_map = {round(float(wi), 10): float(qi) for wi, qi in zip(w_hist, q_hist)}
                        q_map[round(float(w), 10)] = float(q)
                        q_aug = np.array([q_map[round(float(wi), 10)] for wi in w_aug])
                        amb_aug = AmbiguitySet(w_aug, q_aug, q_aug, alpha, model.support_max)
                        val_after, _ = amb_aug.robust_value(np.linspace(1e-6, model.support_max - 1e-6, 61), r_grid_size=r_grid_size)
                        worst_after = min(worst_after, val_after)
                    if worst_after > best_val:
                        best_val, best_w = worst_after, float(w)
                w_hist.append(best_w)
                q_hist.append(float(model.demand(best_w)))
                order = np.argsort(w_hist)
                w_hist = list(np.array(w_hist)[order])
                q_hist = list(np.array(q_hist)[order])
            amb_final = AmbiguitySet(np.array(w_hist), np.array(q_hist), np.array(q_hist), alpha, model.support_max)
            final_val, _ = amb_final.robust_value(np.linspace(1e-6, model.support_max - 1e-6, 61), r_grid_size=r_grid_size)
            rows.append({
                "distribution": label,
                "initial_q1": init_q,
                "round": 5,
                "competitive_ratio": final_val,
                "last_price": w_hist[-1],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "section63_sequential.csv"), index=False)

    for init_q in [0.01, 0.99]:
        sub = df[df["initial_q1"] == init_q]
        fig, ax = plt.subplots(figsize=(7, 4))
        for dist in sub["distribution"].unique():
            vals = sub[sub["distribution"] == dist].sort_values("round")
            ax.plot(vals["round"], vals["competitive_ratio"], marker="o", label=dist)
        ax.set_title(f"q1 = {init_q}")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Competitive Ratio")
        ax.set_ylim(0.0, 1.01)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"figure11_q1_{str(init_q).replace('.', 'p')}.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for dist in sub["distribution"].unique():
            vals = sub[sub["distribution"] == dist].sort_values("round")
            ax.plot(vals["round"], vals["last_price"], marker="o", label=dist)
        ax.set_title(f"q1 = {init_q}")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Experiment Price")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"figure12_q1_{str(init_q).replace('.', 'p')}.png"), dpi=180)
        plt.close(fig)
    return df


# -----------------------------------------------------------------------------
# Section 6.4 bandits / robust initialization
# -----------------------------------------------------------------------------

def revenue_reward(model: DemandModel, p: float, rng: np.random.Generator) -> float:
    buy = rng.binomial(1, float(model.demand(p)))
    return float(p * buy)


def run_ucb(model: DemandModel, T: int, rng: np.random.Generator) -> np.ndarray:
    n_arms = max(5, int(np.sqrt(T / max(np.log(T), 1.0))))
    arms = np.linspace(1e-6, model.support_max - 1e-6, n_arms)
    counts = np.zeros(n_arms, dtype=int)
    succ = np.zeros(n_arms, dtype=float)
    regret = np.zeros(T)
    _, oracle_r = model.oracle_price(4001)
    cum = 0.0
    t = 0
    for i, p in enumerate(arms):
        if t >= T:
            break
        reward = revenue_reward(model, float(p), rng)
        counts[i] += 1
        succ[i] += reward / max(float(p), EPS)
        cum += oracle_r - reward
        regret[t] = cum
        t += 1
    while t < T:
        mean_conv = succ / np.maximum(counts, 1)
        conf = np.sqrt(2.0 * np.log(t + 1) / np.maximum(counts, 1))
        i = int(np.argmax(arms * np.clip(mean_conv + conf, 0.0, 1.0)))
        p = float(arms[i])
        reward = revenue_reward(model, p, rng)
        counts[i] += 1
        succ[i] += reward / max(p, EPS)
        cum += oracle_r - reward
        regret[t] = cum
        t += 1
    return regret


def run_ts(model: DemandModel, T: int, rng: np.random.Generator) -> np.ndarray:
    n_arms = max(5, int(np.sqrt(T / max(np.log(T), 1.0))))
    arms = np.linspace(1e-6, model.support_max - 1e-6, n_arms)
    a = np.ones(n_arms)
    b = np.ones(n_arms)
    regret = np.zeros(T)
    _, oracle_r = model.oracle_price(4001)
    cum = 0.0
    for t in range(T):
        sampled = rng.beta(a, b)
        i = int(np.argmax(arms * sampled))
        p = float(arms[i])
        buy = int(rng.binomial(1, float(model.demand(p))))
        reward = float(p * buy)
        a[i] += buy
        b[i] += 1 - buy
        cum += oracle_r - reward
        regret[t] = cum
    return regret


def run_sp_like(model: DemandModel, T: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = 0.0, model.support_max
    stats: Dict[float, List[float]] = {}
    regret = np.zeros(T)
    _, oracle_r = model.oracle_price(4001)
    cum = 0.0
    def pull(p: float) -> float:
        r = revenue_reward(model, p, rng)
        stats.setdefault(round(p, 6), []).append(r)
        return r
    t = 0
    while t < T:
        x1 = lo + (hi - lo) / 3.0
        x2 = hi - (hi - lo) / 3.0
        r1 = pull(x1)
        cum += oracle_r - r1
        regret[t] = cum
        t += 1
        if t >= T:
            break
        r2 = pull(x2)
        cum += oracle_r - r2
        regret[t] = cum
        t += 1
        m1 = float(np.mean(stats[round(x1, 6)]))
        m2 = float(np.mean(stats[round(x2, 6)]))
        if m1 <= m2:
            lo = x1
        else:
            hi = x2
        if hi - lo < 1e-3:
            lo = max(0.0, lo - 0.05 * model.support_max)
            hi = min(model.support_max, hi + 0.05 * model.support_max)
    return regret


def run_q5(model: DemandModel, T: int, rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    grid5 = np.linspace(1e-6, model.support_max - 1e-6, 5)
    counts = {float(p): 0 for p in grid5}
    succ = {float(p): 0 for p in grid5}
    regret = np.zeros(T)
    _, oracle_r = model.oracle_price(4001)
    cum, t = 0.0, 0
    selected_price = float(grid5[2])

    def play(p: float) -> None:
        nonlocal cum, t
        if t >= T:
            return
        buy = int(rng.binomial(1, float(model.demand(p))))
        reward = float(p * buy)
        cum += oracle_r - reward
        regret[t] = cum
        t += 1

    while t < min(T, 1000):
        for p in grid5:
            for _ in range(10):
                if t >= min(T, 1000):
                    break
                buy = int(rng.binomial(1, float(model.demand(p))))
                counts[float(p)] += 1
                succ[float(p)] += buy
                reward = float(p * buy)
                cum += oracle_r - reward
                regret[t] = cum
                t += 1
            if t >= min(T, 1000):
                break
        ql, qh = [], []
        for p in grid5:
            lo, hi = wilson_interval(succ[float(p)], counts[float(p)], 0.99)
            ql.append(lo)
            qh.append(hi)
        amb = AmbiguitySet(grid5, np.array(ql), np.array(qh), alpha, model.support_max)
        _, selected_price = amb.maximin_revenue_price(np.linspace(1e-6, model.support_max - 1e-6, 201))
        for _ in range(50):
            if t >= min(T, 1000):
                break
            play(selected_price)
    while t < T:
        play(selected_price)
    return regret


def experiment_64_bandits(out_dir: str, T: int = 10_000, n_trials: int = 10, alpha: float = 1.0) -> pd.DataFrame:
    ensure_dir(out_dir)
    configs = [
        ("Beta(2,5)", MODELS["beta25"]),
        ("Uniform(0,1)", MODELS["uniform01"]),
        ("Truncated Normal(1,0.5)", MODELS["truncnorm_1_0p5_01"]),
        ("Chi-squared(k=5)", MODELS["chisq5_trunc010"]),
    ]
    rows = []
    for label, model in configs:
        regrets = {"UCB": [], "TS": [], "SP": [], "Q5": []}
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 + trial)
            regrets["UCB"].append(run_ucb(model, T, rng))
            regrets["TS"].append(run_ts(model, T, rng))
            regrets["SP"].append(run_sp_like(model, T, rng))
            regrets["Q5"].append(run_q5(model, T, rng, alpha=alpha))
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, arrs in regrets.items():
            stacked = np.vstack(arrs)
            mean_curve = stacked.mean(axis=0)
            ax.plot(np.arange(1, T + 1), mean_curve, label=method)
            for t in [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]:
                idx = min(T, t) - 1
                rows.append({"distribution": label, "method": method, "t": t, "mean_regret": float(mean_curve[idx])})
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(label)
        ax.legend()
        fig.tight_layout()
        safe_name = label.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        fig.savefig(os.path.join(out_dir, f"figure13_{safe_name}.png"), dpi=180)
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "section64_bandits.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# Presets and CLI
# -----------------------------------------------------------------------------

PRESETS = {
    "quick": {
        "table2": {"n_trials": 500},
        "sec61": {"n_trials": 10, "r_grid_size": 121},
        "sec63": {"candidate_grid_size": 11, "q_grid_size": 3, "r_grid_size": 81},
        "sec64": {"T": 2000, "n_trials": 3},
    },
    "medium": {
        "table2": {"n_trials": 3000},
        "sec61": {"n_trials": 30, "r_grid_size": 161},
        "sec63": {"candidate_grid_size": 15, "q_grid_size": 5, "r_grid_size": 101},
        "sec64": {"T": 5000, "n_trials": 5},
    },
    "full": {
        "table2": {"n_trials": 10000},
        "sec61": {"n_trials": 100, "r_grid_size": 201},
        "sec63": {"candidate_grid_size": 21, "q_grid_size": 5, "r_grid_size": 121},
        "sec64": {"T": 10000, "n_trials": 10},
    },
}


def run_all(output_root: str = "rewritten_reproduction_results", preset: str = "medium") -> None:
    ensure_dir(output_root)
    cfg = PRESETS[preset]
    print(f"Running preset={preset}")
    print("Running Table 2 feasibility experiment...")
    experiment_table2_feasibility(os.path.join(output_root, "table2"), **cfg["table2"])
    print("Running Section 6.1 parametric benchmark experiment...")
    experiment_61_parametric_benchmarks(os.path.join(output_root, "section61"), **cfg["sec61"])
    print("Running Section 6.3 sequential experimentation...")
    experiment_63_sequential(os.path.join(output_root, "section63"), **cfg["sec63"])
    print("Running Section 6.4 bandit / robust initialization...")
    experiment_64_bandits(os.path.join(output_root, "section64"), **cfg["sec64"])
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="medium")
    parser.add_argument("--out", default="rewritten_reproduction_results")
    args = parser.parse_args()
    run_all(args.out, args.preset)


if __name__ == "__main__":
    main()
