# Robust Quantile-Based Pricing

A research-oriented implementation of robust pricing under limited demand information, combining paper reproduction, historical backtesting, and an adaptive confidence-interval extension.

## Project overview

Pricing decisions are difficult when the seller has only sparse observations of customer purchase probabilities. This project investigates a robust alternative: selecting prices that perform well across a set of demand distributions consistent with the available data.

The work has three parts:

1. **Reproduce** a robust quantile-pricing framework and its synthetic experiments.
2. **Backtest** robust and conventional pricing methods on scanner-style retail data.
3. **Extend** the framework with adaptive confidence-interval selection.

## Research questions

- How well can robust pricing perform when demand is only partially observed?
- How does it compare with empirical-risk minimisation and parametric baselines?
- How do confidence-interval width and feasibility affect pricing performance?
- Can an adaptive rule improve the trade-off between robustness and revenue?

## Methods

### Robust pricing

The implementation constructs ambiguity sets from observed prices and confidence intervals for purchase probabilities. It then searches for a price that maximises worst-case revenue over the feasible demand set.

### Rolling historical backtest

The backtesting pipeline:

- cleans scanner-style retail data,
- creates rolling train/test windows,
- estimates purchase-probability proxies,
- compares robust pricing with ERM, linear, quadratic, and logit baselines,
- reports revenue and stability metrics across windows.

### Adaptive confidence intervals

The project introduces and evaluates two data-driven policies:

- **AdaptiveFeasible** — selects the narrowest confidence level that produces a feasible ambiguity set.
- **AdaptiveValidated** — selects among feasible confidence levels using validation revenue.

These are compared with fixed 90%, 95%, and 99% confidence levels using feasibility, interval width, revenue ratio, and selected confidence level.

## Repository structure

```text
.
├── data/                 # Data archives and data notes
├── docs/                 # Report and implementation alignment
├── notebooks/            # Exploratory and supporting analyses
├── results/              # Reproduction, backtest, and extension outputs
├── scripts/              # Executable experiment pipelines
├── src/                  # Reusable robust-pricing components
└── tests/                # Core, backtest, and CI tests
```

## Main experiment scripts

| Script | Purpose |
|---|---|
| `scripts/01_reproduce_paper.py` | Reproduce the robust-pricing framework and synthetic experiments |
| `scripts/02_rolling_backtest.py` | Evaluate methods on rolling historical windows |
| `scripts/03_adaptive_ci_experiments.py` | Compare fixed and adaptive confidence-interval policies |

## Reproducing the study

```bash
pip install -r requirements.txt
python scripts/01_reproduce_paper.py
python scripts/02_rolling_backtest.py
python scripts/03_adaptive_ci_experiments.py
```

Some experiments require the data and result archives documented under `data/` and `results/`.

## Outputs

The pipelines produce:

- feasibility and performance tables,
- rolling-window backtest summaries,
- revenue-ratio comparisons,
- confidence-interval diagnostics,
- publication-style plots and CSV result files.

## Tools and skills demonstrated

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Statistical Modelling` · `Robust Optimisation` · `Backtesting` · `Experimental Design`

This project demonstrates the ability to reproduce technical research, design empirical evaluations, build reusable analytical code, and extend an existing method with a testable new idea.

## Author

Built by [sincerelyindra](https://github.com/sincerelyindra), with interests in machine learning, economics, decision science, and robust optimisation.
