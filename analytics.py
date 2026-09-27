"""Module 2: Quantitative risk & return engine.

All inputs are monthly decimal returns. Annualization uses 12 periods/year.
Ratios are returned as decimals (0.12 = 12%); formatting happens downstream.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PERIODS = 12
NAVY = "#1B365D"
SLATE = "#4A5568"
LIGHT_SLATE = "#A0AEC0"
GRID = "#E2E8F0"


def _nan_to_none(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def cumulative_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def cagr(r: pd.Series) -> float:
    years = len(r) / PERIODS
    return float((1 + cumulative_return(r)) ** (1 / years) - 1)


def annualized_volatility(r: pd.Series) -> float:
    return float(r.std(ddof=1) * math.sqrt(PERIODS))


def wealth_index(r: pd.Series, dates: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Growth path including the starting point one month before the first return."""
    idx = pd.DatetimeIndex(dates)
    start_date = idx[0] - pd.offsets.MonthEnd(1)
    values = np.concatenate([[start_value], start_value * (1 + r.to_numpy()).cumprod()])
    return pd.Series(values, index=idx.insert(0, start_date))


def drawdown_series(wealth: pd.Series) -> pd.Series:
    return wealth / wealth.cummax() - 1


def drawdown_details(wealth: pd.Series) -> dict:
    dd = drawdown_series(wealth)
    trough = dd.idxmin()
    max_dd = float(dd.min())
    if max_dd >= 0:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None,
                "recovery_date": None, "peak_to_trough_months": 0,
                "recovery_months": 0, "peak_to_recovery_months": 0, "recovered": True}

    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered_points = after[after >= wealth.loc[peak]]
    recovery = recovered_points.index[0] if not recovered_points.empty else None

    def months_between(a, b):
        return (b.to_period("M") - a.to_period("M")).n

    return {
        "max_drawdown": max_dd,
        "peak_date": peak,
        "trough_date": trough,
        "recovery_date": recovery,
        "peak_to_trough_months": months_between(peak, trough),
        "recovery_months": months_between(trough, recovery) if recovery is not None else None,
        "peak_to_recovery_months": months_between(peak, recovery) if recovery is not None else None,
        "recovered": recovery is not None,
    }


def sharpe_ratio(r: pd.Series, rf: pd.Series) -> float:
    excess = r - rf
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * math.sqrt(PERIODS)) if sd > 0 else float("nan")


def sortino_ratio(r: pd.Series, rf: pd.Series) -> float:
    """Annualized excess return over downside deviation (MAR = risk-free rate)."""
    excess = r - rf
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * math.sqrt(PERIODS)
    return float(excess.mean() * PERIODS / downside) if downside > 0 else float("nan")


def capture_ratio(fund: pd.Series, bench: pd.Series, up: bool) -> float:
    """Morningstar-style capture: annualized geometric return in up (or down) benchmark months."""
    mask = bench > 0 if up else bench < 0
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    f = (1 + fund[mask]).prod() ** (PERIODS / n) - 1
    b = (1 + bench[mask]).prod() ** (PERIODS / n) - 1
    return float(f / b) if b != 0 else float("nan")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Return a structured dictionary of performance, risk, and exposure metrics."""
    fund, bench, rf = df["Fund_Return"], df["Benchmark_Return"], df["Risk_Free_Rate"]
    n = len(df)

    fund_cagr, bench_cagr = cagr(fund), cagr(bench)
    fund_wealth = wealth_index(fund, df["Date"])
    bench_wealth = wealth_index(bench, df["Date"])
    fund_dd, bench_dd = drawdown_details(fund_wealth), drawdown_details(bench_wealth)

    # CAPM regression on excess returns: (Rf - Rrf) = a + b (Rb - Rrf)
    reg = stats.linregress(bench - rf, fund - rf)
    active = fund - bench
    tracking_error = float(active.std(ddof=1) * math.sqrt(PERIODS))
    info_ratio = float(active.mean() * PERIODS / tracking_error) if tracking_error > 0 else float("nan")

    metrics = {
        "period": {
            "start": df["Date"].iloc[0],
            "end": df["Date"].iloc[-1],
            "months": n,
            "years": n / PERIODS,
            "avg_risk_free_annual": float((1 + rf.mean()) ** PERIODS - 1),
        },
        "performance": {
            "fund_cagr": fund_cagr,
            "benchmark_cagr": bench_cagr,
            "fund_cumulative_return": cumulative_return(fund),
            "benchmark_cumulative_return": cumulative_return(bench),
            "excess_return_annualized": fund_cagr - bench_cagr,
            "best_month": float(fund.max()),
            "worst_month": float(fund.min()),
            "pct_positive_months": float((fund > 0).mean()),
            "batting_average": float((fund > bench).mean()),
        },
        "risk": {
            "fund_volatility": annualized_volatility(fund),
            "benchmark_volatility": annualized_volatility(bench),
            "fund_max_drawdown": fund_dd["max_drawdown"],
            "benchmark_max_drawdown": bench_dd["max_drawdown"],
            "drawdown": fund_dd,
            "benchmark_drawdown": bench_dd,
        },
        "risk_adjusted": {
            "fund_sharpe": sharpe_ratio(fund, rf),
            "benchmark_sharpe": sharpe_ratio(bench, rf),
            "fund_sortino": sortino_ratio(fund, rf),
            "benchmark_sortino": sortino_ratio(bench, rf),
            "information_ratio": info_ratio,
        },
        "market_exposure": {
            "alpha_annualized": float(reg.intercept * PERIODS),
            "alpha_t_stat": float(reg.intercept / reg.intercept_stderr) if reg.intercept_stderr else float("nan"),
            "beta": float(reg.slope),
            "r_squared": float(reg.rvalue ** 2),
            "correlation": float(np.corrcoef(fund, bench)[0, 1]),
            "tracking_error": tracking_error,
            "up_capture": capture_ratio(fund, bench, up=True),
            "down_capture": capture_ratio(fund, bench, up=False),
        },
    }
    return metrics


def metrics_for_prompt(metrics: dict) -> dict:
    """JSON-friendly view of the metrics (dates as strings, NaN as None)."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m")
        if isinstance(value, float):
            return None if not np.isfinite(value) else round(value, 4)
        return value
    return clean(metrics)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LIGHT_SLATE)
    ax.tick_params(colors=SLATE, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def generate_charts(df: pd.DataFrame, fund_ticker: str, benchmark_ticker: str, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    fund_w = wealth_index(df["Fund_Return"], df["Date"], 10_000)
    bench_w = wealth_index(df["Benchmark_Return"], df["Date"], 10_000)
    paths: dict[str, Path] = {}

    # 1. Growth of $10,000
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    ax.plot(bench_w.index, bench_w, color=LIGHT_SLATE, linewidth=1.6, label=benchmark_ticker)
    ax.plot(fund_w.index, fund_w, color=NAVY, linewidth=2.0, label=fund_ticker)
    # End-value labels, nudged apart vertically when the two lines finish close together.
    fund_end, bench_end = fund_w.iloc[-1], bench_w.iloc[-1]
    close = abs(fund_end - bench_end) < 0.06 * max(fund_w.max(), bench_w.max())
    nudge = 5 if close else 0
    for series, color, dy in ((fund_w, NAVY, nudge if fund_end >= bench_end else -nudge),
                              (bench_w, SLATE, nudge if bench_end > fund_end else -nudge)):
        ax.annotate(f"${series.iloc[-1]:,.0f}", (series.index[-1], series.iloc[-1]),
                    xytext=(4, dy), textcoords="offset points", fontsize=7, color=color,
                    va="center", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.set_title("Growth of $10,000", loc="left", fontsize=10, color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _style_axes(ax)
    ax.margins(x=0.12)
    fig.tight_layout()
    paths["growth"] = out_dir / "growth.png"
    fig.savefig(paths["growth"], bbox_inches="tight")
    plt.close(fig)

    # 2. Drawdown (underwater)
    fund_dd, bench_dd = drawdown_series(fund_w), drawdown_series(bench_w)
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    ax.plot(bench_dd.index, bench_dd, color=LIGHT_SLATE, linewidth=1.2, label=benchmark_ticker)
    ax.fill_between(fund_dd.index, fund_dd, 0, color=NAVY, alpha=0.25, linewidth=0)
    ax.plot(fund_dd.index, fund_dd, color=NAVY, linewidth=1.4, label=fund_ticker)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(top=0.005)
    ax.set_title("Drawdown (Underwater)", loc="left", fontsize=10, color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    _style_axes(ax)
    fig.tight_layout()
    paths["drawdown"] = out_dir / "drawdown.png"
    fig.savefig(paths["drawdown"], bbox_inches="tight")
    plt.close(fig)

    # 3. Monthly returns heatmap (fund), with a per-year compounded total column
    r = df.set_index("Date")["Fund_Return"]
    grid = (r.to_frame("ret")
              .assign(Year=r.index.year, Month=r.index.strftime("%b"))
              .pivot(index="Year", columns="Month", values="ret"))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid = grid.reindex(columns=months)
    grid["Total"] = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)

    height = 0.9 + 0.32 * len(grid)
    fig, ax = plt.subplots(figsize=(7.4, height), dpi=200)
    limit = max(0.01, float(np.nanpercentile(np.abs(grid[months].to_numpy()), 95)))
    annot = grid.map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}")
    sns.heatmap(grid, ax=ax, cmap="RdYlGn", center=0, vmin=-limit, vmax=limit,
                annot=annot, fmt="", annot_kws={"fontsize": 6.5}, cbar=False,
                linewidths=0.8, linecolor="white")
    ax.axvline(len(months), color="white", linewidth=3)
    ax.set_title(f"{fund_ticker} Monthly Returns (%)   ·   Total = compounded return of months shown",
                 loc="left", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=SLATE, labelsize=7, length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    paths["heatmap"] = out_dir / "heatmap.png"
    fig.savefig(paths["heatmap"], bbox_inches="tight")
    plt.close(fig)

    return paths
