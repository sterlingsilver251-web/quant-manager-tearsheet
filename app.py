"""Streamlit web app for the Manager Performance & Due Diligence Tear-Sheet.

Run:  streamlit run app.py

Reuses the CLI pipeline modules unchanged in behavior:
data_loader -> analytics -> ai_commentary -> pdf_generator.
"""

from __future__ import annotations

import os
import re
import tempfile
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_commentary import CommentaryResult, generate_commentary
from analytics import compute_metrics, generate_charts
from data_loader import DataLoadError, MarketData, fetch_macro_snapshot, load_market_data
from pdf_generator import build_tearsheet

NAVY = "#1B365D"
SLATE = "#4A5568"
TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
HORIZONS = [1, 3, 5, 10]
PREVIEW_IMAGE = Path(__file__).resolve().parent / "assets" / "preview.png"

st.set_page_config(
    page_title="Institutional Manager Tear-Sheet & Due Diligence Platform",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
  h1, h2, h3, h4 {{ color: {NAVY}; }}
  [data-testid="stSidebar"] {{ background: #F4F6F9; border-right: 1px solid #E2E8F0; }}
  [data-testid="stSidebar"] h2 {{ font-size: 1.05rem; letter-spacing: .02em; }}

  .hero {{ background: {NAVY}; color: #fff; border-radius: 10px; padding: 22px 28px; margin-bottom: 18px;
           display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px; }}
  .hero .eyebrow {{ color: #A0AEC0; font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
  .hero .title {{ font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin-top: 4px; }}
  .hero .title span {{ font-weight: 400; color: #CBD5E0; font-size: 1.05rem; margin-left: 10px; }}
  .hero .subtitle {{ color: #CBD5E0; font-size: .92rem; margin-top: 6px; }}
  .hero .meta {{ text-align: right; font-size: .85rem; color: #E2E8F0; line-height: 1.7; }}
  .hero .meta b {{ color: #A0AEC0; font-size: .68rem; letter-spacing: .1em; margin-right: 8px; }}

  .kpi {{ background: #F7FAFC; border: 1px solid #E2E8F0; border-left: 4px solid {NAVY};
          border-radius: 8px; padding: 14px 18px; height: 100%; }}
  .kpi .label {{ color: {SLATE}; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
  .kpi .value {{ color: {NAVY}; font-size: 1.9rem; font-weight: 800; line-height: 1.25; margin-top: 4px; }}
  .kpi .sub {{ color: {SLATE}; font-size: .82rem; margin-top: 2px; }}
  .pos {{ color: #2F855A !important; }}
  .neg {{ color: #C53030 !important; }}

  .headline {{ border-left: 4px solid {NAVY}; background: #F7FAFC; padding: 14px 18px; border-radius: 6px;
               color: {NAVY}; font-size: 1.05rem; font-style: italic; }}
  .role {{ display: inline-block; background: {NAVY}; color: #fff; border-radius: 999px; padding: 4px 14px;
           font-size: .8rem; font-weight: 700; letter-spacing: .03em; }}
  .section-title {{ color: {NAVY}; font-weight: 700; font-size: 1.02rem; margin-bottom: 6px; }}
  .section-body {{ color: #2D3748; font-size: .95rem; line-height: 1.6; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(x, digits=1, signed=False) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def num(x, digits=2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{digits}f}"


def sign_class(x, good_if_positive=True) -> str:
    if x is None or x != x or x == 0:
        return ""
    return "pos" if (x > 0) == good_if_positive else "neg"


def server_groq_key() -> str | None:
    """Key configured by the app owner (Streamlit secrets or environment). Never shown in the UI."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets.toml present
        key = None
    return key or os.getenv("GROQ_API_KEY") or None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(fund: str, bench: str, years: int) -> MarketData:
    return load_market_data(fund, bench, years)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_macro_snapshot() -> dict[str, str]:
    return fetch_macro_snapshot()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(fund: str, bench: str, years: int, groq_key: str | None) -> dict:
    with st.status("Building tear-sheet...", expanded=True) as status:
        st.write(f"Fetching {fund} vs {bench} prices and risk-free rate...")
        market = cached_market_data(fund, bench, years)
        macro = cached_macro_snapshot()

        st.write("Computing risk/return analytics...")
        metrics = compute_metrics(market.returns)

        period = metrics["period"]
        context = {
            "fund_ticker": market.fund_ticker, "fund_name": market.fund_name, "fund_type": market.fund_type,
            "benchmark_ticker": market.benchmark_ticker, "benchmark_name": market.benchmark_name,
            "window": f"{period['start']:%b %Y} to {period['end']:%b %Y}", "months": period["months"],
            "requested_years": years, "rf_source": market.rf_source,
        }
        st.write("Generating executive due diligence commentary...")
        commentary = generate_commentary(context, metrics, macro, groq_api_key=groq_key)

        st.write("Rendering charts and PDF...")
        with tempfile.TemporaryDirectory(prefix="tearsheet_web_") as tmp:
            tmp_path = Path(tmp)
            chart_paths = generate_charts(market.returns, market.fund_ticker, market.benchmark_ticker, tmp_path)
            charts = {name: path.read_bytes() for name, path in chart_paths.items()}
            pdf_path = build_tearsheet(market, metrics, chart_paths, commentary, tmp_path / "pdf")
            pdf_bytes = pdf_path.read_bytes()

        status.update(label="Tear-sheet ready", state="complete", expanded=False)

    return {"market": market, "metrics": metrics, "commentary": commentary,
            "charts": charts, "pdf": pdf_bytes, "years": years}


# ---------------------------------------------------------------------------
# Detailed metrics table
# ---------------------------------------------------------------------------

LABELS = {
    "fund_cagr": "Annualized Return (CAGR) - Fund", "benchmark_cagr": "Annualized Return (CAGR) - Benchmark",
    "fund_cumulative_return": "Cumulative Return - Fund", "benchmark_cumulative_return": "Cumulative Return - Benchmark",
    "excess_return_annualized": "Excess Return (annualized)", "pct_positive_months": "% Positive Months",
    "batting_average": "Batting Average vs. Benchmark", "fund_volatility": "Annualized Volatility - Fund",
    "benchmark_volatility": "Annualized Volatility - Benchmark", "fund_max_drawdown": "Maximum Drawdown - Fund",
    "benchmark_max_drawdown": "Maximum Drawdown - Benchmark", "fund_sharpe": "Sharpe Ratio - Fund",
    "benchmark_sharpe": "Sharpe Ratio - Benchmark", "fund_sortino": "Sortino Ratio - Fund",
    "benchmark_sortino": "Sortino Ratio - Benchmark", "information_ratio": "Information Ratio",
    "alpha_annualized": "Alpha (annualized, CAPM)", "alpha_t_stat": "Alpha t-Statistic", "r_squared": "R-Squared",
    "tracking_error": "Tracking Error", "up_capture": "Up-Market Capture", "down_capture": "Down-Market Capture",
    "avg_risk_free_annual": "Average Risk-Free Rate (annual)", "peak_to_trough_months": "Peak-to-Trough (months)",
    "recovery_months": "Trough-to-Recovery (months)", "peak_to_recovery_months": "Peak-to-Recovery (months)",
}
CATEGORY_NAMES = {"period": "Period", "performance": "Performance", "risk": "Risk & Drawdown",
                  "risk_adjusted": "Risk-Adjusted", "market_exposure": "Market Exposure",
                  "drawdown": "Fund Max Drawdown Episode", "benchmark_drawdown": "Benchmark Max Drawdown Episode"}
MONTH_KEYS = {"months", "peak_to_trough_months", "recovery_months", "peak_to_recovery_months"}
RATIO_KEYS = {"fund_sharpe", "benchmark_sharpe", "fund_sortino", "benchmark_sortino", "information_ratio",
              "alpha_t_stat", "beta", "r_squared", "correlation"}


def format_value(key: str, value) -> str:
    if value is None:
        return "Not recovered" if "recovery" in key else "—"
    if isinstance(value, pd.Timestamp):
        return f"{value:%b %Y}"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key in MONTH_KEYS:
        return f"{int(value)}"
    if key == "years":
        return f"{value:.1f}"
    if key in RATIO_KEYS:
        return num(value)
    if isinstance(value, (int, float)):
        return pct(value, 2)
    return str(value)


def metrics_table(metrics: dict) -> pd.DataFrame:
    rows = []

    def walk(category: str, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                walk(CATEGORY_NAMES.get(key, key), value)
            else:
                label = LABELS.get(key, key.replace("_", " ").title())
                rows.append({"Category": category, "Metric": label, "Value": format_value(key, value)})

    for section, values in metrics.items():
        walk(CATEGORY_NAMES.get(section, section), values)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def kpi(col, label: str, value: str, sub: str, value_class: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{escape(label)}</div>'
        f'<div class="value {value_class}">{escape(value)}</div>'
        f'<div class="sub">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def render_report(report: dict) -> None:
    market: MarketData = report["market"]
    metrics: dict = report["metrics"]
    commentary: CommentaryResult = report["commentary"]
    perf, risk, radj, exp, period = (metrics[k] for k in
                                     ("performance", "risk", "risk_adjusted", "market_exposure", "period"))
    fund, bench = market.fund_ticker, market.benchmark_ticker

    st.markdown(f"""
    <div class="hero">
      <div>
        <div class="eyebrow">Manager Performance &amp; Due Diligence Tear-Sheet</div>
        <div class="title">{escape(fund)} <span>vs. {escape(bench)}</span></div>
        <div class="subtitle">{escape(market.fund_name)} &nbsp;·&nbsp; benchmarked against {escape(market.benchmark_name)}</div>
      </div>
      <div class="meta">
        <div><b>HORIZON</b>{report['years']}-Year ({period['months']} months)</div>
        <div><b>PERIOD</b>{period['start']:%b %Y} – {period['end']:%b %Y}</div>
        <div><b>RISK-FREE</b>{escape(market.rf_source)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for note in market.warnings:
        st.warning(note)

    left, right = st.columns([3, 1], vertical_alignment="center")
    left.caption("All figures use month-end adjusted-close returns (distributions reinvested). "
                 "Methodology is detailed on page 2 of the PDF.")
    right.download_button(
        "⬇  Download PDF Tear-Sheet",
        data=report["pdf"],
        file_name=f"TearSheet_{fund}.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
    )

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Annualized Return", pct(perf["fund_cagr"]), f"{bench}: {pct(perf['benchmark_cagr'])}")
    kpi(c2, "Excess Return / Alpha", pct(perf["excess_return_annualized"], signed=True),
        f"Alpha {pct(exp['alpha_annualized'], signed=True)} · t-stat {num(exp['alpha_t_stat'])}",
        sign_class(perf["excess_return_annualized"]))
    kpi(c3, "Sharpe Ratio", num(radj["fund_sharpe"]),
        f"{bench}: {num(radj['benchmark_sharpe'])} · Sortino {num(radj['fund_sortino'])}")
    dd = risk["drawdown"]
    recovery = f"recovered in {dd['peak_to_recovery_months']} mo" if dd["recovered"] else "not yet recovered"
    kpi(c4, "Max Drawdown", pct(risk["fund_max_drawdown"]),
        f"{bench}: {pct(risk['benchmark_max_drawdown'])} · {recovery}", "neg")

    st.write("")
    tab_charts, tab_ai, tab_table = st.tabs(
        ["📈  Performance & Risk", "🧠  Due Diligence Commentary", "📋  Detailed Metrics"])

    with tab_charts:
        g, d = st.columns(2)
        with g.container(border=True):
            st.image(report["charts"]["growth"], width="stretch")
        with d.container(border=True):
            st.image(report["charts"]["drawdown"], width="stretch")
        with st.container(border=True):
            st.image(report["charts"]["heatmap"], width="stretch")

    with tab_ai:
        head, role = st.columns([4, 1], vertical_alignment="center")
        head.markdown(f'<div class="headline">{escape(commentary.headline)}</div>', unsafe_allow_html=True)
        role.markdown(f'<div style="text-align:right"><div style="color:{SLATE};font-size:.7rem;font-weight:700;'
                      f'letter-spacing:.08em;margin-bottom:4px">RECOMMENDED ROLE</div>'
                      f'<span class="role">{escape(commentary.recommended_role)}</span></div>',
                      unsafe_allow_html=True)
        st.write("")
        for title, text in commentary.sections:
            with st.expander(title, expanded=True):
                st.markdown(f'<div class="section-body">{escape(text)}</div>', unsafe_allow_html=True)
        if commentary.source.startswith("Template"):
            st.info(f"{commentary.source}. Add a free Groq API key in the sidebar for AI-written commentary.")
        else:
            st.caption(f"Generated by {commentary.source}. Review before distribution.")

    with tab_table:
        table = metrics_table(metrics)
        st.dataframe(table, hide_index=True, width="stretch", height=min(38 * (len(table) + 1), 900),
                     column_config={"Category": st.column_config.TextColumn(width="medium"),
                                    "Metric": st.column_config.TextColumn(width="large"),
                                    "Value": st.column_config.TextColumn(width="small")})
        st.download_button("Download monthly return series (CSV)",
                           data=market.returns.to_csv(index=False).encode(),
                           file_name=f"{fund}_vs_{bench}_monthly_returns.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Tear-Sheet Parameters")
    with st.form("inputs", border=False):
        fund_in = st.text_input("Fund ticker", value="AGTHX", max_chars=15)
        bench_in = st.text_input("Benchmark ticker", value="SPY", max_chars=15)
        years_in = st.selectbox("Time horizon", HORIZONS, index=HORIZONS.index(5),
                                format_func=lambda y: f"{y} Year" + ("s" if y > 1 else ""))

        st.divider()
        st.subheader("AI Commentary")
        server_key = server_groq_key()
        if server_key:
            st.caption("✅ A Groq API key is configured on this server.")
            user_key = st.text_input("Use your own Groq key instead (optional)", type="password")
        else:
            user_key = st.text_input(
                "Groq API key (optional)", type="password",
                help="Free key from console.groq.com/keys. Used only for this session and never stored. "
                     "Leave blank for rule-based template commentary.")
        submitted = st.form_submit_button("Generate Tear-Sheet", type="primary", width="stretch")

    st.caption("Data: Yahoo Finance (adjusted prices), FRED (T-bill & macro). "
               "For institutional discussion only. Not investment advice.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Institutional Manager Tear-Sheet & Due Diligence Platform")

if submitted:
    fund, bench = fund_in.strip().upper(), bench_in.strip().upper()
    if not fund or not bench:
        st.error("Enter both a fund ticker and a benchmark ticker.")
    elif not TICKER_RE.match(fund) or not TICKER_RE.match(bench):
        st.error("Tickers may only contain letters, numbers, and . - ^ = (e.g. AGTHX, SPY, BRK-B).")
    elif fund == bench:
        st.error("Fund and benchmark tickers must be different.")
    else:
        try:
            st.session_state["report"] = run_pipeline(fund, bench, int(years_in),
                                                      user_key.strip() or server_key)
        except DataLoadError as exc:
            st.session_state.pop("report", None)
            st.error(f"Could not load market data: {exc}")
        except Exception as exc:
            st.session_state.pop("report", None)
            st.error(f"Something went wrong while building the tear-sheet ({exc.__class__.__name__}). "
                     "Please try again or choose different tickers.")
            with st.expander("Technical details"):
                st.code(str(exc) or repr(exc))

if "report" in st.session_state:
    render_report(st.session_state["report"])
elif not submitted:
    st.markdown(
        "Generate an institutional-grade performance and due diligence tear-sheet for any mutual fund "
        "or ETF. Enter a fund and benchmark in the sidebar and click **Generate Tear-Sheet**.")
    if PREVIEW_IMAGE.exists():
        st.image(str(PREVIEW_IMAGE), caption="Sample output: AGTHX vs. SPY, 5 years", width=560)
