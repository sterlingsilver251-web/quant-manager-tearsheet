"""Module 3: AI executive due-diligence commentary.

Sends the quantitative metrics and a dated macro snapshot to a free,
open-weight LLM and gets back four structured narrative sections. Any
OpenAI-compatible endpoint works; two free presets are built in:

  groq    Hosted open models (Llama 3.3 70B) on a free tier. Needs GROQ_API_KEY
          from https://console.groq.com/keys.
  ollama  Open models running locally (https://ollama.com). No key, no cost.

Provider is chosen by LLM_PROVIDER, otherwise automatically: Groq if
GROQ_API_KEY is set, else a running local Ollama server. If neither is
available, or the call fails, a rule-based template commentary built from the
same metrics is returned instead, so the tear-sheet always renders.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from pydantic import BaseModel, Field

from analytics import metrics_for_prompt

PROVIDERS = {
    # name: (base URL, API-key env var or None, default model)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "ollama": ("http://localhost:11434/v1", None, "llama3.1:8b"),
}
REQUEST_TIMEOUT = 180.0

SECTION_TITLES = [
    "Viability & Performance Attribution",
    "Downside Risk & Liquidity Profile",
    "Market Alignment & Current Macro Trends",
    "Portfolio Fit & Asset Allocation Role",
]

SYSTEM_PROMPT = """You are a Senior Institutional Due Diligence Officer at a $10 billion multi-family office and pension fund. You write the executive commentary on one-page manager tear-sheets read by an investment committee.

How you work:
- Ground every claim in the metrics and macro data provided. Cite specific figures (e.g., "alpha of -2.8% with a t-stat of -1.1").
- Do not invent holdings, fees, manager names, AUM, or events that are not in the data. Where a judgment needs information you do not have (e.g., holdings-based factor exposure, fee drag, capacity), say what the committee should verify.
- Infer style and factor tilt only from the return statistics (beta, capture ratios, correlation, drawdown behavior) and say so.
- Distinguish statistical significance from noise: a short track record or a low alpha t-stat is not evidence of skill.
- Be direct and decision-oriented, in the measured tone of an institutional memo. No hype, no hedging boilerplate, no investment advice disclaimers (the report adds its own).
- Metrics arrive as decimals; always write them as percentages in prose (0.104 -> 10.4%), except ratios such as beta, Sharpe, t-stat, and R-squared.
- Each section is one tight paragraph of 90-130 words. The headline is a single sentence. Plain text only: no markdown, bullets, or headings.

Respond with a single JSON object and nothing else, using exactly these string keys:
{
  "headline": "one-sentence overall verdict",
  "viability_performance_attribution": "alpha generation, excess returns, consistency, statistical significance",
  "downside_risk_liquidity": "max drawdown, recovery, down-capture, volatility, liquidity/redemption considerations",
  "market_alignment_macro": "how the implied factor exposure fits current rates, growth vs. value, inflation",
  "portfolio_fit_allocation": "recommended institutional portfolio role and sizing guidance",
  "recommended_role": "short label, e.g. Core Growth Sleeve, Satellite Alpha Generator, Defensive Hedge, Watch List"
}"""


class DueDiligenceCommentary(BaseModel):
    headline: str = Field(description="One-sentence overall verdict for the committee")
    viability_performance_attribution: str = Field(
        description="Alpha generation, excess returns, consistency, and statistical significance")
    downside_risk_liquidity: str = Field(
        description="Max drawdown, recovery, down-capture, volatility, and structural liquidity/redemption considerations")
    market_alignment_macro: str = Field(
        description="How the fund's implied factor exposure fits current rates, growth vs. value, and inflation trends")
    portfolio_fit_allocation: str = Field(
        description="Recommended role in an institutional portfolio and sizing guidance")
    recommended_role: str = Field(
        description="Short label, e.g. 'Core Growth Sleeve', 'Satellite Alpha Generator', 'Defensive Hedge', 'Watch List'")


@dataclass
class CommentaryResult:
    headline: str
    sections: list[tuple[str, str]]
    recommended_role: str
    source: str   # shown in the PDF footer, e.g. "groq (llama-3.3-70b-versatile)" or "Template: <reason>"


def _build_user_prompt(context: dict, metrics: dict, macro: dict[str, str]) -> str:
    macro_lines = "\n".join(f"- {k}: {v}" for k, v in macro.items()) or "- (macro data unavailable)"
    return f"""Prepare the due-diligence commentary for this manager.

<fund>
Ticker: {context['fund_ticker']}
Name: {context['fund_name']}
Vehicle type (Yahoo Finance quoteType): {context['fund_type']}
Benchmark: {context['benchmark_ticker']} ({context['benchmark_name']})
Analysis window: {context['window']} ({context['months']} monthly observations; requested {context['requested_years']} years)
Risk-free source: {context['rf_source']}
</fund>

<metrics>
All values are decimals (0.12 = 12%). Alpha/beta come from a CAPM regression of monthly excess returns; alpha is annualized (x12). Capture ratios use annualized geometric returns in benchmark up/down months (1.00 = 100%).
{json.dumps(metrics, indent=2)}
</metrics>

<macro_backdrop>
{macro_lines}
</macro_backdrop>

Write the headline, the four sections, and a short recommended-role label as the JSON object described."""


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------

def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def template_commentary(context: dict, metrics: dict, reason: str) -> CommentaryResult:
    m = metrics_for_prompt(metrics)
    peak = metrics["risk"]["drawdown"]["peak_date"]
    peak_label = f"{peak:%B %Y}" if peak is not None else "starting"
    perf, risk, radj, exp = m["performance"], m["risk"], m["risk_adjusted"], m["market_exposure"]
    fund, bench = context["fund_ticker"], context["benchmark_ticker"]
    dd = risk["drawdown"]

    excess = perf["excess_return_annualized"] or 0
    alpha, t = exp["alpha_annualized"] or 0, exp["alpha_t_stat"]
    beta = exp["beta"] or 1
    up, down = exp["up_capture"], exp["down_capture"]
    significant = t is not None and abs(t) >= 2

    viability = (
        f"Over {context['window']}, {fund} compounded at {_pct(perf['fund_cagr'])} annually versus "
        f"{_pct(perf['benchmark_cagr'])} for {bench}, an annualized excess return of {_pct(excess)}. "
        f"CAPM alpha was {_pct(alpha)} (t-stat {_num(t)}), which is "
        f"{'statistically meaningful' if significant else 'not statistically distinguishable from zero'} over this window. "
        f"The information ratio of {_num(radj['information_ratio'])} and a batting average of {_pct(perf['batting_average'], 0)} "
        f"{'indicate reasonably consistent value added' if (radj['information_ratio'] or 0) > 0.3 else 'do not yet demonstrate a repeatable edge'}."
    )

    recovery = (f"recovered after {dd['recovery_months']} months" if dd["recovered"]
                else "has not yet recovered to its prior peak")
    downside = (
        f"Maximum drawdown was {_pct(risk['fund_max_drawdown'])} against {_pct(risk['benchmark_max_drawdown'])} for the benchmark; "
        f"the fund fell for {dd['peak_to_trough_months']} months from its {peak_label} peak and {recovery}. "
        f"Annualized volatility of {_pct(risk['fund_volatility'])} and down-capture of {_pct(down, 0)} "
        f"{'point to a more defensive profile than the index' if (down or 1) < 0.95 else 'show the fund has not cushioned drawdowns relative to the index'}. "
        f"As a {context['fund_type'].lower().replace('mutualfund', 'mutual fund')} vehicle, liquidity is expected to be daily; "
        f"the committee should still confirm redemption terms, share-class fees, and capacity."
    )

    macro = (
        f"With beta of {beta:.2f} and up/down capture of {_pct(up, 0)}/{_pct(down, 0)}, the return pattern implies "
        f"{'a higher-beta, pro-cyclical' if beta > 1.05 else 'a lower-beta, defensive' if beta < 0.95 else 'a market-like'} exposure. "
        f"This template commentary does not interpret the current rate, inflation, or growth-versus-value backdrop; "
        f"a holdings-based factor review is recommended before drawing macro conclusions."
    )

    if significant and alpha > 0:
        role = "Satellite Alpha Generator"
    elif beta < 0.9 and (down or 1) < 0.9:
        role = "Defensive Diversifier"
    elif (radj["information_ratio"] or 0) > 0 or abs(excess) < 0.01:
        role = "Core Equity Sleeve"
    else:
        role = "Watch List"
    fit = (
        f"On the quantitative evidence alone, {fund} screens as a candidate for a {role.lower()} role. "
        f"R-squared of {_num(exp['r_squared'])} and tracking error of {_pct(exp['tracking_error'])} describe how much "
        f"of the risk budget it would consume relative to {bench}. Final sizing should follow a qualitative review of "
        f"process, team stability, and fees."
    )

    return CommentaryResult(
        headline=f"{fund} delivered {_pct(excess)} annualized excess return versus {bench} with beta of {beta:.2f}.",
        sections=list(zip(SECTION_TITLES, [viability, downside, macro, fit])),
        recommended_role=role,
        source=f"Template commentary ({reason})",
    )


# ---------------------------------------------------------------------------
# Free / open-source LLM (OpenAI-compatible chat API)
# ---------------------------------------------------------------------------

def _ollama_running(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.removesuffix("/v1") + "/api/tags", timeout=2):
            return True
    except Exception:
        return False


def resolve_provider() -> tuple[str, str, str, str] | str:
    """Return (name, base_url, api_key, model), or a reason string if none is usable.

    Env overrides: LLM_PROVIDER (groq | ollama | custom), LLM_MODEL, LLM_BASE_URL, LLM_API_KEY.
    """
    choice = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not choice:
        if os.environ.get("GROQ_API_KEY"):
            choice = "groq"
        elif _ollama_running(PROVIDERS["ollama"][0]):
            choice = "ollama"
        else:
            return "no free LLM configured: set GROQ_API_KEY or start Ollama"

    if choice in PROVIDERS:
        base_url, key_var, default_model = PROVIDERS[choice]
    elif choice == "custom":
        base_url, key_var, default_model = os.environ.get("LLM_BASE_URL", ""), "LLM_API_KEY", ""
        if not base_url:
            return "LLM_PROVIDER=custom requires LLM_BASE_URL"
    else:
        return f"unknown LLM_PROVIDER '{choice}'"

    api_key = os.environ.get("LLM_API_KEY") or (os.environ.get(key_var) if key_var else None)
    if key_var and not api_key:
        return f"{key_var} not set"
    model = os.environ.get("LLM_MODEL") or default_model
    if not model:
        return "LLM_MODEL not set"
    base_url = os.environ.get("LLM_BASE_URL") or base_url
    # Local servers ignore the key, but the client requires a non-empty value.
    return choice, base_url, api_key or "not-needed", model


def _parse_json_object(text: str) -> dict:
    """Parse the model's JSON, tolerating code fences or stray text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def generate_commentary(context: dict, metrics: dict, macro: dict[str, str]) -> CommentaryResult:
    """Return LLM commentary, or the template fallback. Never raises for API problems."""
    resolved = resolve_provider()
    if isinstance(resolved, str):
        return template_commentary(context, metrics, resolved)
    provider, base_url, api_key, model = resolved

    try:
        import openai
    except Exception as exc:  # missing or broken install
        return template_commentary(context, metrics, f"openai SDK unavailable: {exc.__class__.__name__}")

    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=2)
    prompt = _build_user_prompt(context, metrics_for_prompt(metrics), macro)
    label = f"{provider} ({model})"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2500,
        )
    except openai.AuthenticationError:
        return template_commentary(context, metrics, f"{provider} rejected the API key")
    except openai.NotFoundError:
        hint = f"; run: ollama pull {model}" if provider == "ollama" else ""
        return template_commentary(context, metrics, f"model '{model}' not found on {provider}{hint}")
    except openai.RateLimitError:
        return template_commentary(context, metrics, f"{provider} rate limit reached")
    except openai.APIStatusError as exc:
        return template_commentary(context, metrics, f"{provider} API error {exc.status_code}")
    except openai.APIConnectionError:
        return template_commentary(context, metrics, f"could not reach {provider} at {base_url}")
    except Exception as exc:
        return template_commentary(context, metrics, f"unexpected error: {exc.__class__.__name__}")

    choice = response.choices[0] if response.choices else None
    text = (choice.message.content or "") if choice else ""
    try:
        parsed = DueDiligenceCommentary.model_validate(_parse_json_object(text))
    except Exception:
        reason = "output was truncated" if choice and choice.finish_reason == "length" else "output was not valid JSON"
        return template_commentary(context, metrics, f"{label} {reason}")

    return CommentaryResult(
        headline=parsed.headline.strip(),
        sections=list(zip(SECTION_TITLES, [
            parsed.viability_performance_attribution.strip(),
            parsed.downside_risk_liquidity.strip(),
            parsed.market_alignment_macro.strip(),
            parsed.portfolio_fit_allocation.strip(),
        ])),
        recommended_role=parsed.recommended_role.strip(),
        source=label,
    )
