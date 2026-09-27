#!/usr/bin/env python3
"""Module 5: CLI entry point for the Manager Performance & Due Diligence Tear-Sheet.

Examples:
    python main.py                                   # interactive prompts
    python main.py --fund AGTHX --benchmark SPY --years 5
    python main.py -f PRWCX -b AOR -y 10 --no-open
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
TOTAL_STEPS = 4

USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


class StepFailed(Exception):
    pass


@contextmanager
def step(n: int, label: str):
    print(f"{_c('36', f'[{n}/{TOTAL_STEPS}]')} {label}...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        print(f" {_c('31', 'failed')}")
        raise
    print(f" {_c('32', 'done')} {_c('2', f'({time.perf_counter() - t0:.1f}s)')}")


def prompt(label: str, default: str | None = None, validate=None):
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip() or (default or "")
        if not value:
            print("  A value is required.")
            continue
        if validate:
            error = validate(value)
            if error:
                print(f"  {error}")
                continue
        return value


def validate_years(value: str) -> str | None:
    try:
        years = int(value)
    except ValueError:
        return "Enter a whole number of years (e.g. 3, 5, 10)."
    return None if 1 <= years <= 30 else "Choose between 1 and 30 years."


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a manager performance & due diligence tear-sheet PDF.")
    p.add_argument("-f", "--fund", help="Fund ticker, e.g. AGTHX")
    p.add_argument("-b", "--benchmark", help="Benchmark ticker, e.g. SPY")
    p.add_argument("-y", "--years", type=int, help="Time horizon in years, e.g. 3, 5, 10")
    p.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-open", action="store_true", help="Do not open the PDF when finished")
    p.add_argument("--no-ai", action="store_true", help="Skip the LLM call and use template commentary")
    return p.parse_args(argv)


def open_file(path: Path) -> None:
    opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
    try:
        subprocess.Popen([*opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass  # headless machine; the path is printed anyway


def run(args: argparse.Namespace) -> Path:
    # Heavy imports are deferred so --help and prompts are instant.
    from ai_commentary import generate_commentary, template_commentary
    from analytics import compute_metrics, generate_charts
    from data_loader import DataLoadError, fetch_macro_snapshot, load_market_data
    from pdf_generator import build_tearsheet

    fund = (args.fund or prompt("Fund ticker", "AGTHX")).upper()
    bench = (args.benchmark or prompt("Benchmark ticker", "SPY")).upper()
    if args.years is not None:
        if err := validate_years(str(args.years)):
            raise StepFailed(err)
        years = args.years
    else:
        years = int(prompt("Time horizon in years", "5", validate_years))
    print()

    try:
        with step(1, f"Fetching {fund} vs {bench} prices and risk-free rate"):
            market = load_market_data(fund, bench, years)
            macro = {} if args.no_ai else fetch_macro_snapshot()
    except DataLoadError as exc:
        raise StepFailed(str(exc)) from exc
    for note in market.warnings:
        print(f"      {_c('33', 'note:')} {note}")

    with tempfile.TemporaryDirectory(prefix="tearsheet_") as tmp:
        with step(2, "Computing risk/return analytics and charts"):
            metrics = compute_metrics(market.returns)
            charts = generate_charts(market.returns, market.fund_ticker, market.benchmark_ticker, Path(tmp))

        period = metrics["period"]
        context = {
            "fund_ticker": market.fund_ticker, "fund_name": market.fund_name, "fund_type": market.fund_type,
            "benchmark_ticker": market.benchmark_ticker, "benchmark_name": market.benchmark_name,
            "window": f"{period['start']:%b %Y} to {period['end']:%b %Y}", "months": period["months"],
            "requested_years": years, "rf_source": market.rf_source,
        }
        with step(3, "Generating executive due diligence commentary"):
            commentary = (template_commentary(context, metrics, "AI disabled with --no-ai") if args.no_ai
                          else generate_commentary(context, metrics, macro))
        if commentary.source.startswith("Template"):
            print(f"      {_c('33', 'note:')} {commentary.source}")

        with step(4, "Rendering PDF tear-sheet"):
            pdf = build_tearsheet(market, metrics, charts, commentary, args.output_dir)
    return pdf


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # keep progress and errors in order when piped
    print(_c("1;34", "\nManager Performance & Due Diligence Tear-Sheet\n"))
    try:
        pdf = run(args)
    except StepFailed as exc:
        print(f"\n{_c('31', 'Error:')} {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # last-resort guard so users see a clean message
        print(f"\n{_c('31', 'Unexpected error:')} {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"\n{_c('32', 'Tear-sheet saved:')} {pdf}")
    if not args.no_open:
        open_file(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
