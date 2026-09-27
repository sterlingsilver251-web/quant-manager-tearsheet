"""Module 4: ReportLab PDF tear-sheet generator (two pages, US Letter)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, KeepInFrame, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from ai_commentary import CommentaryResult

NAVY = colors.HexColor("#1B365D")
SLATE = colors.HexColor("#4A5568")
LIGHT_SLATE = colors.HexColor("#A0AEC0")
BORDER = colors.HexColor("#CBD5E0")
PANEL = colors.HexColor("#F7FAFC")
GREEN = colors.HexColor("#2F855A")
RED = colors.HexColor("#C53030")

PAGE_W, PAGE_H = letter
MARGIN = 0.5 * inch
HEADER_H = 0.95 * inch
FOOTER_H = 0.45 * inch
CONTENT_W = PAGE_W - 2 * MARGIN
CONTENT_H = PAGE_H - HEADER_H - FOOTER_H - 2 * MARGIN + 0.22 * inch


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pct(x, digits=1, signed=False):
    if x is None or x != x:
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def num(x, digits=2):
    return "n/a" if x is None or x != x else f"{x:.{digits}f}"


def tone(x, good_if_positive=True):
    if x is None or x != x or x == 0:
        return SLATE
    return GREEN if (x > 0) == good_if_positive else RED


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _styles():
    base = dict(fontName="Helvetica", textColor=SLATE, alignment=TA_LEFT)
    return {
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                                  textColor=NAVY, spaceBefore=2, spaceAfter=4),
        "headline": ParagraphStyle("headline", fontName="Helvetica-Oblique", fontSize=10, leading=13.5,
                                   textColor=NAVY),
        "body": ParagraphStyle("body", fontSize=8.9, leading=12.4, **base),
        "sub": ParagraphStyle("sub", fontName="Helvetica-Bold", fontSize=9.4, leading=12,
                              textColor=NAVY, spaceAfter=3),
        "card_label": ParagraphStyle("card_label", fontName="Helvetica-Bold", fontSize=6.8, leading=8,
                                     textColor=SLATE),
        "card_value": ParagraphStyle("card_value", fontName="Helvetica-Bold", fontSize=16, leading=19,
                                     textColor=NAVY),
        "card_sub": ParagraphStyle("card_sub", fontSize=6.8, leading=8.5, **base),
        "cell": ParagraphStyle("cell", fontSize=7.8, leading=9.5, **base),
        "small": ParagraphStyle("small", fontSize=6.6, leading=8.4, textColor=LIGHT_SLATE, fontName="Helvetica"),
    }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _metric_card(st, label, value, subline, value_color=NAVY):
    value_style = ParagraphStyle("v", parent=st["card_value"], textColor=value_color)
    return [Paragraph(label.upper(), st["card_label"]),
            Paragraph(value, value_style),
            Paragraph(subline, st["card_sub"])]


def _cards_table(st, cards, cols=4):
    rows = [cards[i:i + cols] for i in range(0, len(cards), cols)]
    gap = 0.08 * inch
    col_w = (CONTENT_W - gap * (cols - 1)) / cols
    # Interleave spacer columns so each card has its own border.
    data, widths = [], []
    for c in range(cols):
        widths += [col_w] + ([gap] if c < cols - 1 else [])
    for row in rows:
        cells = []
        for c, card in enumerate(row):
            cells += [card] + ([""] if c < cols - 1 else [])
        data.append(cells)
    table = Table(data, colWidths=widths, rowHeights=0.78 * inch)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7)]
    for r in range(len(rows)):
        for c in range(0, len(widths), 2):
            style += [("BOX", (c, r), (c, r), 0.6, BORDER),
                      ("BACKGROUND", (c, r), (c, r), PANEL),
                      ("LINEBEFORE", (c, r), (c, r), 2.5, NAVY)]
        if r < len(rows) - 1:
            style.append(("BOTTOMPADDING", (0, r), (-1, r), 10))
    table.setStyle(TableStyle(style))
    return table


def _kv_table(st, title, rows, width):
    head = [Paragraph(f"<b>{escape(title)}</b>", ParagraphStyle("h", parent=st["cell"], textColor=colors.white)),
            Paragraph("<b>Fund</b>", ParagraphStyle("h2", parent=st["cell"], textColor=colors.white, alignment=2)),
            Paragraph("<b>Bench</b>", ParagraphStyle("h3", parent=st["cell"], textColor=colors.white, alignment=2))]
    right = ParagraphStyle("r", parent=st["cell"], alignment=2)
    data = [head] + [[Paragraph(escape(k), st["cell"]), Paragraph(f, right), Paragraph(b, right)]
                     for k, f, b in rows]
    t = Table(data, colWidths=[width * 0.48, width * 0.30, width * 0.22])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, BORDER),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _scaled_image(path, max_w, max_h):
    iw, ih = ImageReader(str(path)).getSize()
    scale = min(max_w / iw, max_h / ih)
    return Image(str(path), width=iw * scale, height=ih * scale)


def _divider():
    return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=6, spaceAfter=6)


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

def _make_page_decorator(info: dict):
    def draw(canvas, doc):
        canvas.saveState()
        # Header band
        top = PAGE_H - MARGIN + 0.25 * inch
        canvas.setFillColor(NAVY)
        canvas.rect(0, top - HEADER_H, PAGE_W, HEADER_H + (PAGE_H - top), stroke=0, fill=1)
        canvas.setFillColor(LIGHT_SLATE)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(MARGIN, top - 0.28 * inch, "MANAGER PERFORMANCE & DUE DILIGENCE TEAR-SHEET")
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 22)
        canvas.drawString(MARGIN, top - 0.62 * inch, info["fund_ticker"])
        tw = canvas.stringWidth(info["fund_ticker"], "Helvetica-Bold", 22)
        canvas.setFont("Helvetica", 10.5)
        name = info["fund_name"]
        max_name_w = CONTENT_W * 0.58 - tw
        if canvas.stringWidth(name, "Helvetica", 10.5) > max_name_w:
            while name and canvas.stringWidth(name + "…", "Helvetica", 10.5) > max_name_w:
                name = name[:-1]
            name = name.rstrip() + "…"
        canvas.drawString(MARGIN + tw + 10, top - 0.60 * inch, name)

        right_x = PAGE_W - MARGIN
        lines = [("BENCHMARK", info["benchmark_ticker"]),
                 ("TIME HORIZON", info["horizon"]),
                 ("PERIOD", info["window"]),
                 ("REPORT DATE", info["report_date"])]
        y = top - 0.24 * inch
        label_x = right_x - 2.35 * inch
        for label, value in lines:
            canvas.setFont("Helvetica-Bold", 6.5)
            canvas.setFillColor(LIGHT_SLATE)
            canvas.drawString(label_x, y, label)
            canvas.setFont("Helvetica-Bold", 8.5)
            canvas.setFillColor(colors.white)
            canvas.drawRightString(right_x, y, value)
            y -= 0.165 * inch

        # Footer
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, MARGIN + 0.12 * inch, PAGE_W - MARGIN, MARGIN + 0.12 * inch)
        canvas.setFont("Helvetica", 6.3)
        canvas.setFillColor(SLATE)
        canvas.drawString(MARGIN, MARGIN - 0.02 * inch,
                          f"Data: Yahoo Finance adjusted prices; {info['rf_source']}. Commentary: {info['commentary_source']}.")
        canvas.drawString(MARGIN, MARGIN - 0.14 * inch,
                          "For institutional discussion only. Not investment advice. Past performance does not guarantee future results.")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 0.02 * inch, f"Page {doc.page} of 2")
        canvas.restoreState()
    return draw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_tearsheet(market, metrics: dict, charts: dict[str, Path], commentary: CommentaryResult,
                    output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = "".join(ch for ch in market.fund_ticker if ch.isalnum() or ch in "-_") or "Fund"
    out_path = output_dir / f"TearSheet_{safe_ticker}_{stamp}.pdf"

    st = _styles()
    perf, risk, radj, exp, period = (metrics[k] for k in
                                     ("performance", "risk", "risk_adjusted", "market_exposure", "period"))
    dd, bdd = risk["drawdown"], risk["benchmark_drawdown"]
    bench = market.benchmark_ticker
    window = f"{period['start']:%b %Y} – {period['end']:%b %Y}"

    info = {
        "fund_ticker": market.fund_ticker,
        "fund_name": market.fund_name,
        "benchmark_ticker": bench,
        "horizon": f"{period['years']:g} years",
        "window": window,
        "report_date": datetime.now().strftime("%b %d, %Y"),
        "rf_source": market.rf_source,
        "commentary_source": commentary.source,
    }

    # ---- Page 1 ------------------------------------------------------------
    p1 = []
    role = escape(commentary.recommended_role)
    headline_tbl = Table(
        [[Paragraph(escape(commentary.headline), st["headline"]),
          Paragraph(f"<font size=6.5 color='#A0AEC0'><b>RECOMMENDED ROLE</b></font><br/>"
                    f"<font size=10 color='#1B365D'><b>{role}</b></font>", st["cell"])]],
        colWidths=[CONTENT_W * 0.72, CONTENT_W * 0.28])
    headline_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBEFORE", (1, 0), (1, 0), 0.6, BORDER),
        ("LEFTPADDING", (0, 0), (0, 0), 0), ("LEFTPADDING", (1, 0), (1, 0), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    p1 += [headline_tbl, _divider()]

    recovery_text = (f"recovered in {dd['peak_to_recovery_months']} mo" if dd["recovered"]
                     else "not yet recovered")
    cards = [
        _metric_card(st, "Annualized Return (CAGR)", pct(perf["fund_cagr"]),
                     f"{bench}: {pct(perf['benchmark_cagr'])}"),
        _metric_card(st, "Excess Return (Ann.)", pct(perf["excess_return_annualized"], signed=True),
                     f"Cumulative {pct(perf['fund_cumulative_return'])} vs {pct(perf['benchmark_cumulative_return'])}",
                     tone(perf["excess_return_annualized"])),
        _metric_card(st, "Sharpe Ratio", num(radj["fund_sharpe"]),
                     f"{bench}: {num(radj['benchmark_sharpe'])}  ·  Sortino {num(radj['fund_sortino'])}"),
        _metric_card(st, "Alpha (Ann.)", pct(exp["alpha_annualized"], signed=True),
                     f"t-stat {num(exp['alpha_t_stat'])}  ·  IR {num(radj['information_ratio'])}",
                     tone(exp["alpha_annualized"])),
        _metric_card(st, "Beta", num(exp["beta"]),
                     f"R² {num(exp['r_squared'])}  ·  TE {pct(exp['tracking_error'])}"),
        _metric_card(st, "Maximum Drawdown", pct(risk["fund_max_drawdown"]),
                     f"{bench}: {pct(risk['benchmark_max_drawdown'])}  ·  {recovery_text}", RED),
        _metric_card(st, "Up-Market Capture", pct(exp["up_capture"], 0),
                     "Share of benchmark gains captured", tone((exp["up_capture"] or 1) - 1)),
        _metric_card(st, "Down-Market Capture", pct(exp["down_capture"], 0),
                     "Share of benchmark losses captured", tone((exp["down_capture"] or 1) - 1, good_if_positive=False)),
    ]
    p1 += [_cards_table(st, cards), Spacer(1, 10)]

    half = (CONTENT_W - 0.15 * inch) / 2
    charts_tbl = Table([[_scaled_image(charts["growth"], half - 0.1 * inch, 2.9 * inch),
                         _scaled_image(charts["drawdown"], half - 0.1 * inch, 2.9 * inch)]],
                       colWidths=[half + 0.075 * inch, half + 0.075 * inch])
    charts_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                    ("BOX", (0, 0), (0, 0), 0.6, BORDER), ("BOX", (1, 0), (1, 0), 0.6, BORDER),
                                    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    p1 += [charts_tbl, Spacer(1, 10)]

    def dd_date(d, key):
        return "—" if d[key] is None else f"{d[key]:%b %Y}"

    left_rows = [
        ("Cumulative Return", pct(perf["fund_cumulative_return"]), pct(perf["benchmark_cumulative_return"])),
        ("Annualized Return (CAGR)", pct(perf["fund_cagr"]), pct(perf["benchmark_cagr"])),
        ("Annualized Volatility", pct(risk["fund_volatility"]), pct(risk["benchmark_volatility"])),
        ("Sharpe Ratio", num(radj["fund_sharpe"]), num(radj["benchmark_sharpe"])),
        ("Sortino Ratio", num(radj["fund_sortino"]), num(radj["benchmark_sortino"])),
        ("Maximum Drawdown", pct(risk["fund_max_drawdown"]), pct(risk["benchmark_max_drawdown"])),
        ("Max Drawdown Peak Date", dd_date(dd, "peak_date"), dd_date(bdd, "peak_date")),
        ("Max Drawdown Trough Date", dd_date(dd, "trough_date"), dd_date(bdd, "trough_date")),
        ("Peak-to-Trough (months)", str(dd["peak_to_trough_months"]), str(bdd["peak_to_trough_months"])),
        ("Trough-to-Recovery (months)",
         str(dd["recovery_months"]) if dd["recovered"] else "Not recovered",
         str(bdd["recovery_months"]) if bdd["recovered"] else "Not recovered"),
    ]
    right_rows = [
        ("Alpha (annualized, CAPM)", pct(exp["alpha_annualized"], signed=True), "—"),
        ("Beta", num(exp["beta"]), "1.00"),
        ("R-Squared / Correlation", f"{num(exp['r_squared'])} / {num(exp['correlation'])}", "—"),
        ("Tracking Error", pct(exp["tracking_error"]), "—"),
        ("Information Ratio", num(radj["information_ratio"]), "—"),
        ("Up / Down Capture", f"{pct(exp['up_capture'], 0)} / {pct(exp['down_capture'], 0)}", "100% / 100%"),
        ("Best / Worst Month", f"{pct(perf['best_month'])} / {pct(perf['worst_month'])}", "—"),
        ("% Positive Months", pct(perf["pct_positive_months"], 0), "—"),
        ("Batting Avg. (vs. Benchmark)", pct(perf["batting_average"], 0), "—"),
    ]
    tw = (CONTENT_W - 0.15 * inch) / 2
    tables = Table([[_kv_table(st, "Return & Risk", left_rows, tw),
                     _kv_table(st, "Market Exposure", right_rows, tw)]],
                   colWidths=[tw + 0.075 * inch, tw + 0.075 * inch])
    tables.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    p1.append(tables)

    notes = list(market.warnings)
    if notes:
        p1 += [Spacer(1, 6), Paragraph("Data notes: " + escape(" ".join(notes)), st["small"])]

    # ---- Page 2 ------------------------------------------------------------
    p2 = [_scaled_image(charts["heatmap"], CONTENT_W, 2.9 * inch), _divider(),
          Paragraph("Executive Due Diligence Commentary", st["section"])]
    for i, (title, text) in enumerate(commentary.sections):
        p2 += [Paragraph(escape(title), st["sub"]), Paragraph(escape(text), st["body"])]
        if i < len(commentary.sections) - 1:
            p2.append(_divider())
    p2 += [Spacer(1, 8), Paragraph(
        "Methodology: month-end adjusted-close returns (distributions reinvested). Annualization uses 12 periods. "
        "Sharpe and Sortino use the monthly T-bill rate as the hurdle; Sortino uses downside deviation of excess returns. "
        "Alpha and beta come from an OLS regression of fund excess returns on benchmark excess returns; alpha is monthly "
        "intercept x 12. Capture ratios compare annualized geometric returns in benchmark up and down months. "
        "The final heatmap column compounds the months shown for each year (partial years are not annual returns).",
        st["small"])]

    # ---- Assemble ----------------------------------------------------------
    frame = Frame(MARGIN, MARGIN + FOOTER_H - 0.2 * inch, CONTENT_W, CONTENT_H,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="body")
    doc = BaseDocTemplate(str(out_path), pagesize=letter,
                          leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
                          title=f"{market.fund_ticker} Due Diligence Tear-Sheet",
                          author="Manager Tear-Sheet Generator")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_make_page_decorator(info))])

    # KeepInFrame(shrink) guarantees each page's content fits, so the report is always 2 pages.
    story = [KeepInFrame(CONTENT_W, CONTENT_H, p1, mode="shrink"), PageBreak(),
             KeepInFrame(CONTENT_W, CONTENT_H, p2, mode="shrink")]
    doc.build(story)
    return out_path
