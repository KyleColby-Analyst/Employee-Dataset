"""Render the workforce dashboard as a single PNG.

Each panel is a standalone function that draws onto a given matplotlib Axes,
so panels can be tested, reordered, or reused independently. ``build_figure``
composes them into the final layout.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from config import ColorPalette, DashboardConfig
from insights import generate_insights

logger = logging.getLogger(__name__)


def draw_header(ax: Axes, snapshot: pd.Timestamp, palette: ColorPalette) -> None:
    """Draw the dashboard title block.

    Args:
        ax: Axes to draw into (spans the full figure width).
        snapshot: The dataset's "as of" date, shown in the subtitle.
        palette: Color scheme.
    """
    ax.axis("off")
    ax.text(0, 0.82, "Ascendant Aerospace Corp. — Workforce & Hiring Plan Analysis",
             fontsize=27, fontweight="bold", color=palette.navy, va="top")
    ax.text(
        0, 0.28,
        f"Snapshot as of {snapshot:%B %-d, %Y}  |  Growth, attrition, staffing-gap risk, "
        "and the current requisition plan against workforce history",
        fontsize=13.5, color=palette.slate, va="top",
    )


def draw_kpi_strip(ax: Axes, kpis: dict[str, float], palette: ColorPalette) -> None:
    """Draw the row of headline KPI cards.

    Args:
        ax: Axes to draw into (spans the full figure width).
        kpis: Company-wide KPI dict from ``compute_company_kpis``.
        palette: Color scheme.
    """
    ax.axis("off")
    cards = [
        ("ACTIVE HEADCOUNT", f"{kpis['active_now']:,.0f}", f"{kpis['yoy_growth_pct']:+.1f}% YoY", palette.teal),
        ("HIRES (TTM)", f"{kpis['hires_ttm']:,.0f}", f"{kpis['hires_ttm']/12:.0f}/month avg", palette.navy),
        ("DEPARTURES (TTM)", f"{kpis['terms_ttm']:,.0f}", f"{kpis['attrition_rate_pct']:.1f}% attrition rate", palette.coral),
        ("AVG TENURE (ACTIVE)", f"{kpis['avg_tenure_active_years']:.1f} yrs", "company-wide", palette.slate),
        ("CURRENT REQ PLAN", f"{kpis['planned_total']:,.0f}", "open roles across 3 channels", palette.amber),
    ]
    box_w = 1.0 / len(cards)
    for i, (label, value, sub, color) in enumerate(cards):
        x0 = i * box_w
        pad = 0.012
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 + pad, 0.02), box_w - 2 * pad, 0.9,
            boxstyle="round,pad=0.01,rounding_size=0.02", transform=ax.transAxes,
            facecolor=palette.light_bg, edgecolor=color, linewidth=1.8,
        ))
        cx = x0 + box_w / 2
        ax.text(cx, 0.68, value, fontsize=21, fontweight="bold", color=color, ha="center", va="center", transform=ax.transAxes)
        ax.text(cx, 0.34, label, fontsize=9.5, fontweight="bold", color=palette.slate, ha="center", va="center", transform=ax.transAxes)
        ax.text(cx, 0.13, sub, fontsize=9, color=palette.slate, ha="center", va="center", transform=ax.transAxes, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_headcount_trend(ax: Axes, headcount_trend: pd.DataFrame, palette: ColorPalette) -> None:
    """Draw the company-wide active-headcount trend line.

    Args:
        ax: Axes to draw into (spans the full figure width).
        headcount_trend: Output of ``compute_headcount_trend``.
        palette: Color scheme.
    """
    ax.fill_between(headcount_trend["month"], headcount_trend["headcount"], color=palette.teal, alpha=0.15)
    ax.plot(headcount_trend["month"], headcount_trend["headcount"], color=palette.teal, linewidth=2.6)
    ax.set_title("Where We Have Grown — Company-Wide Active Headcount",
                 fontsize=15, fontweight="bold", color=palette.navy, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=palette.grid, linewidth=0.8)
    ax.set_ylabel("Active employees", fontsize=10.5, color=palette.slate)
    ax.tick_params(colors=palette.slate, labelsize=10)

    last = headcount_trend.iloc[-1]
    ax.annotate(
        f"{int(last['headcount']):,} today", xy=(last["month"], last["headcount"]),
        xytext=(-95, 12), textcoords="offset points", fontsize=10.5, fontweight="bold", color=palette.navy,
        arrowprops=dict(arrowstyle="-", color=palette.slate, lw=0.8),
    )


def draw_net_change_panel(ax: Axes, dept_metrics: pd.DataFrame, palette: ColorPalette) -> None:
    """Draw net headcount change by department (trailing window).

    Args:
        ax: Axes to draw into.
        dept_metrics: Department metrics table.
        palette: Color scheme.
    """
    data = dept_metrics.sort_values("net_change_ttm")
    colors = [palette.teal if v >= 0 else palette.coral for v in data["net_change_ttm"]]
    bars = ax.barh(data.index, data["net_change_ttm"], color=colors, height=0.65)
    ax.axvline(0, color=palette.navy, linewidth=0.9)
    ax.set_title("Net Headcount Change by Department", fontsize=13.5, fontweight="bold", color=palette.navy, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=palette.slate, labelsize=9.2)
    ax.grid(axis="x", color=palette.grid, linewidth=0.8)
    for bar, value in zip(bars, data["net_change_ttm"]):
        x = bar.get_width()
        ax.text(x + (1.2 if x >= 0 else -1.2), bar.get_y() + bar.get_height() / 2, f"{int(value):+d}",
                va="center", ha="left" if x >= 0 else "right", fontsize=8.5, color=palette.navy, fontweight="bold")
    ax.set_xlabel("Hires − departures, trailing window", fontsize=9.5, color=palette.slate)


def draw_attrition_panel(ax: Axes, dept_metrics: pd.DataFrame, company_attrition_pct: float, palette: ColorPalette) -> None:
    """Draw attrition rate by department, with a company-average reference line.

    Args:
        ax: Axes to draw into.
        dept_metrics: Department metrics table.
        company_attrition_pct: Company-wide attrition rate (0-100 scale).
        palette: Color scheme.
    """
    data = dept_metrics.sort_values("attrition_rate_ttm")
    threshold = company_attrition_pct / 100
    colors = [
        palette.coral if v >= threshold else palette.amber if v >= threshold * 0.6 else palette.teal
        for v in data["attrition_rate_ttm"]
    ]
    bars = ax.barh(data.index, data["attrition_rate_ttm"] * 100, color=colors, height=0.65)
    ax.axvline(company_attrition_pct, color=palette.navy, linewidth=1.1, linestyle="--")
    ax.text(company_attrition_pct, len(data) - 0.3, f"  company avg {company_attrition_pct:.1f}%",
            fontsize=8.3, color=palette.navy, fontweight="bold", va="top")
    ax.set_title("Where We Have Lost People — Attrition Rate by Dept", fontsize=13.5, fontweight="bold", color=palette.navy, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=palette.slate, labelsize=8.6)
    ax.grid(axis="x", color=palette.grid, linewidth=0.8)
    for bar, value in zip(bars, data["attrition_rate_ttm"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{value*100:.1f}%",
                va="center", ha="left", fontsize=8.5, color=palette.navy, fontweight="bold")
    ax.set_xlabel("Terminations ÷ avg headcount, trailing window", fontsize=9.5, color=palette.slate)
    ax.set_xlim(0, data["attrition_rate_ttm"].max() * 100 * 1.3)


def draw_risk_panel(ax: Axes, dept_metrics: pd.DataFrame, palette: ColorPalette) -> None:
    """Draw the composite staffing-gap risk ranking.

    Args:
        ax: Axes to draw into (spans the full figure width).
        dept_metrics: Department metrics table (must include risk_score).
        palette: Color scheme.
    """
    data = dept_metrics.sort_values("risk_score")
    colors = [palette.coral if v >= 55 else palette.amber if v >= 42 else palette.teal for v in data["risk_score"]]
    bars = ax.barh(data.index, data["risk_score"], color=colors, height=0.62)
    ax.set_title("Where We Are Likely to Experience Staffing Gaps — Composite Risk Score",
                 fontsize=14.5, fontweight="bold", color=palette.navy, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=palette.slate, labelsize=9.5)
    ax.grid(axis="x", color=palette.grid, linewidth=0.8)
    ax.set_xlabel(
        "Risk score (0-100): blends attrition rate, clearance dependency, avg tenure, "
        "planned-growth intensity, and leadership bench depth", fontsize=9, color=palette.slate,
    )
    for bar, value, dept in zip(bars, data["risk_score"], data.index):
        row = dept_metrics.loc[dept]
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}",
                va="center", ha="left", fontsize=9, color=palette.navy, fontweight="bold")
        detail = (f"{row['attrition_rate_ttm']*100:.0f}% attrition · "
                  f"{row['clearance_pct']*100:.0f}% need clearance · "
                  f"{row['avg_tenure_active']:.1f}y avg tenure")
        ax.text(1.0, bar.get_y() + bar.get_height() / 2, detail, va="center", ha="left",
                fontsize=7.4, color="white", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(handles=[
        mpatches.Patch(color=palette.coral, label="High risk (\u226555)"),
        mpatches.Patch(color=palette.amber, label="Moderate risk (42-55)"),
        mpatches.Patch(color=palette.teal, label="Lower risk (<42)"),
    ], loc="lower right", fontsize=9, frameon=False)


def draw_hiring_plan_panel(ax: Axes, dept_metrics: pd.DataFrame, palette: ColorPalette, top_n: int) -> None:
    """Draw planned hires vs. trailing-window departures for planned departments.

    Args:
        ax: Axes to draw into (spans the full figure width).
        dept_metrics: Department metrics table (must include planned_hires).
        palette: Color scheme.
        top_n: Max number of departments to show (largest planned_hires first).

    Raises:
        ValueError: If no department in ``dept_metrics`` has any planned hires.
    """
    data = dept_metrics[dept_metrics["planned_hires"] > 0].sort_values("planned_hires").tail(top_n)
    if data.empty:
        raise ValueError("No department has planned_hires > 0; nothing to plot in the hiring-plan panel.")

    y = np.arange(len(data))
    h = 0.36
    bars_planned = ax.barh(y + h / 2, data["planned_hires"], height=h, color=palette.navy, label="Planned hires (current req plan)")
    bars_lost = ax.barh(y - h / 2, data["terms_ttm"], height=h, color=palette.coral, alpha=0.85, label="Departures, trailing window")
    ax.set_yticks(y)
    ax.set_yticklabels(data.index, fontsize=9.8)
    ax.set_title("Hiring Plan vs. Historical Losses — Growing Capacity or Just Replacing It?",
                 fontsize=14, fontweight="bold", color=palette.navy, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=palette.slate, labelsize=9.5)
    ax.grid(axis="x", color=palette.grid, linewidth=0.8)
    ax.set_xlabel("Headcount", fontsize=9.5, color=palette.slate)
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    for bar_p, bar_l, planned, lost in zip(bars_planned, bars_lost, data["planned_hires"], data["terms_ttm"]):
        ax.text(bar_p.get_width() + 0.5, bar_p.get_y() + bar_p.get_height() / 2, f"{int(planned)}",
                va="center", fontsize=8.2, color=palette.navy, fontweight="bold")
        ax.text(bar_l.get_width() + 0.5, bar_l.get_y() + bar_l.get_height() / 2, f"{int(lost)}",
                va="center", fontsize=8.2, color=palette.coral, fontweight="bold")


def draw_takeaways_panel(ax: Axes, dept_metrics: pd.DataFrame, kpis: dict[str, float], palette: ColorPalette) -> None:
    """Draw the auto-generated narrative takeaways panel.

    Args:
        ax: Axes to draw into (spans the full figure width).
        dept_metrics: Fully computed department metrics table.
        kpis: Company-wide KPI dict.
        palette: Color scheme.
    """
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.015", transform=ax.transAxes,
        facecolor=palette.light_bg, edgecolor=palette.grid, linewidth=1,
    ))
    ax.text(0.018, 0.93, "What the History Says About This Hiring Plan",
            fontsize=15, fontweight="bold", color=palette.navy, va="top")

    insights = generate_insights(dept_metrics, kpis)
    y0 = 0.80
    for i, (headline, body) in enumerate(insights):
        yy = y0 - i * (0.78 / max(len(insights), 1))
        ax.text(0.022, yy, f"{i + 1}.", fontsize=12.5, fontweight="bold", color=palette.teal, va="top")
        ax.text(0.052, yy, headline, fontsize=12, fontweight="bold", color=palette.navy, va="top", wrap=True)
        ax.text(0.052, yy - 0.052, body, fontsize=10.4, color=palette.slate, va="top", wrap=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def build_figure(pipeline_result: dict[str, object], config: DashboardConfig) -> plt.Figure:
    """Assemble every panel into the final dashboard figure.

    Args:
        pipeline_result: Output of ``data_pipeline.run_pipeline``.
        config: Dashboard configuration (sizing, DPI, palette).

    Returns:
        The composed matplotlib Figure, not yet saved.
    """
    palette = config.palette
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["text.color"] = palette.navy
    plt.rcParams["axes.edgecolor"] = palette.grid

    fig = plt.figure(figsize=(config.fig_width_in, config.fig_height_in), facecolor="white", dpi=config.dpi)
    gs = GridSpec(
        7, 2, figure=fig,
        height_ratios=[0.9, 0.55, 2.6, 3.6, 3.6, 4.6, 3.4],
        hspace=0.55, wspace=0.65,
        left=0.055, right=0.965, top=0.965, bottom=0.02,
    )

    draw_header(fig.add_subplot(gs[0, :]), pipeline_result["snapshot"], palette)
    draw_kpi_strip(fig.add_subplot(gs[1, :]), pipeline_result["kpis"], palette)
    draw_headcount_trend(fig.add_subplot(gs[2, :]), pipeline_result["headcount_trend"], palette)
    draw_net_change_panel(fig.add_subplot(gs[3, 0]), pipeline_result["dept_metrics"], palette)
    draw_attrition_panel(fig.add_subplot(gs[3, 1]), pipeline_result["dept_metrics"],
                          pipeline_result["kpis"]["attrition_rate_pct"], palette)
    draw_risk_panel(fig.add_subplot(gs[4, :]), pipeline_result["dept_metrics"], palette)
    draw_hiring_plan_panel(fig.add_subplot(gs[5, :]), pipeline_result["dept_metrics"], palette,
                            config.top_n_plan_departments)
    draw_takeaways_panel(fig.add_subplot(gs[6, :]), pipeline_result["dept_metrics"], pipeline_result["kpis"], palette)

    fig.text(0.055, 0.006,
              "Source: internal HRIS export  •  Hiring-plan roles mapped to nearest department by function (see hiring_plan.py)",
              fontsize=8.3, color=palette.slate, style="italic")
    return fig


def save_dashboard(fig: plt.Figure, output_path, dpi: int) -> None:
    """Save the figure to disk, creating parent directories as needed.

    Args:
        fig: Figure to save.
        output_path: Destination path.
        dpi: Output resolution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dashboard to %s", output_path)
