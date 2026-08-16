"""Generate the dashboard's narrative takeaways directly from computed metrics.

Every bullet is derived from ``dept_metrics``/``plan_df`` at render time --
nothing here is a hardcoded reference to a specific department or number --
so re-running the pipeline against a refreshed dataset or a revised hiring
plan produces takeaways that reflect the new data.
"""

from __future__ import annotations

import pandas as pd


def _replacement_gap_insight(dept_metrics: pd.DataFrame) -> tuple[str, str] | None:
    """Flag the department where historical losses most outstrip the hiring plan.

    Args:
        dept_metrics: Department metrics table (post plan-merge).

    Returns:
        (headline, body) tuple, or None if no department has any departures.
    """
    candidates = dept_metrics[dept_metrics["terms_ttm"] > 0].copy()
    if candidates.empty:
        return None
    candidates["replacement_gap"] = candidates["terms_ttm"] - candidates["planned_hires"]
    worst = candidates.sort_values("replacement_gap", ascending=False).iloc[0]
    dept = worst.name
    headline = f"{dept} is losing people faster than the plan replaces them."
    body = (
        f"It lost {int(worst['terms_ttm'])} people in the trailing period "
        f"({worst['attrition_rate_ttm']*100:.0f}% attrition) while the current plan allocates only "
        f"{int(worst['planned_hires'])} role(s) there — a net gap of {int(worst['replacement_gap'])} "
        "positions before any growth is accounted for."
    )
    return headline, body


def _biggest_growth_bet_insight(dept_metrics: pd.DataFrame) -> tuple[str, str] | None:
    """Flag the department with the largest planned headcount increase (%).

    Args:
        dept_metrics: Department metrics table (post plan-merge).

    Returns:
        (headline, body) tuple, or None if no department has any planned hires.
    """
    planned = dept_metrics[dept_metrics["planned_hires"] > 0]
    if planned.empty:
        return None
    biggest = planned.sort_values("planned_growth_pct", ascending=False).iloc[0]
    dept = biggest.name
    others_max = planned.drop(index=dept)["planned_growth_pct"].max() if len(planned) > 1 else 0.0
    multiple = biggest["planned_growth_pct"] / others_max if others_max > 0 else float("inf")
    headline = f"The {dept} request is the single largest bet in this plan."
    body = (
        f"It asks for {int(biggest['planned_hires'])} roles — {biggest['planned_growth_pct']:.0f}% headcount "
        f"growth in one department, against a current attrition rate of {biggest['attrition_rate_ttm']*100:.0f}% "
        f"and {biggest['clearance_pct']*100:.0f}% clearance dependency."
    )
    return headline, body


def _unaddressed_risk_insight(dept_metrics: pd.DataFrame) -> tuple[str, str] | None:
    """Flag the highest-risk department that has zero planned hires.

    Args:
        dept_metrics: Department metrics table (post risk-score merge).

    Returns:
        (headline, body) tuple, or None if every department has at least
        one planned hire.
    """
    unaddressed = dept_metrics[dept_metrics["planned_hires"] == 0]
    if unaddressed.empty:
        return None
    riskiest = unaddressed.sort_values("risk_score", ascending=False).iloc[0]
    dept = riskiest.name
    headline = f"{dept} is the highest-risk department not covered by this plan."
    body = (
        f"Risk score {riskiest['risk_score']:.0f}/100, driven by "
        f"{riskiest['avg_tenure_active']:.1f}-year average tenure, "
        f"{riskiest['attrition_rate_ttm']*100:.0f}% attrition, and only "
        f"{int(riskiest['n_managers_active'])} active manager(s) to lead it — with nothing budgeted to backfill it."
    )
    return headline, body


def _clearance_lead_time_insight(dept_metrics: pd.DataFrame, clearance_threshold: float = 0.45) -> tuple[str, str] | None:
    """Flag planned departments with high clearance dependency (longer lead times).

    Args:
        dept_metrics: Department metrics table (post plan-merge).
        clearance_threshold: Minimum clearance-need fraction to flag.

    Returns:
        (headline, body) tuple, or None if no planned department clears the threshold.
    """
    planned = dept_metrics[dept_metrics["planned_hires"] > 0]
    flagged = planned[planned["clearance_pct"] >= clearance_threshold].sort_values(
        "clearance_pct", ascending=False
    )
    if flagged.empty:
        return None
    names = ", ".join(f"{d} ({row['clearance_pct']*100:.0f}%)" for d, row in flagged.head(3).iterrows())
    headline = "Several planned departments carry longer hiring lead times than headcount alone suggests."
    body = (
        f"{names} all require security clearance for a large share of roles — clearance processing, "
        "not sourcing, is historically the bottleneck for these reqs and they should be opened early."
    )
    return headline, body


def _growth_concentration_insight(dept_metrics: pd.DataFrame, kpis: dict[str, float], top_n: int = 3) -> tuple[str, str] | None:
    """Summarize how concentrated company-wide growth is across departments.

    Args:
        dept_metrics: Department metrics table.
        kpis: Company-wide KPI dict from ``compute_company_kpis``.
        top_n: Number of departments to consider "the top" contributors.

    Returns:
        (headline, body) tuple, or None if there's no positive net growth.
    """
    growing = dept_metrics[dept_metrics["net_change_ttm"] > 0].sort_values(
        "net_change_ttm", ascending=False
    )
    total_growth = growing["net_change_ttm"].sum()
    if total_growth <= 0:
        return None
    top = growing.head(top_n)
    share = top["net_change_ttm"].sum() / total_growth * 100
    names = ", ".join(top.index)
    headline = f"Company-wide growth (+{kpis['yoy_growth_pct']:.1f}% YoY) is healthy but concentrated."
    body = (
        f"{names} account for {share:.0f}% of net headcount gains in the trailing period; "
        "smaller departments outside this list are comparatively stable, not attrition risks."
    )
    return headline, body


def generate_insights(
    dept_metrics: pd.DataFrame, kpis: dict[str, float], max_insights: int = 5
) -> list[tuple[str, str]]:
    """Assemble the dashboard's data-driven takeaway bullets.

    Args:
        dept_metrics: Fully computed department metrics table (after
            ``merge_hiring_plan`` and ``compute_risk_scores``).
        kpis: Company-wide KPI dict from ``compute_company_kpis``.
        max_insights: Maximum number of bullets to return.

    Returns:
        List of up to ``max_insights`` (headline, body) tuples. Insight
        generators that find nothing notable (e.g. no unaddressed risk)
        are skipped rather than padded with filler.
    """
    generators = [
        _replacement_gap_insight,
        _biggest_growth_bet_insight,
        _unaddressed_risk_insight,
        _clearance_lead_time_insight,
        lambda dm: _growth_concentration_insight(dm, kpis),
    ]
    insights: list[tuple[str, str]] = []
    for gen in generators:
        if len(insights) >= max_insights:
            break
        result = gen(dept_metrics)
        if result is not None:
            insights.append(result)
    return insights
