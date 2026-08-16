"""Load the employee dataset and compute every metric the dashboard needs.

All per-department aggregation here uses ``groupby`` on boolean/numeric
helper columns rather than iterating over departments or rows in Python --
the helper columns (e.g. ``is_active_now``) are themselves built with
vectorized comparisons, so no ``.iterrows()`` or per-row Python loop is used
anywhere in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import DashboardConfig
from hiring_plan import build_plan_dataframe

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "employee_id", "department", "job_level", "hire_date", "termination_date",
    "employment_status", "termination_reason", "clearance_level", "tenure_years",
)


def load_employee_data(csv_path: Path) -> pd.DataFrame:
    """Load and type-parse the employee CSV.

    Args:
        csv_path: Path to the employee dataset CSV.

    Returns:
        DataFrame with ``hire_date``/``termination_date`` parsed as
        datetime64.

    Raises:
        FileNotFoundError: If ``csv_path`` doesn't exist.
        pd.errors.EmptyDataError: If the file exists but contains no data.
        pd.errors.ParserError: If the file isn't valid CSV.
        ValueError: If any column in ``REQUIRED_COLUMNS`` is missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Employee dataset not found at {csv_path}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["hire_date", "termination_date"])
    except pd.errors.EmptyDataError:
        raise
    except pd.errors.ParserError:
        raise

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Employee dataset is missing required columns: {sorted(missing_cols)}")

    logger.info("Loaded %d employee records from %s", len(df), csv_path)
    return df


def validate_employee_data(df: pd.DataFrame) -> None:
    """Run defensive checks on the loaded employee data before analysis.

    Args:
        df: Employee DataFrame as returned by ``load_employee_data``.

    Raises:
        ValueError: If duplicate employee IDs exist, if any termination
            date precedes its hire date, or if the dataset is empty.
    """
    if df.empty:
        raise ValueError("Employee dataset has zero rows; nothing to analyze.")

    dup_ids = df["employee_id"].duplicated().sum()
    if dup_ids:
        raise ValueError(f"Found {dup_ids} duplicate employee_id values in the dataset")

    terminated = df[df["employment_status"] == "Terminated"]
    bad_dates = (terminated["termination_date"] < terminated["hire_date"]).sum()
    if bad_dates:
        raise ValueError(f"Found {bad_dates} records where termination_date precedes hire_date")

    logger.info("Validation passed: %d rows, no duplicate IDs, no impossible date orderings.", len(df))


def determine_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Infer the "as of" date the dataset represents.

    Uses the latest hire date in the data, which matches how the dataset
    generator defines its snapshot (new hires are generated right up to the
    snapshot date, so the max hire date is a reliable proxy).

    Args:
        df: Employee DataFrame.

    Returns:
        The inferred snapshot timestamp.
    """
    snapshot = df["hire_date"].max()
    logger.info("Inferred snapshot date: %s", snapshot.date())
    return snapshot


def compute_headcount_trend(
    df: pd.DataFrame, snapshot: pd.Timestamp, start_date: pd.Timestamp
) -> pd.DataFrame:
    """Compute company-wide active headcount at each month-end in the window.

    Vectorized via broadcasting: builds an (n_employees x n_months) boolean
    "was this employee active in this month" matrix in one comparison pass,
    then sums down the employee axis. No per-month or per-row Python loop.

    Args:
        df: Employee DataFrame.
        snapshot: Latest month to include.
        start_date: Earliest month to include.

    Returns:
        DataFrame with columns ``month`` and ``headcount``.
    """
    months = pd.date_range(start_date, snapshot, freq="MS")
    hire = df["hire_date"].to_numpy()[:, None]
    term = df["termination_date"].to_numpy()[:, None]
    month_arr = months.to_numpy()[None, :]

    was_hired = hire <= month_arr
    not_yet_terminated = pd.isna(df["termination_date"]).to_numpy()[:, None] | (term > month_arr)
    active_matrix = was_hired & not_yet_terminated

    return pd.DataFrame({"month": months, "headcount": active_matrix.sum(axis=0)})


def compute_department_metrics(
    df: pd.DataFrame, snapshot: pd.Timestamp, ttm_months: int
) -> pd.DataFrame:
    """Compute per-department headcount, attrition, and risk-input metrics.

    Args:
        df: Employee DataFrame.
        snapshot: The "as of" date.
        ttm_months: Length of the trailing window, in months.

    Returns:
        DataFrame indexed by department with columns: active_now,
        active_ttm_ago, hires_ttm, terms_ttm, vol_terms_ttm, net_change_ttm,
        attrition_rate_ttm, avg_tenure_active, pct_senior_active,
        clearance_pct, avg_engagement_active, n_managers_active.
    """
    ttm_start = snapshot - pd.DateOffset(months=ttm_months)
    work = df.copy()

    is_active_now = work["employment_status"] == "Active"
    is_active_ttm_ago = (work["hire_date"] <= ttm_start) & (
        work["termination_date"].isna() | (work["termination_date"] > ttm_start)
    )
    is_hire_ttm = work["hire_date"] > ttm_start
    is_term_ttm = work["termination_date"].between(ttm_start, snapshot, inclusive="right")
    is_vol_term_ttm = is_term_ttm & work["termination_reason"].str.startswith("Voluntary", na=False)
    is_senior_active = is_active_now & (work["job_level"] >= 7)
    needs_clearance = work["clearance_level"] != "No Clearance"

    work["_active_now"] = is_active_now
    work["_active_ttm_ago"] = is_active_ttm_ago
    work["_hire_ttm"] = is_hire_ttm
    work["_term_ttm"] = is_term_ttm
    work["_vol_term_ttm"] = is_vol_term_ttm
    work["_senior_active"] = is_senior_active
    work["_needs_clearance"] = needs_clearance
    # NaN-when-inactive columns so groupby(...).mean() averages only active employees
    work["_tenure_if_active"] = np.where(is_active_now, work["tenure_years"], np.nan)
    work["_engagement_if_active"] = np.where(is_active_now, work.get("engagement_score", np.nan), np.nan)

    grouped = work.groupby("department", observed=True)
    metrics = grouped.agg(
        active_now=("_active_now", "sum"),
        active_ttm_ago=("_active_ttm_ago", "sum"),
        hires_ttm=("_hire_ttm", "sum"),
        terms_ttm=("_term_ttm", "sum"),
        vol_terms_ttm=("_vol_term_ttm", "sum"),
        n_managers_active=("_senior_active", "sum"),
        clearance_pct=("_needs_clearance", "mean"),
        avg_tenure_active=("_tenure_if_active", "mean"),
        avg_engagement_active=("_engagement_if_active", "mean"),
    )

    metrics["net_change_ttm"] = metrics["active_now"] - metrics["active_ttm_ago"]
    avg_headcount = (metrics["active_now"] + metrics["active_ttm_ago"]) / 2
    metrics["attrition_rate_ttm"] = np.where(
        avg_headcount > 0, metrics["terms_ttm"] / avg_headcount, 0.0
    )
    metrics["pct_senior_active"] = np.where(
        metrics["active_now"] > 0, metrics["n_managers_active"] / metrics["active_now"], 0.0
    )

    return metrics


def merge_hiring_plan(dept_metrics: pd.DataFrame, plan_df: pd.DataFrame) -> pd.DataFrame:
    """Attach planned-hire counts to the department metrics table.

    Args:
        dept_metrics: Output of ``compute_department_metrics``.
        plan_df: Output of ``hiring_plan.build_plan_dataframe``.

    Returns:
        A copy of ``dept_metrics`` with two new columns: ``planned_hires``
        and ``planned_growth_pct`` (planned hires as a % of current
        headcount).
    """
    dept_metrics = dept_metrics.copy()
    plan_by_dept = plan_df.groupby("department")["planned_hires"].sum()

    unmapped = set(plan_by_dept.index) - set(dept_metrics.index)
    if unmapped:
        logger.warning(
            "Hiring plan references departments not found in the employee data: %s", unmapped
        )

    dept_metrics["planned_hires"] = plan_by_dept.reindex(dept_metrics.index).fillna(0).astype(int)
    dept_metrics["planned_growth_pct"] = np.where(
        dept_metrics["active_now"] > 0,
        dept_metrics["planned_hires"] / dept_metrics["active_now"] * 100,
        0.0,
    )
    return dept_metrics


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Scale a numeric series to [0, 1]; returns all zeros if constant.

    Args:
        series: Numeric series to normalize.

    Returns:
        Normalized series, same index as input.
    """
    value_range = series.max() - series.min()
    if value_range == 0:
        return series * 0.0
    return (series - series.min()) / value_range


def compute_risk_scores(dept_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite staffing-gap risk score (0-100) per department.

    Blends five normalized signals with equal weight: attrition rate
    (higher = riskier), clearance dependency (higher = longer hiring lead
    time = riskier), average tenure of active staff (lower = riskier),
    planned-growth intensity (a bigger hiring ask is a bigger execution
    risk), and management bench depth (fewer active senior staff = thinner
    "bus factor" = riskier).

    Args:
        dept_metrics: Department metrics table; must already include
            ``planned_growth_pct`` (i.e. run after ``merge_hiring_plan``).

    Returns:
        A copy of ``dept_metrics`` with a new ``risk_score`` column.
    """
    dept_metrics = dept_metrics.copy()
    components = pd.DataFrame({
        "attrition_risk": _min_max_normalize(dept_metrics["attrition_rate_ttm"]),
        "clearance_risk": _min_max_normalize(dept_metrics["clearance_pct"]),
        "tenure_risk": _min_max_normalize(-dept_metrics["avg_tenure_active"].fillna(0)),
        "growth_demand_risk": _min_max_normalize(dept_metrics["planned_growth_pct"]),
        "bus_factor_risk": _min_max_normalize(-dept_metrics["n_managers_active"]),
    })
    dept_metrics["risk_score"] = (components.mean(axis=1) * 100).round(1)
    return dept_metrics


def compute_company_kpis(
    df: pd.DataFrame, dept_metrics: pd.DataFrame, plan_df: pd.DataFrame
) -> dict[str, float]:
    """Roll department metrics up into company-wide headline KPIs.

    Args:
        df: Full employee DataFrame (used for the company-wide tenure
            average, which is computed at the employee level to avoid
            averaging department averages).
        dept_metrics: Output of ``compute_department_metrics``.
        plan_df: Output of ``hiring_plan.build_plan_dataframe``.

    Returns:
        Dict of scalar KPIs: active_now, active_ttm_ago, yoy_growth_pct,
        hires_ttm, terms_ttm, attrition_rate_pct, avg_tenure_active_years,
        planned_total.
    """
    active_now = int(dept_metrics["active_now"].sum())
    active_ttm_ago = int(dept_metrics["active_ttm_ago"].sum())
    hires_ttm = int(dept_metrics["hires_ttm"].sum())
    terms_ttm = int(dept_metrics["terms_ttm"].sum())
    avg_headcount = (active_now + active_ttm_ago) / 2

    return {
        "active_now": active_now,
        "active_ttm_ago": active_ttm_ago,
        "yoy_growth_pct": (active_now - active_ttm_ago) / active_ttm_ago * 100 if active_ttm_ago else 0.0,
        "hires_ttm": hires_ttm,
        "terms_ttm": terms_ttm,
        "attrition_rate_pct": terms_ttm / avg_headcount * 100 if avg_headcount else 0.0,
        "avg_tenure_active_years": df.loc[df["employment_status"] == "Active", "tenure_years"].mean(),
        "planned_total": int(plan_df["planned_hires"].sum()),
    }


def run_pipeline(config: DashboardConfig) -> dict[str, object]:
    """Execute the full data pipeline: load, validate, compute all metrics.

    Args:
        config: Validated dashboard configuration.

    Returns:
        Dict with keys ``employees`` (raw DataFrame), ``headcount_trend``,
        ``dept_metrics`` (with plan + risk score merged in), ``plan_df``,
        and ``kpis``.
    """
    employees = load_employee_data(config.input_csv_path)
    validate_employee_data(employees)

    snapshot = determine_snapshot_date(employees)
    trend_start = (
        pd.Timestamp(config.trend_start_date) if config.trend_start_date else employees["hire_date"].min()
    )

    headcount_trend = compute_headcount_trend(employees, snapshot, trend_start)
    dept_metrics = compute_department_metrics(employees, snapshot, config.ttm_months)

    plan_df = build_plan_dataframe()
    dept_metrics = merge_hiring_plan(dept_metrics, plan_df)
    dept_metrics = compute_risk_scores(dept_metrics)

    kpis = compute_company_kpis(employees, dept_metrics, plan_df)

    return {
        "employees": employees,
        "headcount_trend": headcount_trend,
        "dept_metrics": dept_metrics,
        "plan_df": plan_df,
        "kpis": kpis,
        "snapshot": snapshot,
    }
