"""Tests for the workforce dashboard data pipeline.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DashboardConfig  # noqa: E402
from data_pipeline import (  # noqa: E402
    compute_company_kpis,
    compute_department_metrics,
    compute_headcount_trend,
    compute_risk_scores,
    load_employee_data,
    merge_hiring_plan,
    validate_employee_data,
)
from hiring_plan import Requisition, build_plan_dataframe  # noqa: E402
from insights import generate_insights  # noqa: E402


@pytest.fixture
def sample_employees() -> pd.DataFrame:
    """A small, hand-constructed employee dataset with known properties.

    Two departments: "Alpha" (grows, low attrition) and "Beta" (shrinks,
    high attrition), snapshot fixed at 2026-01-01 for deterministic TTM math.
    """
    snapshot = pd.Timestamp("2026-01-01")
    rows = [
        # Alpha: 3 active, 0 terminated, all hired well before TTM window
        dict(employee_id="E1", department="Alpha", job_level=3, hire_date="2023-01-01",
             termination_date=None, employment_status="Active", termination_reason=None,
             clearance_level="No Clearance", tenure_years=3.0, engagement_score=4.0),
        dict(employee_id="E2", department="Alpha", job_level=8, hire_date="2022-01-01",
             termination_date=None, employment_status="Active", termination_reason=None,
             clearance_level="Secret", tenure_years=4.0, engagement_score=4.2),
        dict(employee_id="E3", department="Alpha", job_level=2, hire_date="2025-06-01",
             termination_date=None, employment_status="Active", termination_reason=None,
             clearance_level="No Clearance", tenure_years=0.6, engagement_score=3.8),
        # Beta: 1 active, 2 terminated within the TTM window (2025-01-01 to 2026-01-01)
        dict(employee_id="E4", department="Beta", job_level=4, hire_date="2021-01-01",
             termination_date=None, employment_status="Active", termination_reason=None,
             clearance_level="Public Trust", tenure_years=5.0, engagement_score=3.5),
        dict(employee_id="E5", department="Beta", job_level=2, hire_date="2022-01-01",
             termination_date="2025-06-01", employment_status="Terminated",
             termination_reason="Voluntary - Better Opportunity", clearance_level="No Clearance",
             tenure_years=3.4, engagement_score=2.9),
        dict(employee_id="E6", department="Beta", job_level=2, hire_date="2023-01-01",
             termination_date="2025-08-01", employment_status="Terminated",
             termination_reason="Involuntary - Performance", clearance_level="No Clearance",
             tenure_years=2.6, engagement_score=2.1),
    ]
    df = pd.DataFrame(rows)
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["termination_date"] = pd.to_datetime(df["termination_date"])
    return df


@pytest.fixture
def snapshot() -> pd.Timestamp:
    return pd.Timestamp("2026-01-01")


class TestLoadAndValidate:
    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_employee_data(tmp_path / "does_not_exist.csv")

    def test_validate_rejects_duplicate_ids(self, sample_employees: pd.DataFrame) -> None:
        dup = pd.concat([sample_employees, sample_employees.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate employee_id"):
            validate_employee_data(dup)

    def test_validate_rejects_termination_before_hire(self, sample_employees: pd.DataFrame) -> None:
        bad = sample_employees.copy()
        bad.loc[bad["employee_id"] == "E5", "termination_date"] = pd.Timestamp("2020-01-01")
        with pytest.raises(ValueError, match="precedes hire_date"):
            validate_employee_data(bad)

    def test_validate_rejects_empty_dataframe(self) -> None:
        with pytest.raises(ValueError, match="zero rows"):
            validate_employee_data(pd.DataFrame(columns=["employee_id"]))

    def test_validate_passes_on_clean_data(self, sample_employees: pd.DataFrame) -> None:
        validate_employee_data(sample_employees)  # should not raise


class TestHeadcountTrend:
    def test_trend_reflects_hires_and_terminations(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        trend = compute_headcount_trend(sample_employees, snapshot, pd.Timestamp("2021-01-01"))
        by_month = trend.set_index("month")["headcount"]
        # By 2021-01: only E4 hired (Beta) -> 1 active
        assert by_month.loc["2021-01-01"] == 1
        # By 2025-07: E1,E2,E3 active (Alpha), E4,E6 active (Beta; E5 terminated 2025-06) -> 5
        assert by_month.loc["2025-07-01"] == 5


class TestDepartmentMetrics:
    def test_beta_terms_and_attrition(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        beta = metrics.loc["Beta"]
        assert beta["terms_ttm"] == 2
        assert beta["active_now"] == 1
        assert beta["vol_terms_ttm"] == 1  # only E5 was voluntary

    def test_alpha_no_terminations(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        alpha = metrics.loc["Alpha"]
        assert alpha["terms_ttm"] == 0
        assert alpha["attrition_rate_ttm"] == 0
        assert alpha["active_now"] == 3

    def test_avg_tenure_excludes_terminated(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        # Beta's only active employee is E4 with tenure_years=5.0
        assert metrics.loc["Beta", "avg_tenure_active"] == pytest.approx(5.0)


class TestHiringPlanMerge:
    def test_merge_attaches_planned_hires(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        plan = build_plan_dataframe((Requisition("Widget Engineer", "Alpha", 5, "RPO"),))
        merged = merge_hiring_plan(metrics, plan)
        assert merged.loc["Alpha", "planned_hires"] == 5
        assert merged.loc["Beta", "planned_hires"] == 0

    def test_build_plan_dataframe_rejects_empty_plan(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_plan_dataframe(())

    def test_build_plan_dataframe_rejects_non_positive_count(self) -> None:
        with pytest.raises(ValueError, match="positive count"):
            build_plan_dataframe((Requisition("Bad Role", "Alpha", 0, "RPO"),))


class TestRiskScores:
    def test_risk_score_in_valid_range(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        plan = build_plan_dataframe((Requisition("Widget Engineer", "Alpha", 5, "RPO"),))
        merged = merge_hiring_plan(metrics, plan)
        scored = compute_risk_scores(merged)
        assert scored["risk_score"].between(0, 100).all()

    def test_higher_attrition_department_scores_higher_risk(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        plan = build_plan_dataframe((Requisition("Widget Engineer", "Alpha", 1, "RPO"),))
        merged = merge_hiring_plan(metrics, plan)
        scored = compute_risk_scores(merged)
        # Beta has attrition and lower tenure/bench than Alpha -> should score >= Alpha
        assert scored.loc["Beta", "risk_score"] >= scored.loc["Alpha", "risk_score"]


class TestCompanyKpis:
    def test_kpis_are_internally_consistent(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        plan = build_plan_dataframe((Requisition("Widget Engineer", "Alpha", 5, "RPO"),))
        merged = merge_hiring_plan(metrics, plan)
        kpis = compute_company_kpis(sample_employees, merged, plan)
        assert kpis["active_now"] == 4  # 3 Alpha + 1 Beta
        assert kpis["terms_ttm"] == 2
        assert kpis["planned_total"] == 5


class TestInsights:
    def test_generate_insights_returns_tuples(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        plan = build_plan_dataframe((Requisition("Widget Engineer", "Alpha", 1, "RPO"),))
        merged = merge_hiring_plan(metrics, plan)
        scored = compute_risk_scores(merged)
        kpis = compute_company_kpis(sample_employees, scored, plan)
        insights = generate_insights(scored, kpis)
        assert isinstance(insights, list)
        for headline, body in insights:
            assert isinstance(headline, str) and headline
            assert isinstance(body, str) and body

    def test_generate_insights_respects_max(self, sample_employees: pd.DataFrame, snapshot: pd.Timestamp) -> None:
        metrics = compute_department_metrics(sample_employees, snapshot, ttm_months=12)
        plan = build_plan_dataframe((Requisition("Widget Engineer", "Alpha", 1, "RPO"),))
        merged = merge_hiring_plan(metrics, plan)
        scored = compute_risk_scores(merged)
        kpis = compute_company_kpis(sample_employees, scored, plan)
        insights = generate_insights(scored, kpis, max_insights=2)
        assert len(insights) <= 2


class TestDashboardConfig:
    def test_rejects_non_csv_input(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"\.csv"):
            DashboardConfig(input_csv_path=tmp_path / "data.txt", output_png_path=tmp_path / "out.png")

    def test_rejects_non_positive_ttm(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ttm_months"):
            DashboardConfig(input_csv_path=tmp_path / "data.csv", output_png_path=tmp_path / "out.png", ttm_months=0)
