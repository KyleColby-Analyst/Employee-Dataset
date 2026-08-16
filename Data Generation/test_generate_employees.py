"""Tests for the synthetic employee dataset generator.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GeneratorConfig  # noqa: E402
from generate_employees import (  # noqa: E402
    assign_clearance,
    assign_departments,
    assign_levels,
    build_employee_dataset,
    generate_company_email,
    validate_dataset,
)


@pytest.fixture
def small_config() -> GeneratorConfig:
    """A small, fast-to-generate config for unit tests."""
    return GeneratorConfig(n_employees=200, seed=7)


@pytest.fixture
def small_dataset(small_config: GeneratorConfig) -> pd.DataFrame:
    """A fully built small employee dataset, shared across tests."""
    return build_employee_dataset(small_config)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded numpy Generator for deterministic unit tests."""
    return np.random.default_rng(123)


class TestGeneratorConfig:
    """Validation behavior of GeneratorConfig."""

    def test_rejects_non_positive_employee_count(self) -> None:
        with pytest.raises(ValueError, match="n_employees must be positive"):
            GeneratorConfig(n_employees=0)

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ValueError, match="seed must be non-negative"):
            GeneratorConfig(seed=-1)

    def test_rejects_hire_window_after_snapshot(self) -> None:
        with pytest.raises(ValueError, match="must be before snapshot_date"):
            from datetime import date
            GeneratorConfig(earliest_hire_date=date(2030, 1, 1), snapshot_date=date(2026, 1, 1))

    def test_default_config_is_valid(self) -> None:
        cfg = GeneratorConfig()
        assert cfg.n_employees > 0


class TestDepartmentAndLevelAssignment:
    """Unit tests for individual vectorized assignment functions."""

    def test_assign_departments_returns_correct_length(self, rng: np.random.Generator) -> None:
        result = assign_departments(rng, 500)
        assert len(result) == 500

    def test_assign_departments_only_known_values(self, rng: np.random.Generator) -> None:
        from reference_data import DEPARTMENTS
        valid_names = {d["name"] for d in DEPARTMENTS}
        result = assign_departments(rng, 300)
        assert set(result.unique()).issubset(valid_names)

    def test_executive_department_always_level_10(self, rng: np.random.Generator) -> None:
        departments = assign_departments(rng, 1000)
        levels = assign_levels(rng, departments)
        exec_levels = levels[(departments == "Executive Leadership").to_numpy()]
        assert (exec_levels == 10).all()

    def test_non_executive_never_level_10(self, rng: np.random.Generator) -> None:
        departments = assign_departments(rng, 1000)
        levels = assign_levels(rng, departments)
        non_exec_levels = levels[(departments != "Executive Leadership").to_numpy()]
        assert (non_exec_levels < 10).all()


class TestEmailGeneration:
    """Email addresses must be unique even with duplicate name pairs."""

    def test_duplicate_names_get_unique_emails(self) -> None:
        first = pd.Series(["Jordan", "Jordan", "Jordan"])
        last = pd.Series(["Lee", "Lee", "Lee"])
        emails = generate_company_email(first, last)
        assert emails.nunique() == 3
        assert emails.iloc[0] == "jordan.lee@ascendant.com"
        assert emails.iloc[1] == "jordan.lee2@ascendant.com"

    def test_emails_are_lowercase_and_well_formed(self) -> None:
        first = pd.Series(["Anaya", "Wei"])
        last = pd.Series(["Okafor", "Chen"])
        emails = generate_company_email(first, last)
        assert all("@ascendant.com" in e for e in emails)
        assert all(e == e.lower() for e in emails)


class TestClearanceAssignment:
    """Clearance level/status pairing must be internally consistent."""

    def test_no_clearance_implies_not_required_status(self, rng: np.random.Generator) -> None:
        departments = assign_departments(rng, 500)
        clearance_level, clearance_status = assign_clearance(rng, departments)
        no_clearance_mask = clearance_level == "No Clearance"
        assert (clearance_status[no_clearance_mask] == "Not Required").all()

    def test_has_clearance_implies_not_not_required(self, rng: np.random.Generator) -> None:
        departments = assign_departments(rng, 500)
        clearance_level, clearance_status = assign_clearance(rng, departments)
        has_clearance_mask = clearance_level != "No Clearance"
        assert (clearance_status[has_clearance_mask] != "Not Required").all()


class TestFullDatasetIntegrity:
    """End-to-end checks on a fully assembled dataset."""

    def test_correct_row_count(self, small_dataset: pd.DataFrame, small_config: GeneratorConfig) -> None:
        assert len(small_dataset) == small_config.n_employees

    def test_no_duplicate_employee_ids(self, small_dataset: pd.DataFrame) -> None:
        assert small_dataset["employee_id"].is_unique

    def test_no_duplicate_emails(self, small_dataset: pd.DataFrame) -> None:
        assert small_dataset["company_email"].is_unique

    def test_termination_date_never_before_hire_date(self, small_dataset: pd.DataFrame) -> None:
        terminated = small_dataset[small_dataset["employment_status"] == "Terminated"]
        assert (terminated["termination_date"] >= terminated["hire_date"]).all()

    def test_active_employees_have_no_termination_date(self, small_dataset: pd.DataFrame) -> None:
        active = small_dataset[small_dataset["employment_status"] == "Active"]
        assert active["termination_date"].isna().all()

    def test_critical_columns_have_no_nulls(self, small_dataset: pd.DataFrame) -> None:
        critical = ["employee_id", "department", "job_level", "hire_date", "base_salary_usd"]
        assert small_dataset[critical].isna().sum().sum() == 0

    def test_salaries_are_positive(self, small_dataset: pd.DataFrame) -> None:
        assert (small_dataset["base_salary_usd"] > 0).all()

    def test_top_level_employee_has_no_manager(self, small_dataset: pd.DataFrame) -> None:
        top_idx = small_dataset["job_level"].idxmax()
        assert pd.isna(small_dataset.loc[top_idx, "manager_id"])

    def test_reproducible_with_same_seed(self, small_config: GeneratorConfig) -> None:
        df_a = build_employee_dataset(small_config)
        df_b = build_employee_dataset(small_config)
        pd.testing.assert_series_equal(df_a["employee_id"], df_b["employee_id"])
        pd.testing.assert_series_equal(df_a["base_salary_usd"], df_b["base_salary_usd"])

    def test_validate_dataset_passes_on_clean_data(self, small_dataset: pd.DataFrame) -> None:
        validate_dataset(small_dataset)  # should not raise

    def test_validate_dataset_catches_duplicate_ids(self, small_dataset: pd.DataFrame) -> None:
        corrupted = small_dataset.copy()
        corrupted.loc[1, "employee_id"] = corrupted.loc[0, "employee_id"]
        with pytest.raises(ValueError, match="duplicate employee_id"):
            validate_dataset(corrupted)
