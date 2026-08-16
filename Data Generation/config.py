"""Configuration schema for the synthetic employee dataset generator.

NOTE ON DEPENDENCIES
---------------------
The original spec calls for Pydantic models. This code was authored in a
network-isolated sandbox where third-party packages (``pydantic``,
``faker``) could not be installed. ``dataclasses`` (stdlib) is used instead,
with the same guarantees that matter here: explicit typing, a single
validated construction point, and immutability after creation. If you have
network access in your own environment, swapping this for a
``pydantic.BaseModel`` is a drop-in change -- the field names and
validation rules are written to translate directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters controlling synthetic employee data generation.

    Args:
        n_employees: Total number of employee records to generate
            (includes both active and terminated employees).
        seed: Random seed for reproducibility. Re-running the generator
            with the same seed and config produces an identical dataset.
        snapshot_date: The "as of" date the dataset represents. Ages,
            tenure, and active/terminated status are all computed relative
            to this date.
        earliest_hire_date: No employee can have a hire date before this.
            Defaults to the company's founding year.
        output_path: Destination path for the generated CSV file.
        random_missingness_rate: Fraction of non-critical fields to null
            out at random, simulating realistic HRIS data quality issues.

    Raises:
        ValueError: If any field fails validation (e.g. non-positive
            employee count, or a hire-date window that doesn't make sense).
    """

    n_employees: int = 2_500
    seed: int = 42
    snapshot_date: date = date(2026, 8, 3)
    earliest_hire_date: date = date(2014, 6, 1)
    output_path: Path = field(default_factory=lambda: Path("data/employees.csv"))
    random_missingness_rate: float = 0.015

    def __post_init__(self) -> None:
        """Validate field values immediately after construction."""
        if self.n_employees <= 0:
            raise ValueError(f"n_employees must be positive, got {self.n_employees}")
        if self.n_employees > 500_000:
            raise ValueError(
                f"n_employees={self.n_employees} is unreasonably large for "
                "an in-memory pandas generation run; consider chunked generation."
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.earliest_hire_date >= self.snapshot_date:
            raise ValueError(
                "earliest_hire_date "
                f"({self.earliest_hire_date}) must be before snapshot_date "
                f"({self.snapshot_date})"
            )
        if not (0.0 <= self.random_missingness_rate < 0.5):
            raise ValueError(
                "random_missingness_rate must be in [0.0, 0.5), got "
                f"{self.random_missingness_rate}"
            )
