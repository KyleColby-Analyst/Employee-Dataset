"""Configuration schema for the workforce dashboard pipeline.

NOTE ON DEPENDENCIES: as with the dataset generator, this was written in a
network-isolated sandbox where ``pydantic`` could not be installed. A frozen
``dataclass`` with ``__post_init__`` validation is used instead -- swap in
``pydantic.BaseModel`` if you have network access and want strict runtime
type coercion in addition to the validation performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ColorPalette:
    """Centralized color scheme so the dashboard's look is defined in one place."""

    navy: str = "#1B2A4A"
    teal: str = "#2A9D8F"
    coral: str = "#E63946"
    amber: str = "#F4A261"
    slate: str = "#6C757D"
    light_bg: str = "#F7F8FA"
    grid: str = "#E3E6EA"


@dataclass(frozen=True)
class DashboardConfig:
    """Parameters controlling data processing and rendering of the dashboard.

    Args:
        input_csv_path: Path to the employee dataset CSV (as produced by
            ``generate_employees.py``).
        output_png_path: Destination path for the rendered dashboard image.
        ttm_months: Length, in months, of the "trailing twelve months"
            window used for hires/departures/attrition-rate calculations.
        trend_start_date: Earliest month shown in the company-wide headcount
            trend line. ``None`` means "use the earliest hire date found in
            the data".
        fig_width_in: Figure width in inches.
        fig_height_in: Figure height in inches.
        dpi: Output resolution in dots per inch.
        top_n_plan_departments: Max number of departments to show in the
            "hiring plan vs. historical losses" panel (sorted by planned
            headcount; keeps that panel readable if the plan spans many
            departments).
        palette: Color scheme for all chart elements.

    Raises:
        ValueError: If any field fails validation.
    """

    input_csv_path: Path
    output_png_path: Path
    ttm_months: int = 12
    trend_start_date: str | None = None
    fig_width_in: float = 20.0
    fig_height_in: float = 30.0
    dpi: int = 150
    top_n_plan_departments: int = 15
    palette: ColorPalette = field(default_factory=ColorPalette)

    def __post_init__(self) -> None:
        """Validate configuration values immediately after construction."""
        if self.ttm_months <= 0:
            raise ValueError(f"ttm_months must be positive, got {self.ttm_months}")
        if self.fig_width_in <= 0 or self.fig_height_in <= 0:
            raise ValueError(
                "fig_width_in and fig_height_in must be positive, got "
                f"({self.fig_width_in}, {self.fig_height_in})"
            )
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")
        if self.top_n_plan_departments <= 0:
            raise ValueError(
                f"top_n_plan_departments must be positive, got {self.top_n_plan_departments}"
            )
        if not self.input_csv_path.suffix == ".csv":
            raise ValueError(
                f"input_csv_path must point to a .csv file, got {self.input_csv_path}"
            )
