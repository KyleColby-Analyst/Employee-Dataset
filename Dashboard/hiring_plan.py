"""The current hiring requisition plan, mapped to departments.

The dataset's ``department`` column is a broad functional category (17
departments), while the hiring plan is expressed as specific job titles.
Where a title doesn't exist verbatim in the dataset, it's mapped to the
closest matching department by function. These mappings are judgment calls,
not ground truth -- they're centralized here (rather than scattered through
the analysis code) specifically so they're easy to review and correct
against the real org chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

RecruitingChannel = Literal["RPO", "Recruiting Firm", "Internal"]


@dataclass(frozen=True)
class Requisition:
    """A single line item in the hiring plan.

    Args:
        role: Job title as given in the hiring plan.
        department: Nearest matching department in the employee dataset.
        count: Number of open positions for this role.
        channel: Recruiting channel responsible for filling the role.
        mapping_note: Brief rationale when the role-to-department mapping
            isn't self-evident from the title alone.
    """

    role: str
    department: str
    count: int
    channel: RecruitingChannel
    mapping_note: str = ""


HIRING_PLAN: tuple[Requisition, ...] = (
    # --- RPO focus ---
    Requisition("Mechanical Engineer II", "Structures & Materials", 17, "RPO"),
    Requisition("Mechanical Engineer I", "Structures & Materials", 33, "RPO"),
    Requisition("Welding Technician", "Manufacturing & Production", 2, "RPO"),
    Requisition("Junior Technician", "Manufacturing & Production", 12, "RPO"),
    Requisition("Program Manager", "Program Management", 4, "RPO"),
    # --- Recruiting firm focus ---
    Requisition("Engineering Lead", "Propulsion Engineering", 1, "Recruiting Firm",
                "Assumed core propulsion engineering leadership; verify against org chart."),
    Requisition("Senior Electrical Engineer", "Avionics & Flight Software", 1, "Recruiting Firm"),
    Requisition("Electrical Engineer", "Avionics & Flight Software", 1, "Recruiting Firm"),
    Requisition("Business Development Manager", "Sales & Business Development", 1, "Recruiting Firm"),
    Requisition("Controller", "Finance & Accounting", 1, "Recruiting Firm"),
    Requisition("Compliance Manager", "Safety & Mission Assurance", 1, "Recruiting Firm",
                "Aerospace regulatory compliance treated as adjacent to Safety & Mission Assurance."),
    Requisition("Sales Engineer", "Sales & Business Development", 2, "Recruiting Firm"),
    Requisition("Lead Vehicle", "Test & Launch Operations", 1, "Recruiting Firm",
                "Assumed vehicle integration/launch role; title is ambiguous as given."),
    Requisition("Sr. Avionics EE", "Avionics & Flight Software", 1, "Recruiting Firm"),
    # --- Internal ---
    Requisition("HR Coordinator", "Human Resources", 1, "Internal"),
    Requisition("IT Manager", "IT & Cybersecurity", 1, "Internal"),
    Requisition("IT Specialist", "IT & Cybersecurity", 1, "Internal"),
    Requisition("Safety Officer", "Safety & Mission Assurance", 2, "Internal"),
    Requisition("Technical Writer", "Quality Engineering", 2, "Internal",
                "Documentation/compliance writing treated as adjacent to Quality Engineering."),
)


def build_plan_dataframe(plan: tuple[Requisition, ...] = HIRING_PLAN) -> pd.DataFrame:
    """Convert the hiring plan into a DataFrame for analysis.

    Args:
        plan: Sequence of requisitions; defaults to the module-level
            ``HIRING_PLAN``.

    Returns:
        DataFrame with one row per requisition line item (columns: role,
        department, count, channel, mapping_note).

    Raises:
        ValueError: If the plan is empty, or any requisition count isn't
            a positive integer.
    """
    if not plan:
        raise ValueError("Hiring plan is empty; nothing to analyze.")
    if any(r.count <= 0 for r in plan):
        bad = [r.role for r in plan if r.count <= 0]
        raise ValueError(f"Requisitions must have a positive count; offending roles: {bad}")

    return pd.DataFrame([
        {"role": r.role, "department": r.department, "planned_hires": r.count,
         "channel": r.channel, "mapping_note": r.mapping_note}
        for r in plan
    ])
