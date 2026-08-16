"""Static reference data for the fictional aerospace employer.

All distributions and lookup tables used to synthesize a realistic-looking
(but entirely fictional) HR dataset for a Blue-Origin-style aerospace
manufacturer live in this module. Keeping them separate from the generation
logic in ``generate_employees.py`` makes the assumptions easy to audit and
tune without touching the generation algorithm itself.
"""

from __future__ import annotations

from typing import Final, TypedDict


class DepartmentProfile(TypedDict):
    """Attributes that drive department-specific generation logic."""

    name: str
    role_noun: str
    clearance_base_prob: float  # probability an employee in this dept needs any clearance
    salary_multiplier: float  # relative to company base salary curve
    advanced_degree_prob: float  # probability of Master's/PhD
    attrition_base_rate: float  # annualized voluntary attrition base rate
    weight: float  # relative headcount weight


DEPARTMENTS: Final[list[DepartmentProfile]] = [
    {"name": "Propulsion Engineering", "role_noun": "Propulsion Engineer",
     "clearance_base_prob": 0.55, "salary_multiplier": 1.22,
     "advanced_degree_prob": 0.65, "attrition_base_rate": 0.08, "weight": 11.0},
    {"name": "Avionics & Flight Software", "role_noun": "Avionics Software Engineer",
     "clearance_base_prob": 0.60, "salary_multiplier": 1.20,
     "advanced_degree_prob": 0.60, "attrition_base_rate": 0.10, "weight": 10.0},
    {"name": "Structures & Materials", "role_noun": "Structures Engineer",
     "clearance_base_prob": 0.40, "salary_multiplier": 1.15,
     "advanced_degree_prob": 0.55, "attrition_base_rate": 0.08, "weight": 8.0},
    {"name": "Manufacturing & Production", "role_noun": "Manufacturing Technician",
     "clearance_base_prob": 0.25, "salary_multiplier": 0.85,
     "advanced_degree_prob": 0.10, "attrition_base_rate": 0.13, "weight": 18.0},
    {"name": "Test & Launch Operations", "role_noun": "Test & Launch Engineer",
     "clearance_base_prob": 0.65, "salary_multiplier": 1.10,
     "advanced_degree_prob": 0.40, "attrition_base_rate": 0.09, "weight": 9.0},
    {"name": "Safety & Mission Assurance", "role_noun": "Safety & Mission Assurance Engineer",
     "clearance_base_prob": 0.50, "salary_multiplier": 1.10,
     "advanced_degree_prob": 0.45, "attrition_base_rate": 0.07, "weight": 5.0},
    {"name": "Quality Engineering", "role_noun": "Quality Engineer",
     "clearance_base_prob": 0.30, "salary_multiplier": 1.00,
     "advanced_degree_prob": 0.35, "attrition_base_rate": 0.10, "weight": 6.0},
    {"name": "Research & Advanced Development", "role_noun": "Research Scientist",
     "clearance_base_prob": 0.45, "salary_multiplier": 1.25,
     "advanced_degree_prob": 0.80, "attrition_base_rate": 0.09, "weight": 4.0},
    {"name": "Supply Chain & Procurement", "role_noun": "Supply Chain Analyst",
     "clearance_base_prob": 0.15, "salary_multiplier": 0.95,
     "advanced_degree_prob": 0.30, "attrition_base_rate": 0.12, "weight": 6.0},
    {"name": "Program Management", "role_noun": "Program Manager",
     "clearance_base_prob": 0.35, "salary_multiplier": 1.15,
     "advanced_degree_prob": 0.50, "attrition_base_rate": 0.09, "weight": 5.0},
    {"name": "IT & Cybersecurity", "role_noun": "IT Systems Engineer",
     "clearance_base_prob": 0.40, "salary_multiplier": 1.05,
     "advanced_degree_prob": 0.35, "attrition_base_rate": 0.14, "weight": 4.0},
    {"name": "Human Resources", "role_noun": "HR Business Partner",
     "clearance_base_prob": 0.05, "salary_multiplier": 0.95,
     "advanced_degree_prob": 0.35, "attrition_base_rate": 0.12, "weight": 4.0},
    {"name": "Finance & Accounting", "role_noun": "Financial Analyst",
     "clearance_base_prob": 0.10, "salary_multiplier": 1.05,
     "advanced_degree_prob": 0.40, "attrition_base_rate": 0.11, "weight": 4.0},
    {"name": "Legal & Contracts", "role_noun": "Contracts Administrator",
     "clearance_base_prob": 0.20, "salary_multiplier": 1.20,
     "advanced_degree_prob": 0.70, "attrition_base_rate": 0.08, "weight": 2.0},
    {"name": "Facilities & EHS", "role_noun": "Facilities Coordinator",
     "clearance_base_prob": 0.10, "salary_multiplier": 0.80,
     "advanced_degree_prob": 0.10, "attrition_base_rate": 0.13, "weight": 3.0},
    {"name": "Sales & Business Development", "role_noun": "Business Development Manager",
     "clearance_base_prob": 0.10, "salary_multiplier": 1.10,
     "advanced_degree_prob": 0.45, "attrition_base_rate": 0.13, "weight": 2.0},
    {"name": "Executive Leadership", "role_noun": "Executive",
     "clearance_base_prob": 0.30, "salary_multiplier": 1.55,
     "advanced_degree_prob": 0.70, "attrition_base_rate": 0.04, "weight": 0.6},
]


class LevelProfile(TypedDict):
    """Attributes for a job level (1 = entry IC, 10 = C-suite)."""

    level: int
    label: str
    salary_base: float
    is_management: bool
    weight: float


LEVELS: Final[list[LevelProfile]] = [
    {"level": 1, "label": "Associate", "salary_base": 68_000, "is_management": False, "weight": 12.0},
    {"level": 2, "label": "", "salary_base": 78_000, "is_management": False, "weight": 18.0},
    {"level": 3, "label": "", "salary_base": 92_000, "is_management": False, "weight": 20.0},
    {"level": 4, "label": "Senior", "salary_base": 112_000, "is_management": False, "weight": 20.0},
    {"level": 5, "label": "Staff", "salary_base": 134_000, "is_management": False, "weight": 12.0},
    {"level": 6, "label": "Principal", "salary_base": 158_000, "is_management": False, "weight": 7.0},
    {"level": 7, "label": "Manager", "salary_base": 150_000, "is_management": True, "weight": 6.0},
    {"level": 8, "label": "Senior Manager", "salary_base": 178_000, "is_management": True, "weight": 3.0},
    {"level": 9, "label": "Director", "salary_base": 210_000, "is_management": True, "weight": 1.5},
    {"level": 10, "label": "Vice President", "salary_base": 235_000, "is_management": True, "weight": 0.5},
]


class LocationProfile(TypedDict):
    city: str
    state: str
    weight: float


LOCATIONS: Final[list[LocationProfile]] = [
    {"city": "Kent", "state": "WA", "weight": 38.0},
    {"city": "Cape Canaveral", "state": "FL", "weight": 20.0},
    {"city": "Huntsville", "state": "AL", "weight": 14.0},
    {"city": "Van Horn", "state": "TX", "weight": 10.0},
    {"city": "El Segundo", "state": "CA", "weight": 9.0},
    {"city": "Houston", "state": "TX", "weight": 6.0},
    {"city": "Washington", "state": "DC", "weight": 3.0},
]

GENDERS: Final[dict[str, float]] = {"Female": 0.31, "Male": 0.67, "Non-binary": 0.02}

# Broad EEOC-style race/ethnicity categories, used only in aggregate for
# representation analytics -- purely synthetic, not tied to any real person.
ETHNICITIES: Final[dict[str, float]] = {
    "White": 0.55,
    "Asian": 0.19,
    "Hispanic or Latino": 0.12,
    "Black or African American": 0.08,
    "Two or More Races": 0.04,
    "American Indian or Alaska Native": 0.01,
    "Native Hawaiian or Other Pacific Islander": 0.01,
}

EDUCATION_LEVELS: Final[dict[str, float]] = {
    "High School / Vocational": 0.12,
    "Associate's Degree": 0.08,
    "Bachelor's Degree": 0.48,
    "Master's Degree": 0.26,
    "PhD": 0.06,
}

# NOTE: "No Clearance" is used instead of the more obvious "None" because
# pandas.read_csv treats the literal string "None" as a null value by
# default (it's in the default na_values list). Using a non-ambiguous
# label avoids silently turning a real category into a missing value for
# anyone who opens this CSV with default pandas settings.
CLEARANCE_LEVELS: Final[dict[str, float]] = {
    "No Clearance": 0.45,
    "Public Trust": 0.20,
    "Secret": 0.25,
    "Top Secret": 0.10,
}

CLEARANCE_STATUSES: Final[list[str]] = ["Active", "In Process", "Expired", "Not Required"]

TERMINATION_REASONS_VOLUNTARY: Final[list[str]] = [
    "Voluntary - Better Opportunity",
    "Voluntary - Relocation",
    "Voluntary - Personal Reasons",
    "Voluntary - Return to School",
]
TERMINATION_REASONS_INVOLUNTARY: Final[list[str]] = [
    "Involuntary - Performance",
    "Involuntary - Layoff / Restructuring",
    "Involuntary - Policy Violation",
]

PERFORMANCE_RATINGS: Final[dict[str, float]] = {
    "Exceeds Expectations": 0.18,
    "Meets Expectations": 0.62,
    "Needs Improvement": 0.14,
    "Unsatisfactory": 0.06,
}

# Name pools (hand-curated, not sourced from any real-person dataset or API).
FIRST_NAMES_FEMALE: Final[list[str]] = [
    "Olivia", "Emma", "Ava", "Sophia", "Isabella", "Mia", "Charlotte", "Amelia",
    "Harper", "Evelyn", "Abigail", "Emily", "Elizabeth", "Sofia", "Avery",
    "Ella", "Scarlett", "Grace", "Chloe", "Victoria", "Riley", "Aria",
    "Lily", "Aubrey", "Zoey", "Penelope", "Layla", "Nora", "Camila", "Hannah",
    "Priya", "Anaya", "Fatima", "Mei", "Ines", "Nadia", "Yuki", "Keisha",
    "Rosa", "Ling", "Aisha", "Junie", "Talia", "Renata", "Simone",
]
FIRST_NAMES_MALE: Final[list[str]] = [
    "Liam", "Noah", "Oliver", "Elijah", "William", "James", "Benjamin",
    "Lucas", "Henry", "Alexander", "Mason", "Michael", "Ethan", "Daniel",
    "Jacob", "Logan", "Jackson", "Levi", "Sebastian", "Mateo", "Jack",
    "Owen", "Theodore", "Aiden", "Samuel", "Joseph", "John", "David",
    "Wei", "Raj", "Kwame", "Diego", "Hiroshi", "Andrei", "Mohammed",
    "Carlos", "Amir", "Tariq", "Lin", "Ravi", "Dmitri", "Kenji", "Marcus",
]
FIRST_NAMES_NONBINARY: Final[list[str]] = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Rowan", "Sage",
]
LAST_NAMES: Final[list[str]] = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Patel", "Kim", "Chen", "Singh", "Khan", "Ali",
    "Ivanov", "Kowalski", "Muller", "Dubois", "Rossi", "Suzuki", "Okafor",
]

EXECUTIVE_TITLES: Final[list[str]] = [
    "Chief Executive Officer",
    "Chief Financial Officer",
    "Chief Technology Officer",
    "Chief Operating Officer",
    "Chief Human Resources Officer",
    "VP of Engineering",
    "VP of Manufacturing",
    "VP of Program Management",
    "VP of Mission Assurance",
    "VP of Business Development",
    "VP of Legal & General Counsel",
    "VP of Supply Chain",
]

COMPANY_NAME: Final[str] = "Ascendant Aerospace Corp."
COMPANY_FOUNDING_YEAR: Final[int] = 2014
