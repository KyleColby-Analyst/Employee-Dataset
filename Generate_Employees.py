"""Generate a synthetic employee dataset for a fictional aerospace company.

This module builds a realistic-looking (entirely fictional) HR/People
Analytics dataset -- demographics, department/level/title, compensation,
security clearance, performance, engagement, and attrition -- for
"Ascendant Aerospace Corp.", modeled loosely on the workforce shape of a
commercial launch-vehicle manufacturer (think Blue Origin / SpaceX).

Design notes
------------
* No row-wise ``.iterrows()``/Python loops over employees are used for
  column construction; all per-employee fields are built with vectorized
  numpy/pandas operations. The only Python-level loop in this module
  iterates over ~17 departments (for manager assignment), not over rows.
* Correlations are deliberately baked in (e.g. salary scales with level and
  department, attrition risk scales with performance/engagement/tenure) so
  the dataset supports meaningful downstream analysis rather than being
  pure noise.
* Reproducibility is guaranteed via a single seeded ``numpy.random.Generator``
  threaded through every function -- never module-level global randomness.

Run directly to generate the default dataset:
    python generate_employees.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import GeneratorConfig
from reference_data import (
    CLEARANCE_LEVELS,
    CLEARANCE_STATUSES,
    COMPANY_NAME,
    DEPARTMENTS,
    EDUCATION_LEVELS,
    ETHNICITIES,
    EXECUTIVE_TITLES,
    FIRST_NAMES_FEMALE,
    FIRST_NAMES_MALE,
    FIRST_NAMES_NONBINARY,
    GENDERS,
    LAST_NAMES,
    LEVELS,
    LOCATIONS,
    PERFORMANCE_RATINGS,
    TERMINATION_REASONS_INVOLUNTARY,
    TERMINATION_REASONS_VOLUNTARY,
)

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging with timestamps for this pipeline.

    Args:
        level: Logging level threshold (e.g. ``logging.INFO``,
            ``logging.DEBUG``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _weighted_sample(
    rng: np.random.Generator,
    options: list[str],
    weights: list[float],
    size: int,
) -> np.ndarray:
    """Draw a weighted-random sample of categorical values.

    Args:
        rng: Seeded numpy random Generator.
        options: Candidate category labels.
        weights: Relative weights (need not sum to 1) matching ``options``.
        size: Number of draws.

    Returns:
        A numpy array of ``size`` sampled labels.

    Raises:
        ValueError: If ``options`` and ``weights`` have mismatched lengths.
    """
    if len(options) != len(weights):
        raise ValueError(
            f"options (len={len(options)}) and weights (len={len(weights)}) "
            "must be the same length"
        )
    probs = np.asarray(weights, dtype=np.float64)
    probs = probs / probs.sum()
    return rng.choice(np.asarray(options, dtype=object), size=size, p=probs)


def assign_departments(rng: np.random.Generator, n: int) -> pd.Series:
    """Assign each employee a department, weighted by realistic headcount share.

    Args:
        rng: Seeded random Generator.
        n: Number of employees.

    Returns:
        Categorical pandas Series of department names, length ``n``.
    """
    names = [d["name"] for d in DEPARTMENTS]
    weights = [d["weight"] for d in DEPARTMENTS]
    values = _weighted_sample(rng, names, weights, n)
    return pd.Series(values, name="department", dtype="category")


def assign_levels(rng: np.random.Generator, departments: pd.Series) -> pd.Series:
    """Assign a job level (1-10) to each employee.

    Level 10 (Vice President) is reserved for the "Executive Leadership"
    department; every other department draws from levels 1-9, weighted
    toward the individual-contributor band as is typical in engineering-
    heavy organizations.

    Args:
        rng: Seeded random Generator.
        departments: Department assignment per employee (same length as
            the desired output).

    Returns:
        Integer pandas Series of job levels, one per employee.
    """
    n = len(departments)
    non_exec_levels = [lv["level"] for lv in LEVELS if lv["level"] < 10]
    non_exec_weights = [lv["weight"] for lv in LEVELS if lv["level"] < 10]

    levels = _weighted_sample(
        rng, [str(lv) for lv in non_exec_levels], non_exec_weights, n
    ).astype(np.int32)

    is_exec = (departments == "Executive Leadership").to_numpy()
    levels[is_exec] = 10
    return pd.Series(levels, name="job_level", dtype="int32")


def assign_titles(rng: np.random.Generator, departments: pd.Series, levels: pd.Series) -> pd.Series:
    """Construct a job title from department role noun + level label.

    Args:
        rng: Seeded random Generator (used to vary executive titles so the
            Executive Leadership department isn't just 30 identical "Vice
            President" rows).
        departments: Department per employee.
        levels: Job level per employee.

    Returns:
        String pandas Series of job titles.
    """
    role_noun_by_dept = {d["name"]: d["role_noun"] for d in DEPARTMENTS}
    label_by_level = {lv["level"]: lv["label"] for lv in LEVELS}

    role_nouns = departments.map(role_noun_by_dept)
    level_labels = levels.map(label_by_level)

    management_titles = np.where(
        levels >= 9,
        "Director, " + departments.astype(str),
        np.where(levels == 7, "Manager, " + departments.astype(str),
                 np.where(levels == 8, "Senior Manager, " + departments.astype(str), "")),
    )
    ic_titles = (level_labels.astype(str) + " " + role_nouns.astype(str)).str.strip()

    exec_mask = (departments == "Executive Leadership").to_numpy()
    mgmt_mask = (levels >= 7).to_numpy() & ~exec_mask

    titles = ic_titles.to_numpy(dtype=object)
    titles[mgmt_mask] = np.asarray(management_titles, dtype=object)[mgmt_mask]

    titles[exec_mask] = rng.choice(EXECUTIVE_TITLES, size=int(exec_mask.sum()))
    return pd.Series(titles, name="job_title", dtype="object")


def assign_gender(rng: np.random.Generator, n: int) -> pd.Series:
    """Assign gender per employee from a fixed distribution.

    Args:
        rng: Seeded random Generator.
        n: Number of employees.

    Returns:
        Categorical Series of gender labels.
    """
    values = _weighted_sample(rng, list(GENDERS.keys()), list(GENDERS.values()), n)
    return pd.Series(values, name="gender", dtype="category")


def assign_ethnicity(rng: np.random.Generator, n: int) -> pd.Series:
    """Assign a broad EEOC-style race/ethnicity category per employee.

    Args:
        rng: Seeded random Generator.
        n: Number of employees.

    Returns:
        Categorical Series of ethnicity labels.
    """
    values = _weighted_sample(rng, list(ETHNICITIES.keys()), list(ETHNICITIES.values()), n)
    return pd.Series(values, name="ethnicity", dtype="category")


def generate_names(rng: np.random.Generator, gender: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Generate first/last names consistent with each employee's gender.

    Args:
        rng: Seeded random Generator.
        gender: Gender per employee (drives first-name pool selection).

    Returns:
        Tuple of (first_name Series, last_name Series).
    """
    n = len(gender)
    first_names = np.empty(n, dtype=object)

    female_mask = (gender == "Female").to_numpy()
    male_mask = (gender == "Male").to_numpy()
    nb_mask = ~female_mask & ~male_mask

    first_names[female_mask] = rng.choice(FIRST_NAMES_FEMALE, size=female_mask.sum())
    first_names[male_mask] = rng.choice(FIRST_NAMES_MALE, size=male_mask.sum())
    first_names[nb_mask] = rng.choice(FIRST_NAMES_NONBINARY, size=nb_mask.sum())

    last_names = rng.choice(LAST_NAMES, size=n)
    return (
        pd.Series(first_names, name="first_name", dtype="object"),
        pd.Series(last_names, name="last_name", dtype="object"),
    )


def generate_company_email(first_name: pd.Series, last_name: pd.Series) -> pd.Series:
    """Build a unique company email address per employee.

    Collisions (e.g. two "Jordan Lee"s) are disambiguated with a numeric
    suffix, entirely vectorized via a cumulative-count-within-group.

    Args:
        first_name: Employee first names.
        last_name: Employee last names.

    Returns:
        String Series of unique email addresses.
    """
    base = (
        first_name.str.lower().str.replace(r"[^a-z]", "", regex=True)
        + "."
        + last_name.str.lower().str.replace(r"[^a-z]", "", regex=True)
    )
    dup_index = base.groupby(base).cumcount()
    suffix = np.where(dup_index == 0, "", (dup_index + 1).astype(str))
    domain = COMPANY_NAME.lower().split(" ")[0] + ".com"
    return (base + suffix + "@" + domain).rename("company_email")


def assign_locations(rng: np.random.Generator, n: int) -> tuple[pd.Series, pd.Series]:
    """Assign a work-site city/state pair per employee.

    Args:
        rng: Seeded random Generator.
        n: Number of employees.

    Returns:
        Tuple of (city Series, state Series).
    """
    labels = [f"{loc['city']}|{loc['state']}" for loc in LOCATIONS]
    weights = [loc["weight"] for loc in LOCATIONS]
    drawn = _weighted_sample(rng, labels, weights, n)
    split = np.char.split(drawn.astype(str), sep="|")
    cities = np.array([s[0] for s in split], dtype=object)
    states = np.array([s[1] for s in split], dtype=object)
    return (
        pd.Series(cities, name="city", dtype="category"),
        pd.Series(states, name="state", dtype="category"),
    )


def generate_hire_dates(
    rng: np.random.Generator, n: int, config: GeneratorConfig
) -> pd.Series:
    """Generate hire dates weighted toward recent years (company growth).

    Uses a triangular distribution skewed toward ``snapshot_date`` to
    reflect a fast-growing company that has hired more people in recent
    years than in its earliest years.

    Args:
        rng: Seeded random Generator.
        n: Number of employees.
        config: Generator configuration (bounds the hire-date window).

    Returns:
        Datetime64 Series of hire dates.
    """
    start_ord = pd.Timestamp(config.earliest_hire_date).toordinal()
    end_ord = pd.Timestamp(config.snapshot_date).toordinal()
    draws = rng.triangular(left=start_ord, mode=end_ord, right=end_ord, size=n)
    ordinals = np.clip(draws, start_ord, end_ord).astype(np.int64)
    dates = pd.to_datetime([pd.Timestamp.fromordinal(o) for o in ordinals])
    return pd.Series(dates, name="hire_date")


def generate_birth_dates(
    rng: np.random.Generator, hire_date: pd.Series, level: pd.Series
) -> pd.Series:
    """Generate birth dates implying a plausible age-at-hire per level.

    Args:
        rng: Seeded random Generator.
        hire_date: Hire date per employee.
        level: Job level per employee (higher levels imply older
            age-at-hire on average).

    Returns:
        Datetime64 Series of birth dates.
    """
    n = len(hire_date)
    min_age = np.select(
        [level <= 3, level <= 6, level <= 8, level >= 9],
        [21, 25, 30, 35],
    )
    max_age = np.select(
        [level <= 3, level <= 6, level <= 8, level >= 9],
        [40, 52, 58, 63],
    )
    age_at_hire_days = rng.uniform(min_age * 365.25, max_age * 365.25, size=n)
    birth_dates = (hire_date - pd.to_timedelta(age_at_hire_days, unit="D")).dt.normalize()
    return pd.Series(birth_dates, name="birth_date")


def assign_education(rng: np.random.Generator, departments: pd.Series) -> pd.Series:
    """Assign education level, tilted upward for advanced-degree-heavy departments.

    Args:
        rng: Seeded random Generator.
        departments: Department per employee.

    Returns:
        Categorical Series of education levels.
    """
    n = len(departments)
    base = _weighted_sample(
        rng, list(EDUCATION_LEVELS.keys()), list(EDUCATION_LEVELS.values()), n
    )
    adv_prob_by_dept = {d["name"]: d["advanced_degree_prob"] for d in DEPARTMENTS}
    dept_adv_prob = departments.map(adv_prob_by_dept).to_numpy(dtype=np.float64)

    upgrade_roll = rng.uniform(size=n)
    is_low_ed = np.isin(base, ["High School / Vocational", "Associate's Degree", "Bachelor's Degree"])
    should_upgrade = is_low_ed & (upgrade_roll < dept_adv_prob)

    upgraded = np.where(
        rng.uniform(size=n) < 0.25, "PhD", "Master's Degree"
    )
    final = np.where(should_upgrade, upgraded, base)
    return pd.Series(final, name="education_level", dtype="category")


def assign_clearance(rng: np.random.Generator, departments: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Assign security clearance level and status per employee.

    Args:
        rng: Seeded random Generator.
        departments: Department per employee (drives clearance likelihood).

    Returns:
        Tuple of (clearance_level Series, clearance_status Series).
    """
    n = len(departments)
    clearance_prob_by_dept = {d["name"]: d["clearance_base_prob"] for d in DEPARTMENTS}
    needs_clearance_prob = departments.map(clearance_prob_by_dept).to_numpy(dtype=np.float64)
    needs_clearance = rng.uniform(size=n) < needs_clearance_prob

    levels = np.full(n, "No Clearance", dtype=object)
    tier_labels = [lv for lv in CLEARANCE_LEVELS if lv != "No Clearance"]
    tier_weights = [CLEARANCE_LEVELS[lv] for lv in tier_labels]
    levels[needs_clearance] = _weighted_sample(
        rng, tier_labels, tier_weights, int(needs_clearance.sum())
    )

    statuses = np.full(n, "Not Required", dtype=object)
    status_labels = ["Active", "In Process", "Expired"]
    status_weights = [0.80, 0.12, 0.08]
    statuses[needs_clearance] = _weighted_sample(
        rng, status_labels, status_weights, int(needs_clearance.sum())
    )

    return (
        pd.Series(levels, name="clearance_level", dtype="category"),
        pd.Series(statuses, name="clearance_status", dtype="category"),
    )


def assign_compensation(
    rng: np.random.Generator, level: pd.Series, departments: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Compute base salary and target bonus percentage per employee.

    Salary = level base salary * department multiplier * lognormal noise,
    which keeps pay within a realistic band while preserving the
    level/department salary hierarchy analysts would expect to find.

    Args:
        rng: Seeded random Generator.
        level: Job level per employee.
        departments: Department per employee.

    Returns:
        Tuple of (base_salary Series [float32], bonus_target_pct Series [float32]).
    """
    n = len(level)
    salary_base_by_level = {lv["level"]: lv["salary_base"] for lv in LEVELS}
    mult_by_dept = {d["name"]: d["salary_multiplier"] for d in DEPARTMENTS}

    base = level.map(salary_base_by_level).to_numpy(dtype=np.float64)
    mult = departments.map(mult_by_dept).to_numpy(dtype=np.float64)
    noise = rng.lognormal(mean=0.0, sigma=0.08, size=n)

    salary = np.round(base * mult * noise, -2)  # round to nearest $100
    bonus_pct = np.round(np.select(
        [level <= 3, level <= 6, level <= 8, level >= 9],
        [0.05, 0.10, 0.15, 0.25],
    ) * rng.uniform(0.9, 1.1, size=n), 3)

    return (
        pd.Series(salary, name="base_salary_usd", dtype="float32"),
        pd.Series(bonus_pct, name="bonus_target_pct", dtype="float32"),
    )


def assign_performance_and_engagement(
    rng: np.random.Generator, n: int
) -> tuple[pd.Series, pd.Series]:
    """Assign the latest performance rating and engagement survey score.

    Args:
        rng: Seeded random Generator.
        n: Number of employees.

    Returns:
        Tuple of (performance_rating Series, engagement_score Series [float32]).
    """
    ratings = _weighted_sample(
        rng, list(PERFORMANCE_RATINGS.keys()), list(PERFORMANCE_RATINGS.values()), n
    )
    rating_series = pd.Series(ratings, name="performance_rating", dtype="category")

    rating_center = rating_series.map(
        {"Exceeds Expectations": 4.4, "Meets Expectations": 3.7,
         "Needs Improvement": 2.6, "Unsatisfactory": 1.8}
    ).to_numpy(dtype=np.float64)
    engagement = np.clip(rng.normal(rating_center, 0.55, size=n), 1.0, 5.0)
    engagement_series = pd.Series(
        np.round(engagement, 1), name="engagement_score", dtype="float32"
    )
    return rating_series, engagement_series


def simulate_attrition(
    rng: np.random.Generator,
    hire_date: pd.Series,
    level: pd.Series,
    departments: pd.Series,
    performance: pd.Series,
    engagement: pd.Series,
    config: GeneratorConfig,
) -> pd.DataFrame:
    """Simulate whether/when each employee terminated, and why.

    Termination probability blends department base attrition rate, job
    level (management attrition is lower), performance rating, and
    engagement score into an annualized hazard, then converts that hazard
    into a "has this person left by now" probability given their tenure
    (an exponential survival approximation).

    Args:
        rng: Seeded random Generator.
        hire_date: Hire date per employee.
        level: Job level per employee.
        departments: Department per employee.
        performance: Performance rating per employee.
        engagement: Engagement score per employee.
        config: Generator configuration (provides snapshot date).

    Returns:
        DataFrame with columns ``employment_status``, ``termination_date``,
        and ``termination_reason`` (the latter two are ``NaT``/``None`` for
        active employees).
    """
    n = len(hire_date)
    snapshot = pd.Timestamp(config.snapshot_date)
    tenure_years = ((snapshot - hire_date).dt.days / 365.25).to_numpy()
    tenure_years = np.clip(tenure_years, 0.0, None)

    base_rate_by_dept = {d["name"]: d["attrition_base_rate"] for d in DEPARTMENTS}
    dept_rate = departments.map(base_rate_by_dept).to_numpy(dtype=np.float64)

    level_factor = np.select(
        [level <= 2, level <= 6, level <= 8, level >= 9],
        [1.25, 1.0, 0.75, 0.55],
    )
    perf_factor = performance.map(
        {"Exceeds Expectations": 0.65, "Meets Expectations": 1.0,
         "Needs Improvement": 1.6, "Unsatisfactory": 2.4}
    ).to_numpy(dtype=np.float64)
    engagement_factor = 1.0 + (3.0 - engagement.to_numpy(dtype=np.float64)) * 0.18

    annual_hazard = dept_rate * level_factor * perf_factor * engagement_factor
    prob_left_by_now = 1.0 - np.exp(-annual_hazard * tenure_years)

    left = rng.uniform(size=n) < prob_left_by_now

    frac_of_tenure = rng.beta(a=2.0, b=1.5, size=n)  # skews toward later-in-tenure exits
    term_days = np.clip(frac_of_tenure * tenure_years * 365.25, 30, None)
    termination_date = pd.Series(
        np.where(left, (hire_date + pd.to_timedelta(term_days, unit="D")).to_numpy(), np.datetime64("NaT")),
        name="termination_date",
    )
    termination_date = pd.to_datetime(termination_date).dt.normalize().clip(upper=snapshot)

    poor_perf = performance.isin(["Unsatisfactory", "Needs Improvement"]).to_numpy()
    involuntary_roll = rng.uniform(size=n)
    is_involuntary = left & poor_perf & (involuntary_roll < 0.55)
    is_layoff = left & ~is_involuntary & (rng.uniform(size=n) < 0.06)
    is_voluntary = left & ~is_involuntary & ~is_layoff

    reason = np.full(n, None, dtype=object)
    reason[is_involuntary] = rng.choice(
        TERMINATION_REASONS_INVOLUNTARY[:2], size=int(is_involuntary.sum())
    )
    reason[is_layoff] = TERMINATION_REASONS_INVOLUNTARY[2]
    reason[is_voluntary] = rng.choice(
        TERMINATION_REASONS_VOLUNTARY, size=int(is_voluntary.sum())
    )

    status = np.where(left, "Terminated", "Active")

    return pd.DataFrame({
        "employment_status": pd.Series(status, dtype="category"),
        "termination_date": termination_date,
        "termination_reason": pd.Series(reason, dtype="category"),
    })


def assign_managers(df: pd.DataFrame) -> pd.Series:
    """Assign each employee a manager within their own department.

    For each department, active management-eligible employees at higher
    levels are candidate managers for lower-level employees. The single
    highest-level employee company-wide has no manager (implicit CEO).
    This is the one place the code loops over a small, fixed set of
    groups (departments) rather than employee rows.

    Args:
        df: Employee DataFrame; must already contain ``employee_id``,
            ``department``, and ``job_level`` columns.

    Returns:
        Series of ``manager_id`` values aligned to ``df``'s index (``None``
        for employees with no eligible manager, e.g. the CEO).
    """
    manager_id = pd.Series(pd.NA, index=df.index, dtype="object")
    top_employee_idx = df["job_level"].idxmax()

    for dept_name, group in df.groupby("department", observed=True):
        levels_sorted = sorted(group["job_level"].unique(), reverse=True)
        for lvl in levels_sorted:
            higher_pool = group.loc[group["job_level"] > lvl, "employee_id"]
            current_idx = group.loc[group["job_level"] == lvl].index
            if higher_pool.empty:
                continue
            rng_local = np.random.default_rng(abs(hash((dept_name, lvl))) % (2**32))
            chosen = rng_local.choice(higher_pool.to_numpy(), size=len(current_idx))
            manager_id.loc[current_idx] = chosen

    manager_id.loc[top_employee_idx] = pd.NA
    return manager_id.rename("manager_id")


def introduce_missingness(
    rng: np.random.Generator, df: pd.DataFrame, config: GeneratorConfig
) -> pd.DataFrame:
    """Null out a small fraction of non-critical fields to mimic real HRIS exports.

    Only fields that would plausibly go unrecorded in a real system
    (engagement score, education level, bonus target) are affected --
    identifiers, dates, and compensation are left intact so downstream
    joins and financial totals stay reliable.

    Args:
        rng: Seeded random Generator.
        df: Employee DataFrame to perturb (not mutated in place).
        config: Generator configuration (provides missingness rate).

    Returns:
        A copy of ``df`` with realistic missing values introduced.
    """
    df = df.copy()
    nullable_cols = ["engagement_score", "education_level", "ethnicity"]
    for col in nullable_cols:
        mask = rng.uniform(size=len(df)) < config.random_missingness_rate
        df.loc[mask, col] = np.nan
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Run defensive checks on the assembled dataset before writing it out.

    Args:
        df: Fully assembled employee DataFrame.

    Raises:
        ValueError: If duplicate employee IDs, duplicate emails, or
            impossible date orderings (hire after termination) are found.
    """
    dup_ids = df["employee_id"].duplicated().sum()
    if dup_ids:
        raise ValueError(f"Found {dup_ids} duplicate employee_id values")

    dup_emails = df["company_email"].duplicated().sum()
    if dup_emails:
        raise ValueError(f"Found {dup_emails} duplicate company_email values")

    terminated = df[df["employment_status"] == "Terminated"]
    bad_dates = (terminated["termination_date"] < terminated["hire_date"]).sum()
    if bad_dates:
        raise ValueError(f"Found {bad_dates} records with termination_date before hire_date")

    critical_cols = ["employee_id", "department", "job_level", "hire_date", "base_salary_usd"]
    null_counts = df[critical_cols].isna().sum()
    if null_counts.any():
        raise ValueError(f"Unexpected nulls in critical columns:\n{null_counts[null_counts > 0]}")

    logger.info("Validation passed: %d rows, %d columns, no duplicate keys.", *df.shape)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and cast low-cardinality text to category.

    Args:
        df: Employee DataFrame.

    Returns:
        A new DataFrame with reduced memory footprint. Note: CSV output
        does not preserve dtype metadata -- this mainly benefits any
        in-memory analysis done immediately after generation (or if the
        dataset is instead persisted to Parquet).
    """
    df = df.copy()
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def build_employee_dataset(config: GeneratorConfig) -> pd.DataFrame:
    """Orchestrate full generation of the synthetic employee dataset.

    Args:
        config: Validated generator configuration.

    Returns:
        The complete, validated employee DataFrame.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_employees
    logger.info("Generating %d employee records for %s (seed=%d)", n, COMPANY_NAME, config.seed)

    department = assign_departments(rng, n)
    level = assign_levels(rng, department)
    title = assign_titles(rng, department, level)
    gender = assign_gender(rng, n)
    ethnicity = assign_ethnicity(rng, n)
    first_name, last_name = generate_names(rng, gender)
    email = generate_company_email(first_name, last_name)
    city, state = assign_locations(rng, n)
    hire_date = generate_hire_dates(rng, n, config)
    birth_date = generate_birth_dates(rng, hire_date, level)
    education = assign_education(rng, department)
    clearance_level, clearance_status = assign_clearance(rng, department)
    base_salary, bonus_pct = assign_compensation(rng, level, department)
    performance, engagement = assign_performance_and_engagement(rng, n)
    attrition = simulate_attrition(
        rng, hire_date, level, department, performance, engagement, config
    )
    is_veteran = pd.Series(rng.uniform(size=n) < 0.12, name="is_veteran")
    remote_eligible = pd.Series(
        (department.isin(["IT & Cybersecurity", "Finance & Accounting", "Human Resources",
                           "Legal & Contracts", "Sales & Business Development"]).to_numpy()
         & (rng.uniform(size=n) < 0.6)),
        name="remote_eligible",
    )

    employee_id = pd.Series(
        [f"AAC-{i:06d}" for i in range(1, n + 1)], name="employee_id"
    )

    df = pd.concat([
        employee_id, first_name, last_name, email, gender, ethnicity, birth_date,
        hire_date, department, title, level.rename("job_level"), city, state,
        education, clearance_level, clearance_status, base_salary, bonus_pct,
        performance, engagement, is_veteran, remote_eligible, attrition,
    ], axis=1)

    df["manager_id"] = assign_managers(df)

    snapshot = pd.Timestamp(config.snapshot_date)
    end_date = df["termination_date"].where(df["employment_status"] == "Terminated", snapshot)
    df["tenure_years"] = np.round(((end_date - df["hire_date"]).dt.days / 365.25), 2).astype("float32")
    df["age_years"] = np.round(((snapshot - df["birth_date"]).dt.days / 365.25), 1).astype("float32")

    df = introduce_missingness(rng, df, config)
    df = optimize_dtypes(df)

    validate_dataset(df)
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Write the dataset to CSV, creating parent directories as needed.

    Args:
        df: Employee DataFrame to persist.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows x %d columns to %s", *df.shape, output_path)


def main() -> None:
    """Entry point: generate the default dataset and write it to disk."""
    configure_logging()
    config = GeneratorConfig()
    df = build_employee_dataset(config)
    save_dataset(df, config.output_path)


if __name__ == "__main__":
    main()