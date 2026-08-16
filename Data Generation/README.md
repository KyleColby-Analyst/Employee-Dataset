# Ascendant Aerospace Corp. — Synthetic Employee Dataset

A fully synthetic HR/People Analytics dataset for a fictional commercial
aerospace/launch-vehicle company, generated from scratch in Python. Built as
the foundation for a People Analytics portfolio project (hiring pipeline &
time-to-fill analysis to follow in a Streamlit dashboard).

**No real people, companies, or HR records are used anywhere in this
project.** All names, employee IDs, and records are procedurally generated.

## What's in the dataset

`data/employees.csv` — one row per employee, 2,500 rows by default, with:

| Column | Description |
|---|---|
| `employee_id` | Unique ID, e.g. `AAC-000001` |
| `first_name`, `last_name`, `company_email` | Synthetic identity |
| `gender`, `ethnicity` | Broad EEOC-style demographic categories (aggregate analytics only) |
| `birth_date`, `age_years` | Derived from a plausible age-at-hire per level |
| `hire_date`, `tenure_years` | Hire date skews recent (fast-growing company) |
| `department`, `job_title`, `job_level` (1–10) | 17 departments modeled on a real launch-vehicle org chart |
| `city`, `state` | Weighted across real aerospace-industry hub locations |
| `education_level` | Tilted toward advanced degrees in engineering/research depts |
| `clearance_level`, `clearance_status` | `No Clearance` / `Public Trust` / `Secret` / `Top Secret`, weighted by department |
| `base_salary_usd`, `bonus_target_pct` | Scales with level and department |
| `performance_rating`, `engagement_score` | Latest review cycle / survey |
| `is_veteran`, `remote_eligible` | Flags |
| `employment_status`, `termination_date`, `termination_reason` | Attrition outcome |
| `manager_id` | Manager within the same department, one level up |

## Why the numbers aren't random noise

Attrition, pay, and clearance are all correlated on purpose so the dataset
supports real analysis instead of being pure noise:

- **Salary** = level base pay × department multiplier × small random noise.
- **Attrition risk** is a function of department base rate, job level
  (management attrition is lower), performance rating, and engagement score,
  converted from an annual hazard into a tenure-adjusted probability.
- **Clearance requirement** is weighted by department (Propulsion, Avionics,
  Test & Launch, Safety & Mission Assurance run highest).
- **Education level** skews toward Master's/PhD in Research, Legal, and
  core engineering departments.

## Project structure

```
people_analytics/
├── config.py                 # GeneratorConfig: validated run parameters
├── reference_data.py          # All lookup tables/distributions (departments, levels, names, etc.)
├── generate_employees.py       # Generation logic + CLI entry point
├── tests/
│   └── test_generate_employees.py
├── data/
│   └── employees.csv           # Generated output (regenerate any time)
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python generate_employees.py          # writes data/employees.csv
pytest tests/ -v                      # run the test suite
```

To generate a different size or a different point-in-time snapshot, edit
`GeneratorConfig` in `config.py` (or import it and pass your own values —
see `main()` in `generate_employees.py`):

```python
from config import GeneratorConfig
from generate_employees import build_employee_dataset, save_dataset

config = GeneratorConfig(n_employees=5000, seed=99)
df = build_employee_dataset(config)
save_dataset(df, config.output_path)
```

Every run is fully reproducible: the same `seed` + config always produces
byte-identical output.

## A note on dependencies

This was originally written against a spec calling for Pydantic and Faker.
It was authored in a network-isolated sandbox where those packages weren't
installable, so:
- **Pydantic → `dataclasses`**: `GeneratorConfig` is a frozen dataclass with
  the same validation-on-construction behavior. Swapping in a
  `pydantic.BaseModel` is a drop-in change if you have network access.
- **Faker → hand-curated name pools**: see `reference_data.py`. Swap in
  `faker.Faker()` calls for `generate_names()` if you'd prefer broader name
  variety.

## What's next

This dataset is the foundation for the hiring-pipeline / time-to-fill
analysis (requisitions, funnel stages, offers, recruiter performance) and
Streamlit dashboard planned as the next phase of this project.
