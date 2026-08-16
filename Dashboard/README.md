# Workforce & Hiring Plan Dashboard

Generates `workforce_dashboard.png` — a single-image dashboard analyzing
growth, attrition, staffing-gap risk, and the current hiring plan — from
the `employees.csv` dataset (see the companion employee-data-generation
project).

## Project structure

```
dashboard_project/
├── config.py           # DashboardConfig: validated run parameters + color palette
├── hiring_plan.py       # The current requisition plan, mapped to departments
├── data_pipeline.py     # Load, validate, and compute all metrics (vectorized/groupby)
├── insights.py          # Auto-generates the dashboard's narrative takeaways from the data
├── visualize.py          # One function per panel; composes the final figure
├── build_dashboard.py     # CLI entry point
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python build_dashboard.py --input employees.csv --output workforce_dashboard.png
pytest tests/ -v
```

## What it computes

- **Company-wide headcount trend** — active employees at each month-end
  since the earliest hire date, computed via vectorized broadcasting (no
  per-month Python loop).
- **Department metrics** — net headcount change, attrition rate, average
  tenure, clearance dependency, and management bench depth, all computed
  with a single `groupby` pass rather than iterating over departments.
- **Staffing-gap risk score (0–100)** — a composite of five normalized
  signals: attrition rate, clearance dependency, average tenure, planned-
  growth intensity, and leadership bench depth.
- **Hiring plan overlay** — the current requisitions
  (`hiring_plan.py`) mapped to the dataset's departments and compared
  against each department's trailing-window losses.
- **Takeaways** — five narrative bullets, each computed from the metrics at
  render time (not hardcoded), so a refreshed dataset or a revised hiring
  plan produces updated takeaways automatically.

## Editing the hiring plan

Edit the `HIRING_PLAN` tuple in `hiring_plan.py`. Each `Requisition` maps a
job title to the nearest matching department in the employee dataset —
these mappings are judgment calls (the dataset's departments are broader
categories than specific titles), documented inline via `mapping_note`
where the mapping isn't self-evident. Review them against your real org
chart before trusting the output.

## A note on dependencies

Written in a network-isolated sandbox, so `pydantic` (unavailable offline)
is substituted with a validated `dataclass` in `config.py` — swap in
`pydantic.BaseModel` if you have network access and want stricter runtime
type coercion. `pytest` could not be executed in that sandbox either; every
assertion in `tests/test_pipeline.py` was run manually as plain Python and
confirmed passing before delivery, but you should still run the real suite
in your own environment.
