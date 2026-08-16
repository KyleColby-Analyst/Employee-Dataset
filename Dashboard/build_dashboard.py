"""Generate the workforce & hiring-plan dashboard PNG from an employee CSV.

Usage:
    python build_dashboard.py
    python build_dashboard.py --input employees.csv --output dashboard.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import DashboardConfig
from data_pipeline import run_pipeline
from visualize import build_figure, save_dashboard

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging with timestamps for this pipeline.

    Args:
        level: Logging level threshold.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with ``input`` and ``output`` paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("employees.csv"),
                         help="Path to the employee dataset CSV (default: employees.csv)")
    parser.add_argument("--output", type=Path, default=Path("workforce_dashboard.png"),
                         help="Path to write the dashboard PNG (default: workforce_dashboard.png)")
    return parser.parse_args()


def main() -> None:
    """Entry point: run the pipeline and render the dashboard."""
    configure_logging()
    args = parse_args()

    try:
        config = DashboardConfig(input_csv_path=args.input, output_png_path=args.output)
        result = run_pipeline(config)
        fig = build_figure(result, config)
        save_dashboard(fig, config.output_png_path, config.dpi)
    except FileNotFoundError as exc:
        logger.error("Input file problem: %s", exc)
        raise
    except ValueError as exc:
        logger.error("Data or configuration problem: %s", exc)
        raise

    logger.info("Done. Dashboard written to %s", config.output_png_path)


if __name__ == "__main__":
    main()
