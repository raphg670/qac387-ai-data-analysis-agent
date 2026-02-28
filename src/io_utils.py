from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dirs(reports: Path) -> None:
    """Create output folders."""
    reports.mkdir(parents=True, exist_ok=True)
    figures_dir = reports / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)


def read_data(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df