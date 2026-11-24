from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "date",
    "sku",
    "category",
    "region",
    "units_sold",
    "true_demand",
    "price",
    "on_promo",
    "stockout",
    "returns",
    "seasonal_index",
    "is_new_launch",
    "is_declining",
    "month",
    "net_units",
    "revenue",
}


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_data(df: pd.DataFrame) -> None:
    """Run schema and business-logic data quality checks."""
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    # Check for duplicate (sku, region, date) records
    dupes = df.duplicated(subset=["sku", "region", "date"]).sum()
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate (sku, region, date) rows.")

    # Verify revenue determinism: revenue == net_units * price
    expected_revenue = df["net_units"] * df["price"]
    if not np.allclose(df["revenue"], expected_revenue, atol=1e-4):
        max_diff = np.abs(df["revenue"] - expected_revenue).max()
        raise ValueError(
            f"Revenue validation failed: max discrepancy of {max_diff:.4f} detected."
        )

    # Verify net_units determinism: net_units == units_sold - returns
    if not (df["net_units"] == (df["units_sold"] - df["returns"])).all():
        raise ValueError("Data integrity error: net_units != units_sold - returns.")

    # Verify uncensored demand >= observed units_sold
    if not (df["true_demand"] >= df["units_sold"]).all():
        raise ValueError("Data integrity error: true_demand < units_sold detected.")


def load_and_clean_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load raw e-commerce demand dataset, enforce types, sort, and validate."""
    raw_path = Path(config["paths"]["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    # Parse weekly Monday dates and sort strictly by entity + time
    date_col = config["schema"]["date_col"]
    entity_cols = config["schema"]["entity_cols"]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=entity_cols + [date_col]).reset_index(drop=True)

    # Enforce clean dtypes
    for col in config["schema"]["categorical_cols"]:
        df[col] = df[col].astype("category")

    int_cols = [
        "units_sold",
        "true_demand",
        "on_promo",
        "stockout",
        "returns",
        "is_new_launch",
        "is_declining",
        "month",
        "net_units",
    ]
    for col in int_cols:
        df[col] = df[col].astype("int64")

    float_cols = ["price", "seasonal_index", "revenue"]
    for col in float_cols:
        df[col] = df[col].astype("float64")

    validate_data(df)
    return df