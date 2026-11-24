from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extract weekly calendar and cyclical seasonal features."""
    df = df.copy()
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype("int64")
    df["quarter"] = df[date_col].dt.quarter.astype("int64")

    # Cyclical encoding so December (12) and January (1) sit close together
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    entity_cols: List[str],
    lag_weeks: List[int],
    rolling_windows: List[int],
) -> pd.DataFrame:
    """Create leak-free historical lags and trailing rolling stats per (sku, region)."""
    df = df.copy()
    grouped = df.groupby(entity_cols, observed=True)

    # 1. Lagged observed sales and lagged stockout indicators
    for lag in lag_weeks:
        df[f"units_sold_lag_{lag}w"] = grouped["units_sold"].shift(lag)
        df[f"stockout_lag_{lag}w"] = grouped["stockout"].shift(lag)
        df[f"on_promo_lag_{lag}w"] = grouped["on_promo"].shift(lag)

    # 2. Trailing rolling windows (shifted by 1 week to prevent same-week leakage)
    shifted_sales = grouped["units_sold"].shift(1)
    shifted_stockout = grouped["stockout"].shift(1)

    for window in rolling_windows:
        df[f"units_sold_roll_mean_{window}w"] = (
            shifted_sales.groupby([df[c] for c in entity_cols], observed=True)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(entity_cols))), drop=True)
        )
        df[f"units_sold_roll_std_{window}w"] = (
            shifted_sales.groupby([df[c] for c in entity_cols], observed=True)
            .rolling(window=window, min_periods=2)
            .std()
            .reset_index(level=list(range(len(entity_cols))), drop=True)
        )
        df[f"stockout_rate_roll_{window}w"] = (
            shifted_stockout.groupby([df[c] for c in entity_cols], observed=True)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(entity_cols))), drop=True)
        )

    return df


def add_price_features(
    df: pd.DataFrame,
    entity_cols: List[str],
    price_lag_weeks: List[int],
) -> pd.DataFrame:
    """Create relative price dynamics and promo discount signals."""
    df = df.copy()
    grouped = df.groupby(entity_cols, observed=True)

    for lag in price_lag_weeks:
        lagged_price = grouped["price"].shift(lag)
        df[f"price_lag_{lag}w"] = lagged_price
        df[f"price_pct_change_{lag}w"] = (df["price"] - lagged_price) / lagged_price.replace(0, np.nan)

    # Trailing 8-week average price to capture baseline vs. discounted price ratio
    shifted_price = grouped["price"].shift(1)
    rolling_avg_price = (
        shifted_price.groupby([df[c] for c in entity_cols], observed=True)
        .rolling(window=8, min_periods=1)
        .mean()
        .reset_index(level=list(range(len(entity_cols))), drop=True)
    )
    df["price_to_roll8w_ratio"] = df["price"] / rolling_avg_price.replace(0, np.nan)

    # Explicit interaction between active promotion and seasonal multiplier
    df["promo_x_seasonal"] = df["on_promo"] * df["seasonal_index"]
    return df


def build_features(df: pd.DataFrame, config: Dict[str, Any], save_processed: bool = True) -> pd.DataFrame:
    """Run full feature engineering pipeline on validated data."""
    date_col = config["schema"]["date_col"]
    entity_cols = config["schema"]["entity_cols"]
    lag_weeks = config["features"]["lag_weeks"]
    rolling_windows = config["features"]["rolling_windows"]
    price_lag_weeks = config["features"]["price_lag_weeks"]

    # Ensure strict chronological ordering before any shift/rolling operation
    df_feat = df.sort_values(by=entity_cols + [date_col]).reset_index(drop=True)

    df_feat = add_calendar_features(df_feat, date_col=date_col)
    df_feat = add_lag_and_rolling_features(
        df_feat,
        entity_cols=entity_cols,
        lag_weeks=lag_weeks,
        rolling_windows=rolling_windows,
    )
    df_feat = add_price_features(
        df_feat,
        entity_cols=entity_cols,
        price_lag_weeks=price_lag_weeks,
    )

    if save_processed:
        out_path = Path(config["paths"]["processed_data"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_parquet(out_path, index=False)

    return df_feat


def get_feature_target_split(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate model input features (X) from target (y), dropping all leakage columns."""
    date_col = config["schema"]["date_col"]
    target_col = config["schema"]["target_col"]
    leakage_cols = set(config["schema"]["leakage_cols"])

    excluded = leakage_cols | {date_col}
    feature_cols = [c for c in df.columns if c not in excluded]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols