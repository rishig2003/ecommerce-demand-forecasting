import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def time_based_split(
    df: pd.DataFrame, date_col: str, val_weeks: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train and validation sets by trailing weeks."""
    max_date = df[date_col].max()
    cutoff_date = max_date - pd.Timedelta(weeks=val_weeks)

    train_df = df[df[date_col] <= cutoff_date].copy()
    val_df = df[df[date_col] > cutoff_date].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Time split produced an empty partition (cutoff={cutoff_date.date()}, "
            f"train_rows={len(train_df)}, val_rows={len(val_df)})."
        )

    return train_df, val_df


def compute_forecast_metrics(val_df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Compute statistical accuracy and business inventory impact metrics."""
    y_true = val_df[target_col].to_numpy()
    y_pred = val_df["predicted_demand"].to_numpy()
    prices = val_df["price"].to_numpy()

    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Weighted Absolute Percentage Error (WAPE) — handles zero-demand weeks safely
    total_actual_demand = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(total_actual_demand, 1e-8))

    # Forecast bias (> 0 over-forecasting; < 0 under-forecasting)
    bias = float(np.mean(y_pred - y_true))

    # Business impact metrics (assuming inventory ordered equals predicted_demand)
    under_forecast_units = np.maximum(0.0, y_true - y_pred)
    over_forecast_units = np.maximum(0.0, y_pred - y_true)

    unmet_demand_units = float(np.sum(under_forecast_units))
    excess_inventory_units = float(np.sum(over_forecast_units))
    estimated_stockout_revenue_loss = float(np.sum(under_forecast_units * prices))

    metrics: Dict[str, Any] = {
        "val_rows": int(len(val_df)),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "wape_pct": round(wape * 100, 2),
        "mean_bias_units": round(bias, 4),
        "total_true_demand_units": round(total_actual_demand, 2),
        "unmet_demand_units": round(unmet_demand_units, 2),
        "excess_inventory_units": round(excess_inventory_units, 2),
        "estimated_stockout_revenue_loss_usd": round(estimated_stockout_revenue_loss, 2),
    }

    # Category-level breakdown (e.g., Apparel, Electronics, Grocery)
    category_metrics = {}
    for category, group in val_df.groupby("category", observed=True):
        g_true = group[target_col].to_numpy()
        g_pred = group["predicted_demand"].to_numpy()
        g_wape = float(np.sum(np.abs(g_true - g_pred)) / max(np.sum(g_true), 1e-8))
        category_metrics[str(category)] = {
            "rmse": round(float(root_mean_squared_error(g_true, g_pred)), 4),
            "mae": round(float(mean_absolute_error(g_true, g_pred)), 4),
            "wape_pct": round(g_wape * 100, 2),
        }
    metrics["by_category"] = category_metrics

    # Cold-start SKU performance (first ~8 weeks)
    cold_start = val_df[val_df["is_new_launch"] == 1]
    if not cold_start.empty:
        cs_true = cold_start[target_col].to_numpy()
        cs_pred = cold_start["predicted_demand"].to_numpy()
        metrics["cold_start_sku_metrics"] = {
            "rows": int(len(cold_start)),
            "rmse": round(float(root_mean_squared_error(cs_true, cs_pred)), 4),
            "mae": round(float(mean_absolute_error(cs_true, cs_pred)), 4),
        }

    return metrics


def evaluate_and_save_report(
    val_df: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run validation metrics, print summary, and persist JSON report."""
    target_col = config["schema"]["target_col"]
    metrics = compute_forecast_metrics(val_df, target_col=target_col)

    metrics_path = Path(config["paths"]["metrics_output"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Validation Results | RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | "
        f"WAPE: {metrics['wape_pct']}% | "
        f"Est. Stockout Loss: ${metrics['estimated_stockout_revenue_loss_usd']:,.2f}"
    )
    print(f"Metrics report saved to: {metrics_path}")
    return metrics