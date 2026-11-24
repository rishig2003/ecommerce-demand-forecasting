import json
import pandas as pd
import pytest

from src.validation import (
    time_based_split,
    compute_forecast_metrics,
    evaluate_and_save_report,
)


def test_time_based_split_no_temporal_leakage():
    """Ensure train dates are strictly earlier than validation dates."""
    dates = pd.date_range("2025-01-06", periods=20, freq="W-MON")
    df = pd.DataFrame({
        "date": dates,
        "sku": ["SKU_001"] * 20,
        "region": ["North"] * 20,
        "true_demand": range(100, 120),
    })

    train_df, val_df = time_based_split(df, date_col="date", val_weeks=4)

    assert len(val_df) == 4
    assert len(train_df) == 16
    assert train_df["date"].max() < val_df["date"].min()


def test_time_based_split_raises_on_excessive_val_weeks():
    """Ensure an empty train partition raises a clear ValueError."""
    dates = pd.date_range("2025-01-06", periods=4, freq="W-MON")
    df = pd.DataFrame({"date": dates, "true_demand": [10, 20, 30, 40]})

    with pytest.raises(ValueError, match="empty partition"):
        time_based_split(df, date_col="date", val_weeks=10)


def test_compute_forecast_metrics_business_impact():
    """Verify WAPE, bias, unmet demand, excess inventory, and stockout revenue loss."""
    val_df = pd.DataFrame({
        "category": ["apparel", "apparel"],
        "is_new_launch": [0, 1],
        "price": [50.0, 100.0],
        "true_demand": [100.0, 100.0],
        # Row 0 under-forecasts by 20 units ($1,000 stockout loss)
        # Row 1 over-forecasts by 10 units (10 units excess inventory)
        "predicted_demand": [80.0, 110.0],
    })

    metrics = compute_forecast_metrics(val_df, target_col="true_demand")

    assert metrics["mae"] == 15.0
    assert metrics["wape_pct"] == 15.0  # (20 + 10) / 200 = 15%
    assert metrics["mean_bias_units"] == -5.0  # (-20 + 10) / 2 = -5
    assert metrics["unmet_demand_units"] == 20.0
    assert metrics["excess_inventory_units"] == 10.0
    assert metrics["estimated_stockout_revenue_loss_usd"] == 1000.0  # 20 * $50
    assert "apparel" in metrics["by_category"]
    assert metrics["cold_start_sku_metrics"]["rows"] == 1


def test_evaluate_and_save_report_writes_json(tmp_path):
    """Verify evaluate_and_save_report serializes valid JSON to the configured path."""
    output_file = tmp_path / "reports" / "metrics.json"
    config = {
        "schema": {"target_col": "true_demand"},
        "paths": {"metrics_output": str(output_file)},
    }
    val_df = pd.DataFrame({
        "category": ["electronics"],
        "is_new_launch": [0],
        "price": [200.0],
        "true_demand": [50.0],
        "predicted_demand": [45.0],
    })

    metrics = evaluate_and_save_report(val_df, config)

    assert output_file.exists()
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved["estimated_stockout_revenue_loss_usd"] == 1000.0
    assert saved == metrics