import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

from src.data_loader import load_config
from src.features import get_feature_target_split
from src.model import load_booster

app = Flask(__name__)

# Load configuration and trained artifacts
CONFIG = load_config("config/config.yaml")
MODEL_PATH = CONFIG["paths"]["model_output"]
METRICS_PATH = Path(CONFIG["paths"]["metrics_output"])
FEATURES_PATH = Path(CONFIG["paths"]["processed_data"])


def load_artifacts():
    """Load trained model, metrics JSON, and processed feature table."""
    if not Path(MODEL_PATH).exists() or not FEATURES_PATH.exists():
        raise RuntimeError(
            "Model or processed features not found. Run `uv run python main.py` first."
        )

    booster = load_booster(MODEL_PATH)
    df_features = pd.read_parquet(FEATURES_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return booster, df_features, metrics


BOOSTER, DF_FEATURES, METRICS = load_artifacts()
_, _, FEATURE_COLS = get_feature_target_split(DF_FEATURES, CONFIG)


@app.route("/")
def index():
    """Render the main forecasting dashboard."""
    skus = sorted(DF_FEATURES["sku"].astype(str).unique().tolist())
    regions = sorted(DF_FEATURES["region"].astype(str).unique().tolist())

    # Grab the latest week of data to showcase stockout recovery vs actual sales
    latest_date = DF_FEATURES["date"].max()
    latest_slice = DF_FEATURES[DF_FEATURES["date"] == latest_date].copy()

    X_latest, _, _ = get_feature_target_split(latest_slice, CONFIG)
    latest_slice["predicted_demand"] = np.clip(BOOSTER.predict(X_latest), a_min=0, a_max=None)

    # Prioritize rows where a stockout occurred so the UI highlights uncensored recovery
    sample_rows = (
        latest_slice.sort_values(by=["stockout", "true_demand"], ascending=[False, False])
        .head(15)
        .to_dict(orient="records")
    )

    return render_template(
        "index.html",
        metrics=METRICS,
        skus=skus,
        regions=regions,
        latest_date=str(latest_date.date()),
        sample_rows=sample_rows,
    )


@app.route("/api/sku-defaults", methods=["GET"])
def get_sku_defaults():
    """Return the latest price and promo status for a selected (sku, region)."""
    sku = request.args.get("sku")
    region = request.args.get("region")

    subset = DF_FEATURES[
        (DF_FEATURES["sku"].astype(str) == sku) & (DF_FEATURES["region"].astype(str) == region)
    ].sort_values("date")

    if subset.empty:
        return jsonify({"error": "No records found for this SKU and region combination."}), 404

    latest = subset.iloc[-1]
    return jsonify({
        "price": round(float(latest["price"]), 2),
        "on_promo": int(latest["on_promo"]),
        "category": str(latest["category"]),
    })


@app.route("/api/simulate", methods=["POST"])
def simulate_demand():
    """Run live LightGBM inference with user-specified price and promo inputs."""
    payload = request.get_json(force=True)
    sku = payload.get("sku")
    region = payload.get("region")
    custom_price = float(payload.get("price"))
    custom_promo = int(payload.get("on_promo"))

    subset = DF_FEATURES[
        (DF_FEATURES["sku"].astype(str) == sku) & (DF_FEATURES["region"].astype(str) == region)
    ].sort_values("date")

    if subset.empty:
        return jsonify({"error": "No historical context found for this SKU and region."}), 404

    base_row = subset.iloc[[-1]].copy()
    X_base, _, _ = get_feature_target_split(base_row, CONFIG)
    baseline_pred = float(np.clip(BOOSTER.predict(X_base)[0], a_min=0, a_max=None))

    # Apply simulated price & promotion changes
    sim_row = base_row.copy()
    old_price = float(sim_row["price"].iloc[0])
    sim_row["price"] = custom_price
    sim_row["on_promo"] = custom_promo

    # Update relative price ratio feature proportionally if present
    if "price_ratio_8w" in sim_row.columns and old_price > 0:
        sim_row["price_ratio_8w"] = sim_row["price_ratio_8w"] * (custom_price / old_price)

    X_sim, _, _ = get_feature_target_split(sim_row, CONFIG)
    simulated_pred = float(np.clip(BOOSTER.predict(X_sim)[0], a_min=0, a_max=None))

    return jsonify({
        "sku": sku,
        "region": region,
        "category": str(base_row["category"].iloc[0]),
        "baseline_demand": round(baseline_pred, 2),
        "predicted_demand": round(simulated_pred, 1),
        "projected_revenue": round(simulated_pred * custom_price, 2),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)