from pathlib import Path
from typing import Dict, Any, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import get_feature_target_split


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """Train LightGBM demand forecasting model and save artifact to disk."""
    model_params = config["model"]["params"].copy()
    early_stopping_rounds = config["model"]["early_stopping_rounds"]

    X_train, y_train, feature_cols = get_feature_target_split(train_df, config)
    X_val, y_val, _ = get_feature_target_split(val_df, config)

    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=model_params.get("metric", "rmse"),
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )

    model_path = Path(config["paths"]["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))

    print(
        f"Training complete | Best iteration: {model.best_iteration_} | "
        f"Features: {len(feature_cols)} | Train rows: {len(X_train):,} | "
        f"Val rows: {len(X_val):,} | Model saved to: {model_path}"
    )

    val_scored = val_df.copy()
    val_scored["predicted_demand"] = np.clip(model.predict(X_val), a_min=0, a_max=None)
    return model, val_scored


def load_booster(model_path: str) -> lgb.Booster:
    """Load a saved LightGBM booster from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    return lgb.Booster(model_file=str(path))


def predict_demand(
    df_features: pd.DataFrame, config: Dict[str, Any], model: lgb.LGBMRegressor | lgb.Booster | None = None
) -> pd.DataFrame:
    """Generate non-negative demand forecasts for an engineered feature DataFrame."""
    if model is None:
        model = load_booster(config["paths"]["model_output"])

    X, _, _ = get_feature_target_split(df_features, config)
    preds = np.clip(model.predict(X), a_min=0, a_max=None)

    out = df_features[["date", "sku", "category", "region", "price", "on_promo"]].copy()
    out["predicted_demand"] = preds
    return out