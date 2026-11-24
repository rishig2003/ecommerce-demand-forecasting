from src.data_loader import load_config, load_and_clean_data
from src.features import build_features
from src.validation import time_based_split, evaluate_and_save_report
from src.model import train_model


def main() -> None:
    config = load_config("config/config.yaml")

    # 1. Ingest & validate raw weekly data
    df_clean = load_and_clean_data(config)

    # 2. Engineer leak-free lag, rolling, price, and calendar features
    df_features = build_features(df_clean, config, save_processed=True)

    # 3. Chronological out-of-time split
    train_df, val_df = time_based_split(
        df_features,
        date_col=config["schema"]["date_col"],
        val_weeks=config["split"]["val_weeks"],
    )

    # 4. Train LightGBM with early stopping & score validation weeks
    _, val_scored = train_model(train_df, val_df, config)

    # 5. Compute statistical + stockout business metrics & save report
    evaluate_and_save_report(val_scored, config)


if __name__ == "__main__":
    main()