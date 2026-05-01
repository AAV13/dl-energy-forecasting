import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle, os

RAW_PATH = "data/time_series_60min_singleindex.csv"
OUT_PATH  = "data/processed.pkl"

FEATURE_COLS = [
    "DE_load_actual_entsoe_transparency",
    "DE_load_forecast_entsoe_transparency",
    "DE_solar_generation_actual",
    "DE_solar_capacity",
    "DE_wind_onshore_generation_actual",
    "DE_wind_offshore_generation_actual",
    "DE_wind_onshore_capacity",
    "DE_wind_offshore_capacity",
]

TARGET_COLS = [
    "DE_solar_generation_actual",
    "DE_wind_onshore_generation_actual",
    "DE_wind_offshore_generation_actual",
    "price",   # unified column
]

def load_and_clean(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)

    # Merge the two price columns — critical fix for pre/post Oct 2018 zone split
    df["price"] = df["DE_LU_price_day_ahead"].fillna(df["AT_price_day_ahead"])
    cols_needed = FEATURE_COLS + ["price"]
    df = df[cols_needed]

    # Interpolate short gaps, forward-fill longer ones
    df = df.interpolate(method="linear", limit=3)
    df = df.ffill()
    df = df.bfill()
    df = df.fillna(0)

    # Filter to study period
    df = df["2015-01-01":"2020-06-30"]
    print(f"Shape after cleaning: {df.shape}")
    print(f"Nulls remaining:\n{df.isnull().sum()}")
    return df

def add_temporal_features(df):
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(float)
    return df

def split_and_scale(df):
    train = df[:"2018-12-31"]
    val   = df["2019-01-01":"2019-12-31"]
    test  = df["2020-01-01":]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)

    return (
        pd.DataFrame(train_scaled, index=train.index, columns=df.columns),
        pd.DataFrame(val_scaled,   index=val.index,   columns=df.columns),
        pd.DataFrame(test_scaled,  index=test.index,  columns=df.columns),
        scaler,
        df.columns.tolist()
    )

if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    df = add_temporal_features(df)
    train, val, test, scaler, cols = split_and_scale(df)

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "train": train, "val": val, "test": test,
            "scaler": scaler, "columns": cols,
            "feature_cols": FEATURE_COLS + [
                "hour_sin","hour_cos","dow_sin","dow_cos",
                "month_sin","month_cos","is_weekend"
            ],
            "target_cols": TARGET_COLS
        }, f)

    print(f"Saved processed data to {OUT_PATH}")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")