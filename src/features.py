import pandas as pd
import numpy as np


def create_time_features(df):
    print("    Parsing dates and extracting time components...")
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int32")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    # 🔥 NEW: cyclical features
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    print("    Added cyclical time features")
    return df


def create_lag_features(df):
    # 🔥 NEW short-term lags added
    lags = [1, 2, 3, 7, 14, 28, 56, 112]
    print(f"    Computing lag features: {lags}")

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

    return df

    na_counts = df[[f"lag_{l}" for l in lags]].isna().sum().sum()
    print(f"    Lag features complete — total NaNs introduced (expected): {na_counts:,}")
    return df


def create_rolling_features(df):
    windows = [7, 14, 30]
    print(f"    Computing rolling mean/std features for windows: {windows}")

    for w in windows:
        print(f"      Window {w}d — mean and std...")
        df[f"rmean_{w}"] = (
            df.groupby("id")["sales"]
            .shift(7)
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"rstd_{w}"] = (
            df.groupby("id")["sales"]
            .shift(7)
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )

    print(f"    Rolling features complete: rmean/rstd for windows {windows}")
    return df


def create_price_features(df):
    print("    Computing price features...")

    df["price_norm"] = df["sell_price"] / df.groupby("id")["sell_price"].transform("mean")
    print("      price_norm done")

    df["price_change_1"] = df.groupby("id")["sell_price"].pct_change(1)
    df["price_change_7"] = df.groupby("id")["sell_price"].pct_change(7)
    df["price_change_28"] = df.groupby("id")["sell_price"].pct_change(28)
    print("      price_change (1, 7, 28 days) done")

    print("    Price features complete.")
    return df


def create_group_features(df, train_end_date=None):
    """
    Compute group-level average sales.

    To avoid data leakage, averages are computed only on rows up to
    `train_end_date` (the last training date) and then joined back onto
    all rows. If `train_end_date` is not provided, the function falls back
    to using the full dataframe — only acceptable for exploratory runs.
    """
    print("    Computing group-level average features (store, dept, cat)...")

    if train_end_date is not None:
        print(f"      Using train-only window (up to {train_end_date}) to prevent leakage...")
        train_mask = df["date"] <= pd.Timestamp(train_end_date)
        agg_source = df[train_mask]
    else:
        print("      WARNING: train_end_date not provided — using full dataset (may leak future data).")
        agg_source = df

    store_avg = agg_source.groupby("store_id")["sales"].mean().rename("store_avg")
    dept_avg = agg_source.groupby("dept_id")["sales"].mean().rename("dept_avg")
    cat_avg = agg_source.groupby("cat_id")["sales"].mean().rename("cat_avg")

    df = df.merge(store_avg, on="store_id", how="left")
    df = df.merge(dept_avg, on="dept_id", how="left")
    df = df.merge(cat_avg, on="cat_id", how="left")

    print("      store_avg, dept_avg, cat_avg joined.")
    print("    Group features complete.")
    return df


def encode_categorical(df):
    cat_cols = [
        "item_id", "store_id", "dept_id", "cat_id", "state_id",
        "weekday",
        "event_name_1", "event_type_1",
        "event_name_2", "event_type_2",
    ]

    print(f"    Encoding categorical columns: {cat_cols}")

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    if "d" in df.columns:
        print("    Converting 'd' to numeric...")
        df["d"] = df["d"].str.replace("d_", "").astype("int16")

    print("    Categorical encoding complete.")
    return df