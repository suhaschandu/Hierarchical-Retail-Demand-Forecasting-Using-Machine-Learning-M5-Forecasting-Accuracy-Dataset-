import pandas as pd
import numpy as np


def recursive_forecast(model, df, features, horizon=28):

    df = df.sort_values(["id", "date"]).copy()
    last_date = df["date"].max()

    lag_features = sorted(
        [f for f in features if f.startswith("lag_")],
        key=lambda x: int(x.split("_")[1])
    )

    rmean_features = [f for f in features if f.startswith("rmean_")]
    rstd_features = [f for f in features if f.startswith("rstd_")]

    print(f"    Forecasting {horizon} days from {last_date.date()}")

    future_preds = []

    for i in range(1, horizon + 1):

        current_date = last_date + pd.Timedelta(days=i)

        temp_df = df.groupby("id").tail(1).copy()
        temp_df["date"] = current_date

        # Update time features for the new date
        temp_df["day"] = current_date.day
        temp_df["month"] = current_date.month
        temp_df["year"] = current_date.year
        temp_df["dayofweek"] = current_date.dayofweek
        temp_df["weekofyear"] = int(current_date.isocalendar().week)
        temp_df["is_weekend"] = int(current_date.dayofweek >= 5)
        temp_df["sin_dow"] = np.sin(2 * np.pi * current_date.dayofweek / 7)
        temp_df["cos_dow"] = np.cos(2 * np.pi * current_date.dayofweek / 7)
        temp_df["sin_month"] = np.sin(2 * np.pi * current_date.month / 12)
        temp_df["cos_month"] = np.cos(2 * np.pi * current_date.month / 12)

        # Drop features that will be re-merged to avoid _x and _y suffix
        cols_to_drop = lag_features + rmean_features + rstd_features
        temp_df = temp_df.drop(columns=[c for c in cols_to_drop if c in temp_df.columns])

        # =====================
        # LAG FEATURES
        # =====================
        for feat in lag_features:
            lag = int(feat.split("_")[1])

            lag_vals = (
                df.groupby("id")["sales"]
                .apply(lambda s: s.iloc[-lag] if len(s) >= lag else 0)
                .reset_index()
            )

            lag_vals.columns = ["id", feat]
            temp_df = temp_df.merge(lag_vals, on="id", how="left")

        # =====================
        # ROLLING MEAN
        # =====================
        for feat in rmean_features:
            w = int(feat.split("_")[1])

            vals = (
                df.groupby("id")["sales"]
                .apply(lambda s: s.shift(7).rolling(w).mean().iloc[-1] if len(s) > 7 else 0)
                .reset_index()
            )

            vals.columns = ["id", feat]
            temp_df = temp_df.merge(vals, on="id", how="left")

        # =====================
        # ROLLING STD
        # =====================
        for feat in rstd_features:
            w = int(feat.split("_")[1])

            vals = (
                df.groupby("id")["sales"]
                .apply(lambda s: s.shift(7).rolling(w).std().iloc[-1] if len(s) > 7 else 0)
                .reset_index()
            )

            vals.columns = ["id", feat]
            temp_df = temp_df.merge(vals, on="id", how="left")

        # =====================
        # 🔥 CRITICAL FIX
        # =====================
        missing = [c for c in features if c not in temp_df.columns]
        if missing:
            print(f"      ⚠ Missing {len(missing)} features → auto-filled")

        for col in features:
            if col not in temp_df.columns:
                temp_df[col] = 0

        temp_df = temp_df[features].fillna(0)

        # =====================
        # PREDICTION
        # =====================
        raw_preds = model.predict(temp_df)

        preds = np.expm1(raw_preds)
        preds = np.clip(preds, 0, 50)  # prevent explosion

        temp_df["sales"] = preds

        # =====================
        # UPDATE DF (RECURSION)
        # =====================
        df = pd.concat([df, temp_df.assign(date=current_date)], ignore_index=True)

        future_preds.append(temp_df)

        if i % 7 == 0 or i == horizon:
            print(f"      Day {i}/{horizon} done — avg={preds.mean():.2f}")

    return pd.concat(future_preds, ignore_index=True)


def reconcile_predictions(df):
    print("    Final clipping + rounding...")
    df["sales"] = np.maximum(0, df["sales"]).round().astype(int)
    return df