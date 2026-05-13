import numpy as np
import pandas as pd

import sys
import os
import joblib
from datetime import datetime

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

log_file = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

import os
os.makedirs("logs", exist_ok=True)

sys.stdout = Logger(log_file)

print(f"Logging to: {log_file}")


from src.data_loader import load_data, melt_sales, merge_data
from src.features import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_price_features,
    create_group_features,
    encode_categorical,
)
from src.train import split_data
from src.model import train_lgb
from src.predict import recursive_forecast, reconcile_predictions
from src.metrics import wrmsse

print("=" * 60)
print("  DEMAND FORECAST PIPELINE STARTING")
print("=" * 60)

# =========================
# LOAD OR PROCESS DATA
# =========================
print("\n[1/5] DATA LOADING & FEATURE ENGINEERING")
print("-" * 40)

try:
    print("  Attempting to load cached processed data...")
    df = pd.read_parquet("data/processed.parquet")
    print(f"  Loaded processed data — shape: {df.shape}")
except:
    print("  Processing raw data...\n")

    sales, calendar, prices = load_data()
    sales_long = melt_sales(sales)
    df = merge_data(sales_long, calendar, prices)

    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_price_features(df)
    df = create_group_features(df)
    df = encode_categorical(df)

    df.to_parquet("data/processed.parquet")
    print("  Saved processed data!")

# =========================
# SPLIT DATA
# =========================
print("\n[2/5] SPLITTING DATA")
print("-" * 40)

train, valid1, valid2 = split_data(df)

# =========================
# GLOBAL MODEL
# =========================
print("\n[3/5] TRAINING GLOBAL MODEL")
print("-" * 40)

global_model, features = train_lgb(train, valid1)

# =========================
# STORE + CATEGORY MODELS
# =========================
print("\n[4/5] TRAINING STORE × CAT MODELS")
print("-" * 40)

models = {}

stores = train["store_id"].unique()
cats = train["cat_id"].unique()

for store in stores:
    for cat in cats:

        combo = (store, cat)

        train_s = train[(train["store_id"] == store) & (train["cat_id"] == cat)]

        if len(train_s) < 1000:
            print(f"  [SKIP] store={store}, cat={cat} — too small ({len(train_s)})")
            continue

        valid1_s = valid1[(valid1["store_id"] == store) & (valid1["cat_id"] == cat)]
        valid2_s = valid2[(valid2["store_id"] == store) & (valid2["cat_id"] == cat)]

        if len(valid1_s) == 0 or len(valid2_s) == 0:
            print(f"  [SKIP] store={store}, cat={cat} — empty validation")
            continue

        print(f"\n  Training store={store}, cat={cat}")
        print(f"    Train: {len(train_s):,} | Valid1: {len(valid1_s):,} | Valid2: {len(valid2_s):,}")

        model1, _ = train_lgb(train_s, valid1_s)
        model2, _ = train_lgb(train_s, valid2_s)

        models[combo] = [model1, model2]

print("\n  Saving models to disk...")
os.makedirs("models", exist_ok=True)
joblib.dump(global_model, "models/global_model.pkl")
joblib.dump(models, "models/store_cat_models.pkl")
print("  Models saved successfully.")

# =========================
# PREDICTION
# =========================
print("\n[5/5] GENERATING VALIDATION PREDICTIONS")
print("-" * 40)

# 🔥 START WITH GLOBAL (fallback safety)
preds = global_model.predict(valid2[features])

for (store, cat) in models:

    idx = (
        (valid2["store_id"] == store) &
        (valid2["cat_id"] == cat)
    )

    if idx.sum() == 0:
        continue

    store_models = models[(store, cat)]

    local_pred = 0
    for m in store_models:
        local_pred += m.predict(valid2[idx][features])

    local_pred /= len(store_models)

    global_pred = preds[idx.values]

    # 🔥 BLEND
    blended = 0.5 * local_pred + 0.5 * global_pred

    preds[idx.values] = blended

print("\n  Post-processing predictions...")
preds = np.expm1(preds)
preds = np.maximum(0, preds)

print(f"  Prediction stats → min: {preds.min():.2f}, max: {preds.max():.2f}, mean: {preds.mean():.2f}")

# =========================
# FUTURE FORECAST
# =========================
print("\n[5b] FUTURE FORECAST")
print("-" * 40)

recent_df = df[df["date"] > df["date"].max() - pd.Timedelta(days=60)].copy()

future_list = []

for (store, cat) in models:

    model = models[(store, cat)][0]

    df_store = recent_df[
        (recent_df["store_id"] == store) &
        (recent_df["cat_id"] == cat)
    ].copy()

    if len(df_store) == 0:
        continue

    future_store = recursive_forecast(model, df_store, features)
    future_list.append(future_store)

future = pd.concat(future_list)
future = reconcile_predictions(future)

print(future.head())

# =========================
# EVALUATION
# =========================
print("\n[FINAL] WRMSSE")
print("-" * 40)

score = wrmsse(train, valid2, preds)

print("=" * 60)
print(f"  WRMSSE: {score:.5f}")
print("=" * 60)