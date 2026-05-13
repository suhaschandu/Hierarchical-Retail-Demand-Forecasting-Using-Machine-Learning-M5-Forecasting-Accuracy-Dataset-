"""
app.py — Flask backend for Hierarchical Retail Demand Forecasting
-----------------------------------------------------------------
HOW TO RUN:
  1. Make sure you have run main.py first (saves models to models/)
  2. Install Flask if needed:  pip install flask
  3. In VS Code terminal, run:  python app.py
  4. Open your browser at:     http://127.0.0.1:5000
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

from src.predict import recursive_forecast, reconcile_predictions

app = Flask(__name__)

# ── LOAD MODELS ONCE AT STARTUP ──────────────────────────────
print("Loading models from disk...")

GLOBAL_MODEL_PATH = "models/global_model.pkl"
STORE_CAT_MODEL_PATH = "models/store_cat_models.pkl"
DATA_PATH = "data/processed.parquet"

if not os.path.exists(GLOBAL_MODEL_PATH):
    raise FileNotFoundError(
        "models/global_model.pkl not found. Please run main.py first."
    )

global_model = joblib.load(GLOBAL_MODEL_PATH)

# ── LOAD STORE/CAT MODELS SAFELY ─────────────────────────────
# Large pickle files can trigger MemoryError.
# Fallback to empty dict if loading fails.
try:
    store_cat_models = joblib.load(STORE_CAT_MODEL_PATH)
    print(f"Loaded {len(store_cat_models)} store/category models")
except MemoryError:
    print("\nWARNING: MemoryError while loading store_cat_models.pkl")
    print("Falling back to global model only.\n")
    store_cat_models = {}

# ── FEATURE LIST ─────────────────────────────────────────────
features = list(global_model.feature_name())

# Only load required columns to reduce RAM usage.
# "id" is required by recursive_forecast's groupby("id") calls.
# "item_id" is required for per-item filtering when the user
# provides an Item ID in the UI.
required_columns = list(set([
    "date",
    "store_id",
    "cat_id",
    "item_id",  # needed for per-item filtering
    "sales",
    "id",       # needed by recursive_forecast's groupby("id")
] + features))

# ── LOAD PROCESSED DATA ──────────────────────────────────────
print("Loading processed data...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "data/processed.parquet not found. Please run main.py first."
    )

df_all = pd.read_parquet(
    DATA_PATH,
    columns=required_columns
)

df_all["date"] = pd.to_datetime(df_all["date"])

# Reduce memory usage
for col in df_all.select_dtypes(include=["float64"]).columns:
    df_all[col] = df_all[col].astype("float32")

for col in df_all.select_dtypes(include=["int64"]).columns:
    df_all[col] = df_all[col].astype("int32")

gc.collect()

print(f"Processed data loaded — shape: {df_all.shape}")
print(f"Feature count: {len(features)}")

# ── REBUILD LABEL ENCODINGS FROM RAW CSV ─────────────────────
# encode_categorical() uses .astype("category").cat.codes which
# assigns integer codes in sorted alphabetical order.
# We reconstruct the same mapping here from the original CSV
# so the frontend's string values (e.g. "CA_1", "HOBBIES",
# "HOBBIES_1_001") can be converted to the integer codes stored
# in processed.parquet.
print("Rebuilding label encodings from sales_train.csv...")

_ids = pd.read_csv("data/sales_train.csv", usecols=["store_id", "cat_id", "item_id"])
store_to_code = {v: i for i, v in enumerate(sorted(_ids["store_id"].unique()))}
cat_to_code   = {v: i for i, v in enumerate(sorted(_ids["cat_id"].unique()))}
item_to_code  = {v: i for i, v in enumerate(sorted(_ids["item_id"].unique()))}
del _ids
gc.collect()

print(f"  Stores mapped: {store_to_code}")
print(f"  Cats mapped:   {cat_to_code}")
print(f"  Items mapped:  {len(item_to_code)} unique item IDs")
print("Models ready. Starting server...\n")


# ── SERVE HTML ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── FORECAST ENDPOINT ────────────────────────────────────────
@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        body = request.get_json()

        store   = body["store"]
        cat     = body["cat"]
        dept    = body["dept"]
        item_id = body.get("item", "").strip()

        start_date = pd.Timestamp(body["date"])
        horizon    = int(body["horizon"])

        # ── CONVERT STRING LABELS TO INTEGER CODES ───────────
        # processed.parquet stores store_id/cat_id/item_id as
        # integer codes (from encode_categorical). The frontend
        # sends plain strings, so we map them before filtering.
        store_code = store_to_code.get(store)
        cat_code   = cat_to_code.get(cat)

        if store_code is None or cat_code is None:
            return jsonify({
                "error": f"Unknown store={store} or cat={cat}. "
                         f"Valid stores: {list(store_to_code.keys())}, "
                         f"Valid cats: {list(cat_to_code.keys())}"
            }), 400

        # Validate item_id if provided
        item_code = None
        if item_id:
            item_code = item_to_code.get(item_id)
            if item_code is None:
                return jsonify({
                    "error": f"Unknown item_id={item_id}. "
                             f"Example valid IDs: {list(item_to_code.keys())[:5]}"
                }), 400

        # ── CONTEXT WINDOW ───────────────────────────────────
        recent_cutoff = start_date - pd.Timedelta(days=120)

        mask = (
            (df_all["store_id"] == store_code) &
            (df_all["cat_id"]   == cat_code)   &
            (df_all["date"]     >= recent_cutoff)
        )

        # If a specific item was requested, narrow down to that item only
        if item_code is not None:
            mask = mask & (df_all["item_id"] == item_code)

        df_ctx = df_all.loc[mask].copy()

        if len(df_ctx) == 0:
            label = f"store={store}, cat={cat}"
            if item_id:
                label += f", item={item_id}"
            return jsonify({"error": f"No data found for {label}"}), 400

        # ── MODEL SELECTION ─────────────────────────────────
        combo = (store, cat)

        if combo in store_cat_models:
            model = store_cat_models[combo][0]
        else:
            model = global_model

        # ── FORECAST ────────────────────────────────────────
        last_date = df_ctx["date"].max()
        n_ids     = df_ctx["id"].nunique()

        future_df = recursive_forecast(
            model,
            df_ctx,
            features,
            horizon=horizon
        )

        future_df = reconcile_predictions(future_df)

        # Reconstruct date column — predict.py strips it from
        # temp_df before appending to future_preds, so the
        # returned dataframe has no date column. Each of the
        # `horizon` steps produces exactly n_ids rows (one per
        # item), so we rebuild the dates here.
        future_df["date"] = [
            last_date + pd.Timedelta(days=d)
            for d in range(1, horizon + 1)
            for _ in range(n_ids)
        ]

        future_by_date = (
            future_df.groupby("date")["sales"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        forecast_vals  = future_by_date["sales"].tolist()
        forecast_dates = [d.strftime("%b %d") for d in future_by_date["date"]]

        # ── HISTORY ─────────────────────────────────────────
        hist_start = start_date - pd.Timedelta(days=28)

        hist_mask = (
            (df_all["store_id"] == store_code) &
            (df_all["cat_id"]   == cat_code)   &
            (df_all["date"]     >= hist_start)  &
            (df_all["date"]     < start_date)
        )

        if item_code is not None:
            hist_mask = hist_mask & (df_all["item_id"] == item_code)

        hist_df = df_all.loc[hist_mask].copy()

        hist_by_date = (
            hist_df.groupby("date")["sales"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        history_vals  = hist_by_date["sales"].tolist()
        history_dates = [d.strftime("%b %d") for d in hist_by_date["date"]]

        # ── VALIDATION PREDICTIONS ──────────────────────────
        predicted_vals = []

        if len(hist_df) > 0:
            if all(f in hist_df.columns for f in features):

                hist_feats = hist_df[features]
                raw_preds  = global_model.predict(hist_feats)
                preds_exp  = np.maximum(0, np.expm1(raw_preds))

                hist_df["pred"] = preds_exp

                pred_by_date = (
                    hist_df.groupby("date")["pred"]
                    .sum()
                    .reset_index()
                    .sort_values("date")
                )

                predicted_vals = pred_by_date["pred"].tolist()

        # ── STATS ───────────────────────────────────────────
        total     = int(sum(forecast_vals))
        daily_avg = round(total / max(horizon, 1), 1)
        peak      = max(forecast_vals) if forecast_vals else 0
        peak_idx  = forecast_vals.index(peak) if forecast_vals else 0
        peak_dt   = start_date + pd.Timedelta(days=peak_idx)
        peak_label = peak_dt.strftime("%b %d (%a)")

        return jsonify({
            "dates":     history_dates + forecast_dates,
            "history":   history_vals,
            "predicted": predicted_vals,
            "forecast":  forecast_vals,
            "total":     total,
            "daily_avg": daily_avg,
            "peak":      int(peak),
            "peak_date": peak_label,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── RUN SERVER ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Demand Forecast Server")
    print("  http://127.0.0.1:5000")
    print("=" * 50)

    # IMPORTANT:
    # debug=False prevents Flask from spawning
    # a second process that duplicates memory.
    app.run(
        debug=False,
        use_reloader=False,
        port=5000
    )