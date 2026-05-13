import numpy as np


def rmsse(y_true, y_pred, y_train):
    """
    Root Mean Squared Scaled Error for a single time series.
    Scaling denominator uses only non-zero consecutive differences
    to avoid division-by-zero on flat/intermittent series.
    """
    if len(y_train) < 2:
        return 0.0

    diff = y_train[1:] - y_train[:-1]
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    scale = np.mean(diff ** 2)
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / scale)


def wrmsse(train_df, valid_df, preds, verbose=True):
    """
    Weighted Root Mean Squared Scaled Error across all item series.
    Weights are proportional to each item's total sales in the last
    28 days of training (matching the M5 competition definition).
    """
    valid_df = valid_df.copy()
    valid_df["preds"] = preds

    ids = valid_df["id"].unique()
    total_ids = len(ids)

    if verbose:
        print(f"  Computing RMSSE for {total_ids:,} item series...")

    scores = []
    weights = []
    skipped = 0

    for i, item_id in enumerate(ids):
        train_item = train_df[train_df["id"] == item_id]["sales"].values
        valid_item = valid_df[valid_df["id"] == item_id]

        if len(train_item) < 2 or len(valid_item) == 0:
            skipped += 1
            continue

        y_true = valid_item["sales"].values
        y_pred = valid_item["preds"].values

        score = rmsse(y_true, y_pred, train_item)
        weight = train_item[-28:].sum()

        scores.append(score)
        weights.append(weight)

        if verbose and (i + 1) % max(1, total_ids // 5) == 0:
            print(f"    Processed {i + 1:,}/{total_ids:,} series...")

    if verbose:
        print(f"  Series scored: {len(scores):,}  |  Skipped (too short / empty): {skipped:,}")

    weights = np.array(weights, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)

    if weights.sum() == 0:
        if verbose:
            print("  WARNING: All weights are zero — returning unweighted mean.")
        return float(np.mean(scores))

    weights /= weights.sum()
    final_score = float(np.sum(scores * weights))

    if verbose:
        print(f"  Score breakdown — mean RMSSE: {scores.mean():.5f}  |  weighted WRMSSE: {final_score:.5f}")

    return final_score