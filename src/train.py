import pandas as pd


def split_data(df):
    """
    Creates three temporal splits:
      - train  : everything before the last 56 days
      - valid1 : days 57–28 before end  (used for model1 early-stopping)
      - valid2 : last 28 days           (held-out evaluation / model2 early-stopping)
    """
    max_date = df["date"].max()

    split1 = max_date - pd.Timedelta(days=56)
    split2 = max_date - pd.Timedelta(days=28)

    train = df[df["date"] <= split1].copy()
    valid1 = df[(df["date"] > split1) & (df["date"] <= split2)].copy()
    valid2 = df[df["date"] > split2].copy()

    print(f"  Date range   : {df['date'].min().date()} → {max_date.date()}")
    print(f"  Train        : up to {split1.date()}  — {len(train):,} rows")
    print(f"  Valid1       : {(split1 + pd.Timedelta(days=1)).date()} → {split2.date()}  — {len(valid1):,} rows")
    print(f"  Valid2       : {(split2 + pd.Timedelta(days=1)).date()} → {max_date.date()}  — {len(valid2):,} rows")

    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    train_nans = train[lag_cols].isna().mean().mean()
    print(f"  Mean NaN rate in train lag features: {100 * train_nans:.1f}%  (rows at series start — expected)")

    return train, valid1, valid2