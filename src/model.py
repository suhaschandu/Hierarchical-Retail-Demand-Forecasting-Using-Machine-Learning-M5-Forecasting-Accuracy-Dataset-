import lightgbm as lgb
import numpy as np


def train_lgb(train, valid):

    # All columns except the target and date are features.
    # "id" is kept as an encoded integer feature for LightGBM.
    features = [
    col for col in train.columns
    if col not in ["sales", "date", "id"]
    ]

    X_train = train[features]
    y_train = np.log1p(train["sales"])

    X_valid = valid[features]
    y_valid = np.log1p(valid["sales"])

    print(f"      Feature count: {len(features)}")
    print(f"      Train rows: {len(X_train):,}  |  Valid rows: {len(X_valid):,}")

    params = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "tweedie",

    "learning_rate": 0.01,
    "num_leaves": 1024,
    "min_data_in_leaf": 50,

    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,

    "lambda_l1": 0.1,
    "lambda_l2": 0.1,

    "verbosity": -1,
    }

    print("      Fitting LightGBM (up to 3000 rounds, early stop @ 100)...")

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(500),
        ]
    )

    print(f"      Best iteration: {model.best_iteration}  |  Best score: {model.best_score['valid_0']['tweedie']:.5f}")

    return model, features