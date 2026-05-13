import pandas as pd


def load_data():
    print("  Loading sales_train.csv...")
    sales = pd.read_csv("data/sales_train.csv")
    print(f"    Sales shape: {sales.shape}")

    print("  Loading calendar.csv...")
    calendar = pd.read_csv("data/calendar.csv")
    print(f"    Calendar shape: {calendar.shape}")

    print("  Loading sell_prices.csv...")
    prices = pd.read_csv("data/sell_prices.csv")
    print(f"    Prices shape: {prices.shape}")

    print("  All raw files loaded successfully.")
    return sales, calendar, prices


def melt_sales(sales):
    id_cols = ["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    print(f"    Melting {len(day_cols)} day columns into long format...")
    sales_long = sales.melt(
        id_vars=id_cols,
        var_name="d",
        value_name="sales"
    )

    print(f"    Melted shape: {sales_long.shape}")
    return sales_long


def merge_data(sales_long, calendar, prices):
    print("    Merging calendar onto sales...")
    df = sales_long.merge(calendar, on="d", how="left")
    print(f"    After calendar merge: {df.shape}")

    print("    Merging sell prices onto sales...")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    print(f"    After price merge: {df.shape}")

    missing_prices = df["sell_price"].isna().sum()
    print(f"    Missing sell_price values: {missing_prices:,} ({100 * missing_prices / len(df):.1f}%)")

    return df