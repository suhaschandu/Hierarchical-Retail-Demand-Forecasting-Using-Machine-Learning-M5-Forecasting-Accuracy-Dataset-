# Hierarchical Retail Demand Forecasting Using Machine Learning

This project builds a **hierarchical retail demand forecasting pipeline** on the **M5 Forecasting Accuracy** dataset. The goal is to predict **daily unit sales** for Walmart products across item, department, category, store, and state levels using **feature engineering + LightGBM**.

The repository focuses on:
- exploratory data analysis (EDA)
- large-scale time-series feature engineering
- demand forecasting with gradient boosting
- evaluation using **RMSE** and **WRMSSE**
- reproducible modeling for a DS 606 team project

## Team
- Amrutha Kakumani
- Sitaram Suhas Chandu
- Molapally Tharun Kumar

## Project Context
Retail demand forecasting is challenging because demand changes with:
- weekly seasonality
- holidays and events
- SNAP benefit days
- price fluctuations
- zero-inflated sales patterns

Using the **M5 Forecasting Accuracy** dataset, this project models sales for:
- **30,490 items**
- **10 Walmart stores**
- **3 states**: CA, TX, WI
- **1,941 days** of historical data

The forecasting task is a **time-series regression** problem with a **28-day forecast horizon**.

## Dataset
Source: **Kaggle M5 Forecasting Accuracy**

Files used in this phase:
- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`

After reshaping and preprocessing:
- Raw rows after melt: **58,327,370**
- Valid rows after feature engineering: **40,352,373**
- Training rows: **39,498,653**
- Validation rows: **853,720**

## Project Workflow

### 1. Data Loading
The notebook loads the three main M5 files and prepares them for modeling.

### 2. Reshaping Sales Data
The original wide sales table is converted into long format using `pd.melt()` so that each row represents:
- item-store-day combination
- sales value for that day

### 3. Merging External Context
The long sales table is merged with:
- calendar features
- weekly sell prices

This creates a unified modeling table containing demand, date, price, and event information.

### 4. Feature Engineering
The model uses **35 engineered features** across several groups.

#### Lag Features
- `lag_7`
- `lag_14`
- `lag_21`
- `lag_28`
- `lag_56`
- `lag_365`

#### Rolling Statistics
- `rmean_7`
- `rmean_14`
- `rmean_28`
- `rmean_56`
- `rstd_28`
- `rolling_trend`

#### Price Features
- `sell_price`
- `price_change`
- `price_norm`
- `price_roll_mean_7`
- `price_momentum`

#### Time Features
- `year`
- `month`
- `week`
- `day`
- `momentum`

#### Calendar / Event Features
- `event_name_1`
- `event_type_1`
- `event_name_2`
- `event_type_2`
- `snap_CA`
- `snap_TX`
- `snap_WI`

#### Categorical Identifiers
- `item_id`
- `dept_id`
- `cat_id`
- `store_id`
- `state_id`
- `sales_diff`

All lag and rolling features are computed **within each item series** to avoid time leakage.

## Exploratory Data Analysis Highlights
Key findings from EDA:
- **Strong weekly seasonality** with weekend demand peaks
- **Event and SNAP spikes** that increase sales substantially
- **Zero inflation (~25%)**, making standard regression less suitable
- **Price sensitivity**, where lower prices often correspond to demand increases

Additional dataset structure findings:
- Categories: **FOODS, HOUSEHOLD, HOBBIES**
- Store distribution across states:
  - CA: 4 stores
  - TX: 3 stores
  - WI: 3 stores

## Train / Validation Strategy
A strict **temporal split** is used.

- Training period: all dates up to the final 28 days
- Validation period: **last 28 days**

This matches the competition setup and avoids leakage from future observations.

Other choices:
- no random shuffle
- no standard k-fold cross-validation
- target transformed with `log1p(sales)`

## Model
The main model is **LightGBM** with a **Tweedie objective**, chosen because it handles:
- large datasets efficiently
- mixed numeric and categorical inputs
- zero-inflated positive demand values

### Hyperparameters
- Objective: `tweedie`
- Tweedie variance power: `1.1`
- Metric: `rmse`
- Learning rate: `0.03`
- Num leaves: `1024`
- Min data in leaf: `100`
- Feature fraction: `0.8`
- Bagging fraction: `0.8`
- Bagging frequency: `1`
- L1 regularization: `0.1`
- L2 regularization: `0.1`
- Max boosting rounds: `3500`
- Early stopping patience: `300`

### Best Iteration
- Best round: **184**

## Results
### Validation Performance
- **Validation RMSE (log-space): 0.487**
- **Validation RMSE (original scale): 2.153**
- **WRMSSE: 0.861**

### Training Convergence
- Round 100: 0.4892
- Round 184: 0.4874 ← best
- Round 300: 0.4876
- Round 400: 0.4880 → early stop region

## Why WRMSSE Matters
The M5 competition uses **WRMSSE** instead of plain RMSE because it:
- normalizes for scale differences between slow-moving and fast-moving items
- weights errors by business value / revenue share
- provides direct comparability with Kaggle competition benchmarks

Our project reports both:
- **RMSE** for model monitoring
- **WRMSSE** for business-aligned evaluation

## Feature Importance
Top signals identified by LightGBM:
1. `lag_28`
2. `rmean_28`
3. `lag_7`
4. `rmean_7`
5. `sell_price`
6. `lag_14`
7. `rmean_14`
8. `lag_56`
9. `item_id`
10. `month`

This shows that:
- monthly purchasing cycles matter strongly
- weekly seasonality is highly predictive
- price information meaningfully affects demand
- temporal patterns improve forecast quality

## Repository Structure
```text
.
├── P2.ipynb                 # Phase 2 notebook: EDA + feature engineering + LightGBM
├── README.md                # Project overview and instructions
├── data/                    # Place dataset files here (not included in repo)
│   ├── calendar.csv
│   ├── sell_prices.csv
│   └── sales_train_validation.csv
```

## How to Run
### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
```

### 3. Download the M5 dataset
Download the dataset from Kaggle and place these files in your local `data/` folder:
- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`

### 4. Update the data path
In the notebook, replace the current Google Drive path with your local or mounted path, for example:
```python
DATA_PATH = "./data/"
```

### 5. Run the notebook
Open `P2.ipynb` in:
- Jupyter Notebook
- JupyterLab
- Google Colab

Run all cells in order.

## Notebook Coverage
The notebook currently includes:
- imports and setup
- dataset loading
- memory optimization for numeric columns
- long-format transformation
- merge with calendar and sell prices
- date features
- lag and rolling demand features
- price-based features
- categorical encoding preparation
- train/validation split
- LightGBM training
- RMSE calculation
- feature importance plotting
- custom WRMSSE implementation

## Current Limitations
- baseline models are not yet fully integrated in the notebook
- hierarchical reconciliation is planned for the next phase
- hyperparameter tuning can still improve WRMSSE
- notebook path is currently set for Google Drive / Colab
- full competition-level feature set is intentionally reduced for interpretability and efficiency

## Next Steps
Planned improvements for Phase 3:
1. add Naive and Moving Average baselines
2. apply hierarchical reconciliation methods
3. perform hyperparameter tuning
4. push WRMSSE closer to leaderboard range
5. run deeper error analysis by item-store combinations
6. improve reproducibility and documentation

## References
1. Kaggle. *M5 Forecasting Accuracy*.  
2. Makridakis, S., Spiliotis, E., Assimakopoulos, V., & Nikolopoulos, K. (2020). *The M5 Accuracy Competition: Results, Findings and Conclusions*. International Journal of Forecasting, 36(1), 44–74.  
3. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.).  
4. Microsoft. *LightGBM Documentation*.  
5. Januschowski, T. et al. (2020). *Criteria for classifying forecasting methods*. International Journal of Forecasting, 36(1), 167–177.

## Acknowledgments
This project was developed for **DS 606** as part of a team-based forecasting and machine learning project using the M5 competition dataset.
