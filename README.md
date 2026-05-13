# Hierarchical Retail Demand Forecasting

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An advanced hierarchical demand forecasting pipeline designed for large-scale retail environments. This project was developed as part of the M5 Forecasting Competition (Walmart Sales), achieving a **WRMSSE score of 0.627**.

## 🏗️ Architecture: Global-Local Blended Model

Our approach leverages a hybrid architecture to balance broad market trends with local store-specific nuances.

### 1. Global LightGBM Model
- **Purpose**: Learns high-level patterns across all 3,049 items and 10 stores.
- **Data**: Trained on the full 59M row dataset.
- **Strength**: Captures state-wide seasonality and cross-category price elasticities.

### 2. Local (Store × Category) Models
- **Purpose**: Fine-tuned specialized models for each Store-Category combination (e.g., CA_1 x FOODS).
- **Strength**: Captures regional store-specific spikes and local market dynamics.

### 3. The 50/50 Blend
The final prediction is an ensemble:
`Final_Prediction = (0.5 * Global_Pred) + (0.5 * Local_Pred)`
This strategy reduced our WRMSSE significantly compared to using a single global model.

---

## 📁 Project Structure

```text
demand_forecast/
├── data/               # Dataset (not uploaded)
├── models/             # Trained model objects (.joblib)
├── src/                # Source code
│   ├── data_loader.py  # Data ingestion and cleaning
│   ├── features.py     # Feature engineering pipeline
│   ├── model.py        # Model architecture (LightGBM)
│   ├── train.py        # Training orchestration
│   ├── predict.py      # Recursive forecasting logic
│   └── metrics.py      # WRMSSE evaluation
├── main.py             # Main entry point for the pipeline
├── app.py              # Interactive dashboard backend
└── docs/               # GitHub Pages Presentation
    └── index.html      # Premium landing page
```

---

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/suhaschandu/Hierarchical-Retail-Demand-Forecasting-Using-Machine-Learning-M5-Forecasting-Accuracy-Dataset-.git
    cd demand_forecast
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the pipeline:**
    ```bash
    python main.py
    ```

---

## 🧠 Feature Engineering (48 Total)

The model's accuracy is driven by a diverse set of engineered features:

| Category | Features |
| :--- | :--- |
| **Temporal** | Day of week, Month, Year, Event types (Cultural, National, Religious), SNAP benefit days |
| **Lags** | 7, 14, 21, and 28-day historical sales lags |
| **Rolling Windows** | 7, 14, and 30-day moving averages and standard deviations |
| **Price Dynamics** | Price change (weekly), Relative price to dept/cat average, Price max/min scaling |
| **Recursive** | Dynamic features updated day-by-day during the 28-day forecast horizon |

---

## 📈 Performance & Results

- **Metric**: WRMSSE (Weighted Root Mean Squared Scaled Error)
- **Score**: 0.627
- **Validation**: Time-series split (using the last 28 days of the evaluation set).

### Hierarchical Coverage
The model is optimized across all 12 levels of the M5 hierarchy:
1. Total Sales
2. State
3. Store
4. Category
5. Department
6. Item
...and their cross-combinations.

---

## 🌐 Live Presentation Site

We have created a premium, blog-style interactive presentation for this project.
**[View the Live Story & Interactive Charts →](https://tharunkm78.github.io/Hierarchical-Retail-Demand-Forecasting-Using-Machine-Learning-M5-Forecasting-Accuracy-Dataset/)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by Team D*
