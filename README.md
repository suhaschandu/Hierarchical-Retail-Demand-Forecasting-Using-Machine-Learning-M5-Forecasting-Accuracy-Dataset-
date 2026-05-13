# Hierarchical Retail Demand Forecasting

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An advanced hierarchical demand forecasting pipeline designed for large-scale retail environments. This project was developed as part of the M5 Forecasting Competition (Walmart Sales), achieving a **WRMSSE score of 0.627**.

## 🚀 Overview

This repository implements a global-local blended architecture using LightGBM to forecast demand across 30,490 item-store combinations. The model is trained on over 59 million rows of historical sales data, incorporating complex feature engineering for seasonality, price dynamics, and special events.

### Key Metrics
- **WRMSSE Score:** 0.627
- **Series Modelled:** 30,490
- **Training Dataset:** 59M+ rows
- **Feature Set:** 48 engineered features

## 📁 Project Structure

```text
demand_forecast/
├── data/               # Dataset (not uploaded, see instructions below)
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
└── index.html          # Interactive dashboard frontend
```

## 📊 Dataset

The dataset used for this project is from the **M5 Forecasting - Accuracy** competition on Kaggle.

**Do not upload the raw data to this repository.**

### How to Download
1.  Visit the [M5 Forecasting - Accuracy Kaggle page](https://www.kaggle.com/c/m5-forecasting-accuracy/data).
2.  Download the following files:
    *   `calendar.csv`
    *   `sales_train_evaluation.csv`
    *   `sell_prices.csv`
3.  Place the downloaded `.csv` files into the `data/` directory.

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/demand_forecast.git
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

4.  **Launch the Dashboard:**
    ```bash
    python app.py
    ```
    Open `index.html` in your browser to view the interactive forecast results.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by Team D*
