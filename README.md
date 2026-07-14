# Restaurant Sales ML Pipeline

This project trains machine-learning models on the provided restaurant order data for four tasks:

1. Sales forecasting: daily sales, weekly sales, store-wise revenue, and product-wise demand.
2. Product demand prediction: products expected to sell tomorrow and category trend direction.
3. Store performance analysis: best and low-performing stores, including revenue by location.
4. Customer segmentation: clustering customers by membership, city, purchase frequency, and spending.

## Setup

```powershell
python -m venv .venv_ml
.\.venv_ml\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

```powershell
.\.venv_ml\Scripts\python.exe src\train_models.py --data-dir source_data --output-dir outputs
```

## Main Outputs

- `outputs\models\*.joblib`: trained ML models and preprocessors.
- `outputs\metrics.json`: validation metrics for each supervised model and customer clustering.
- `outputs\predictions\tomorrow_product_demand.csv`: product demand predictions for the day after the latest order date.
- `outputs\predictions\tomorrow_store_revenue.csv`: store revenue predictions for the day after the latest order date.
- `outputs\reports\category_trends.csv`: growing or declining product categories.
- `outputs\reports\store_performance.csv`: store revenue ranking and performance labels.
- `outputs\reports\customer_segments.csv`: customer-level segment assignment and features.
