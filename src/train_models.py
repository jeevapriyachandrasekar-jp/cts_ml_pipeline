from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "models": output_dir / "models",
        "predictions": output_dir / "predictions",
        "reports": output_dir / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_source_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["orders", "order_items", "products", "stores", "customers"]:
        tables[name] = pd.read_csv(data_dir / f"{name}.csv")

    tables["orders"]["order_date"] = pd.to_datetime(tables["orders"]["order_date"])
    tables["customers"]["signup_date"] = pd.to_datetime(tables["customers"]["signup_date"])
    tables["stores"]["opened_date"] = pd.to_datetime(tables["stores"]["opened_date"])
    return tables


def build_sales_fact(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"]
    order_items = tables["order_items"]
    products = tables["products"]
    stores = tables["stores"]
    customers = tables["customers"]

    fact = (
        order_items.merge(orders, on="order_id", how="left")
        .merge(products, on="product_id", how="left", suffixes=("", "_product"))
        .merge(stores, on="store_id", how="left", suffixes=("", "_store"))
        .merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
    )

    fact["revenue"] = fact["quantity"] * fact["unit_price"]
    fact["order_date"] = pd.to_datetime(fact["order_date"])
    return fact


def add_date_features(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    result = df.copy()
    dates = pd.to_datetime(result[date_col])
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    result["year"] = dates.dt.year
    result["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    return result


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str] | None = None,
    lags: Iterable[int] = (1, 7, 14),
    rolling_windows: Iterable[int] = (7, 14),
) -> pd.DataFrame:
    result = df.copy().sort_values((group_cols or []) + ["order_date"])
    if group_cols:
        grouped = result.groupby(group_cols, sort=False)[target_col]
        for lag in lags:
            result[f"{target_col}_lag_{lag}"] = grouped.shift(lag)
        for window in rolling_windows:
            result[f"{target_col}_rolling_{window}"] = grouped.transform(lambda values: values.shift(1).rolling(window).mean())
    else:
        for lag in lags:
            result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)
        for window in rolling_windows:
            result[f"{target_col}_rolling_{window}"] = result[target_col].shift(1).rolling(window).mean()
    return result


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    try:
        rmse = root_mean_squared_error(y_true, y_pred)
    except NameError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train_temporal_regressor(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model,
    model_path: Path,
) -> tuple[Pipeline, dict[str, float]]:
    df = df.dropna(subset=[target]).copy()
    cutoff = df["order_date"].quantile(0.8)
    train_df = df[df["order_date"] <= cutoff]
    test_df = df[df["order_date"] > cutoff]

    if test_df.empty:
        split_index = max(1, int(len(df) * 0.8))
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]), categorical_features),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(train_df[numeric_features + categorical_features], train_df[target])

    predictions = pipeline.predict(test_df[numeric_features + categorical_features])
    metrics = regression_metrics(test_df[target], predictions)
    joblib.dump(pipeline, model_path)
    return pipeline, metrics


def train_sales_forecasts(fact: pd.DataFrame, paths: dict[str, Path]) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
    models = {}
    metrics = {}

    daily = (
        fact.groupby("order_date", as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), orders=("order_id", "nunique"))
    )
    daily = add_date_features(add_lag_features(daily, "revenue"))
    daily_numeric = [
        "quantity",
        "orders",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "revenue_lag_1",
        "revenue_lag_7",
        "revenue_lag_14",
        "revenue_rolling_7",
        "revenue_rolling_14",
    ]
    models["daily_sales"], metrics["daily_sales"] = train_temporal_regressor(
        daily,
        "revenue",
        daily_numeric,
        [],
        GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=250, learning_rate=0.05, max_depth=3),
        paths["models"] / "daily_sales_model.joblib",
    )

    weekly = daily.set_index("order_date").resample("W-MON").agg({"revenue": "sum", "quantity": "sum", "orders": "sum"}).reset_index()
    weekly = add_date_features(add_lag_features(weekly, "revenue", lags=(1, 2, 4), rolling_windows=(4, 8)))
    weekly_numeric = [
        "quantity",
        "orders",
        "month",
        "quarter",
        "year",
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_4",
        "revenue_rolling_4",
        "revenue_rolling_8",
    ]
    models["weekly_sales"], metrics["weekly_sales"] = train_temporal_regressor(
        weekly,
        "revenue",
        weekly_numeric,
        [],
        RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
        paths["models"] / "weekly_sales_model.joblib",
    )

    store_daily = (
        fact.groupby(["order_date", "store_id", "store_name", "city", "state", "store_type"], as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), orders=("order_id", "nunique"))
    )
    store_daily = add_date_features(add_lag_features(store_daily, "revenue", group_cols=["store_id"]))
    store_numeric = [
        "quantity",
        "orders",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "revenue_lag_1",
        "revenue_lag_7",
        "revenue_lag_14",
        "revenue_rolling_7",
        "revenue_rolling_14",
    ]
    store_categorical = ["store_id", "city", "state", "store_type"]
    models["store_revenue"], metrics["store_revenue"] = train_temporal_regressor(
        store_daily,
        "revenue",
        store_numeric,
        store_categorical,
        RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
        paths["models"] / "store_revenue_model.joblib",
    )

    product_daily = (
        fact.groupby(["order_date", "product_id", "product_name", "category"], as_index=False)
        .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    )
    product_daily = add_date_features(add_lag_features(product_daily, "quantity", group_cols=["product_id"]))
    product_numeric = [
        "revenue",
        "orders",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "quantity_lag_1",
        "quantity_lag_7",
        "quantity_lag_14",
        "quantity_rolling_7",
        "quantity_rolling_14",
    ]
    product_categorical = ["product_id", "category"]
    models["product_demand"], metrics["product_demand"] = train_temporal_regressor(
        product_daily,
        "quantity",
        product_numeric,
        product_categorical,
        RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
        paths["models"] / "product_demand_model.joblib",
    )

    make_next_day_predictions(fact, models["store_revenue"], models["product_demand"], paths)
    return models, metrics


def latest_group_features(
    grouped_daily: pd.DataFrame,
    entity_cols: list[str],
    target_col: str,
    prediction_date: pd.Timestamp,
) -> pd.DataFrame:
    history = add_lag_features(grouped_daily, target_col, group_cols=[entity_cols[0]])
    latest = history.sort_values("order_date").groupby(entity_cols[0], as_index=False).tail(1).copy()
    latest["order_date"] = prediction_date
    latest = add_date_features(latest)
    return latest


def make_next_day_predictions(
    fact: pd.DataFrame,
    store_model: Pipeline,
    product_model: Pipeline,
    paths: dict[str, Path],
) -> None:
    prediction_date = fact["order_date"].max() + pd.Timedelta(days=1)

    store_daily = (
        fact.groupby(["order_date", "store_id", "store_name", "city", "state", "store_type"], as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), orders=("order_id", "nunique"))
    )
    store_next = latest_group_features(store_daily, ["store_id", "store_name", "city", "state", "store_type"], "revenue", prediction_date)
    store_features = store_model.named_steps["preprocess"].feature_names_in_
    store_next["predicted_revenue"] = np.maximum(0, store_model.predict(store_next[list(store_features)]))
    store_next[
        ["order_date", "store_id", "store_name", "city", "state", "store_type", "predicted_revenue"]
    ].sort_values("predicted_revenue", ascending=False).to_csv(paths["predictions"] / "tomorrow_store_revenue.csv", index=False)

    product_daily = (
        fact.groupby(["order_date", "product_id", "product_name", "category"], as_index=False)
        .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    )
    product_next = latest_group_features(product_daily, ["product_id", "product_name", "category"], "quantity", prediction_date)
    product_features = product_model.named_steps["preprocess"].feature_names_in_
    product_next["predicted_quantity"] = np.maximum(0, product_model.predict(product_next[list(product_features)]))
    product_next[
        ["order_date", "product_id", "product_name", "category", "predicted_quantity"]
    ].sort_values("predicted_quantity", ascending=False).to_csv(paths["predictions"] / "tomorrow_product_demand.csv", index=False)


def analyze_category_trends(fact: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    max_date = fact["order_date"].max()
    current_start = max_date - pd.Timedelta(days=29)
    previous_start = current_start - pd.Timedelta(days=30)

    category_daily = fact.groupby(["order_date", "category"], as_index=False).agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"))
    current = category_daily[category_daily["order_date"].between(current_start, max_date)].groupby("category").sum(numeric_only=True)
    previous = category_daily[category_daily["order_date"].between(previous_start, current_start - pd.Timedelta(days=1))].groupby("category").sum(numeric_only=True)

    trends = current.join(previous, how="outer", lsuffix="_last_30d", rsuffix="_prev_30d").fillna(0).reset_index()
    trends["quantity_growth_pct"] = np.where(
        trends["quantity_prev_30d"] > 0,
        (trends["quantity_last_30d"] - trends["quantity_prev_30d"]) / trends["quantity_prev_30d"] * 100,
        np.nan,
    )
    trends["revenue_growth_pct"] = np.where(
        trends["revenue_prev_30d"] > 0,
        (trends["revenue_last_30d"] - trends["revenue_prev_30d"]) / trends["revenue_prev_30d"] * 100,
        np.nan,
    )
    trends["trend"] = np.select(
        [trends["quantity_growth_pct"] > 5, trends["quantity_growth_pct"] < -5],
        ["Growing", "Declining"],
        default="Stable",
    )
    trends.sort_values("quantity_growth_pct", ascending=False).to_csv(report_path, index=False)
    return trends


def analyze_store_performance(fact: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    store_perf = (
        fact.groupby(["store_id", "store_name", "city", "state", "store_type"], as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), orders=("order_id", "nunique"), customers=("customer_id", "nunique"))
    )
    store_perf["avg_order_revenue"] = store_perf["revenue"] / store_perf["orders"]
    high = store_perf["revenue"].quantile(0.75)
    low = store_perf["revenue"].quantile(0.25)
    store_perf["performance_label"] = np.select(
        [store_perf["revenue"] >= high, store_perf["revenue"] <= low],
        ["Best-performing", "Low-performing"],
        default="Middle-performing",
    )
    store_perf.sort_values("revenue", ascending=False).to_csv(report_path, index=False)
    return store_perf


def segment_customers(tables: dict[str, pd.DataFrame], fact: pd.DataFrame, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, float | int]]:
    customers = tables["customers"]
    max_date = fact["order_date"].max()
    order_level = (
        fact.groupby(["order_id", "customer_id", "order_date"], as_index=False)
        .agg(order_revenue=("revenue", "sum"), order_quantity=("quantity", "sum"))
    )
    customer_features = (
        order_level.groupby("customer_id", as_index=False)
        .agg(
            purchase_frequency=("order_id", "nunique"),
            spending=("order_revenue", "sum"),
            avg_order_value=("order_revenue", "mean"),
            total_quantity=("order_quantity", "sum"),
            last_order_date=("order_date", "max"),
        )
        .merge(customers[["customer_id", "city", "state", "membership_type", "signup_date"]], on="customer_id", how="right")
    )
    customer_features[["purchase_frequency", "spending", "avg_order_value", "total_quantity"]] = customer_features[
        ["purchase_frequency", "spending", "avg_order_value", "total_quantity"]
    ].fillna(0)
    customer_features["recency_days"] = (max_date - customer_features["last_order_date"]).dt.days.fillna(999)
    customer_features["account_age_days"] = (max_date - customer_features["signup_date"]).dt.days.clip(lower=0).fillna(0)

    numeric = ["purchase_frequency", "spending", "avg_order_value", "total_quantity", "recency_days", "account_age_days"]
    categorical = ["membership_type", "city", "state"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]), categorical),
        ]
    )
    matrix = preprocessor.fit_transform(customer_features[numeric + categorical])

    best_k = 3
    best_score = -1.0
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_k = k
            best_score = score

    clusterer = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    customer_features["segment"] = clusterer.fit_predict(matrix)

    segment_summary = customer_features.groupby("segment", as_index=False).agg(
        customers=("customer_id", "count"),
        avg_frequency=("purchase_frequency", "mean"),
        avg_spending=("spending", "mean"),
        avg_recency_days=("recency_days", "mean"),
    )
    segment_summary.to_csv(paths["reports"] / "customer_segment_summary.csv", index=False)
    customer_features.drop(columns=["last_order_date", "signup_date"]).to_csv(paths["reports"] / "customer_segments.csv", index=False)
    joblib.dump({"preprocessor": preprocessor, "model": clusterer, "features": numeric + categorical}, paths["models"] / "customer_segmentation_model.joblib")

    return customer_features, {"k": int(best_k), "silhouette": round(float(best_score), 4)}


def run(data_dir: Path, output_dir: Path) -> None:
    paths = ensure_dirs(output_dir)
    tables = load_source_data(data_dir)
    fact = build_sales_fact(tables)

    _, metrics = train_sales_forecasts(fact, paths)
    analyze_category_trends(fact, paths["reports"] / "category_trends.csv")
    analyze_store_performance(fact, paths["reports"] / "store_performance.csv")
    _, segmentation_metrics = segment_customers(tables, fact, paths)
    metrics["customer_segmentation"] = segmentation_metrics

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Training complete. Outputs saved to: {output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train restaurant sales forecasting, demand, performance, and customer segmentation models.")
    parser.add_argument("--data-dir", type=Path, default=Path("source_data"), help="Directory containing source CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for models, predictions, reports, and metrics.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir)
