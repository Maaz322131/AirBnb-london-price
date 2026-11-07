"""Train and evaluate nightly-price models on Inside Airbnb London listings.

Usage:   python src/train.py
Expects: data/listings.csv.gz (or data/listings.csv)
Outputs: models/price_model.joblib, reports/metrics.json, reports/choices.json, reports/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

NUM_COLS = ["accommodates", "bedrooms", "bathrooms", "beds", "minimum_nights",
            "availability_365", "number_of_reviews", "review_scores_rating",
            "latitude", "longitude", "amenities_count"]
CAT_COLS = ["room_type", "neighbourhood_cleansed", "property_type", "host_is_superhost"]


def find_data() -> Path:
    for name in ["listings.csv.gz", "listings.csv"]:
        p = ROOT / "data" / name
        if p.exists():
            return p
    raise SystemExit("Dataset not found. Put listings.csv.gz in the data/ folder (see README).")


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # price comes as text like "$1,250.00"
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                                errors="coerce")
    df = df.dropna(subset=["price"])
    # remove extreme outliers / data errors
    df = df[(df["price"] >= 10) & (df["price"] <= 1000)]
    # number of amenities from the list-like string
    df["amenities_count"] = df["amenities"].fillna("[]").str.count(",") + 1
    if 'bathrooms_text' in df.columns:
        text = df['bathrooms_text'].astype(str)
        parsed = text.str.extract(r'(\d+\.?\d*)')[0].astype(float)
        parsed[text.str.contains('half', case=False) & parsed.isna()] = 0.5
        df['bathrooms'] = df['bathrooms'].fillna(parsed) if 'bathrooms' in df.columns else parsed
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "unknown"
    df["host_is_superhost"] = df["host_is_superhost"].fillna("unknown").astype(str)
    return df[NUM_COLS + CAT_COLS + ["price"]].reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)], sparse_threshold=0)


def main():
    df = load_and_clean(find_data())
    print(f"Listings after cleaning: {len(df)} | median price: {df['price'].median():.0f}")
    X, y = df.drop(columns="price"), np.log1p(df["price"])  # log target tames skewed prices

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=8,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    # 1) Compare models with 5-fold CV on the training set only (R2 on log price)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, model in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        cv_scores[name] = {"mean_r2": float(s.mean()), "std_r2": float(s.std())}
        print(f"{name:18s} CV R2 = {s.mean():.3f} +/- {s.std():.3f}")

    # 2) Fit best model, evaluate once on the test set, in real price units
    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean_r2"])
    best = Pipeline([("prep", build_preprocessor()), ("model", candidates[best_name])])
    best.fit(X_train, y_train)
    pred = np.expm1(best.predict(X_test))
    actual = np.expm1(y_test)
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    r2 = r2_score(actual, pred)
    print(f"\nBest: {best_name} | Test MAE = {mae:.1f} per night | RMSE = {rmse:.1f} | R2 = {r2:.3f}")

    # 3) Save artefacts
    joblib.dump(best, ROOT / "models" / "price_model.joblib", compress=3)
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "cv": cv_scores, "test_mae": float(mae),
                   "test_rmse": rmse, "test_r2": float(r2),
                   "median_price": float(df["price"].median())}, f, indent=2)

    # choices for the app dropdowns (+ typical lat/lon per neighbourhood)
    nb = df.groupby("neighbourhood_cleansed")[["latitude", "longitude"]].median()
    choices = {
        "room_type": sorted(df["room_type"].dropna().unique().tolist()),
        "property_type": df["property_type"].value_counts().head(25).index.tolist(),
        "neighbourhoods": nb.round(5).to_dict(orient="index"),
        "defaults": {c: float(df[c].median()) for c in
                     ["number_of_reviews", "review_scores_rating", "availability_365"]},
    }
    with open(ROOT / "reports" / "choices.json", "w") as f:
        json.dump(choices, f, indent=2)

    # 4) Plots
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, pred, s=4, alpha=0.3)
    lim = [0, min(600, float(actual.max()))]
    ax.plot(lim, lim, "r--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual price per night"); ax.set_ylabel("Predicted price per night")
    ax.set_title(f"Actual vs predicted - {best_name}")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "actual_vs_predicted.png", dpi=150); plt.close(fig)

    r = permutation_importance(best, X_test, y_test, scoring="r2", n_repeats=3,
                               random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax)
    ax.set_xlabel("Drop in R2 when feature is shuffled")
    ax.set_title("Top 10 features driving price predictions")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["price"], bins=60); ax.set_xlabel("Price per night"); ax.set_ylabel("Listings")
    ax.set_title("Distribution of nightly prices (after cleaning)")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "price_distribution.png", dpi=150); plt.close(fig)
    print("Saved model, metrics and plots.")


if __name__ == "__main__":
    main()


