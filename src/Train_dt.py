
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from src.config import FEATURES, TARGET
from src.preprocessing import clean_data, features_engineering, fill_missing_values, encode_categorical


def train_tree_model(df):

    # 1. Clean + feature engineering
    df = clean_data(df)
    df = features_engineering(df)

    # 2. Select features (same as XGB)
    X = df[
        FEATURES["surface_col"]
        + FEATURES["binary_col"]
        + FEATURES["count_col"]
        + FEATURES["distance_col"]
        + FEATURES["ordinal_col"]
        + FEATURES["geographical_col"]
    ]
    y = df[TARGET]

    # 3. Handle missing + encoding
    X = fill_missing_values(X)
    X = encode_categorical(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Pipeline (no scaler needed for trees)
    pipeline = Pipeline([
        ("model", DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ))
    ])

    #  6. Train
    pipeline.fit(X_train, y_train)

    #  7. Predict
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    print("r2_train:", r2_score(y_train, y_train_pred))
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    # 8. Save full pipeline (model + preprocessing logic already applied before)
    joblib.dump(pipeline, "model/model_tree.pkl")

    return pipeline


if __name__ == "__main__":
    df = pd.read_csv("data/listings_clean_duplicate_final.csv")
    train_tree_model(df)
