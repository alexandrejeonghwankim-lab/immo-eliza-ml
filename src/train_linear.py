
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, TARGET
from src.preprocessing import clean_data, features_engineering, fill_missing_values, encode_categorical


def train_linear_model(df):

    # ✅ 1. Clean data
    df = clean_data(df)
    df = features_engineering(df)

    # ✅ 2. Select features
    X = df[
        FEATURES["surface_col"]
        + FEATURES["binary_col"]
        + FEATURES["count_col"]
        + FEATURES["distance_col"]
        + FEATURES["ordinal_col"]
        + FEATURES["geographical_col"]
    ]
    y = df[TARGET]

    # ✅ 3. Missing values + encoding
    X = fill_missing_values(X)
    X = encode_categorical(X)

    # ✅ 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ 5. Create pipeline (IMPORTANT ✅)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    # ✅ 6. Train
    pipeline.fit(X_train, y_train)

    # ✅ 7. Predict
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    print("r2_train", r2_score(y_train, y_train_pred))
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    # ✅ 8. Save EVERYTHING together
    joblib.dump(pipeline, "model/model_linear.pkl")

    return pipeline


if __name__ == "__main__":
    df = pd.read_csv("data/listings_clean_duplicate_final.csv")
    train_linear_model(df)

