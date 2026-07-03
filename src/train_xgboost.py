
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from src.config import FEATURES, TARGET
from src.preprocessing import clean_data, features_engineering
from src.pipeline import build_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np



def train_xgb_model(df):

    df = clean_data(df)
    df = features_engineering(df)
    #print(df.columns.tolist()) 


    
    categorical_cols = []
    if "categorical_col" in FEATURES:
        categorical_cols = [col for col in FEATURES["categorical_col"] if col in df.columns]

    X = df[FEATURES["surface_col"] + FEATURES["binary_col"] + FEATURES["count_col"] + FEATURES["distance_col"]+ FEATURES["ordinal_col"]+ FEATURES["geographical_col"] + FEATURES.get("categorical_col",[])]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = X.columns.tolist()
    categorical_cols = [col for col in FEATURES.get("categorical_col", []) if col in X.columns]


    pipeline = build_pipeline(
      
        model=XGBRegressor(n_estimators=800,
                            max_depth=6,
                            learning_rate=0.02,
                            subsample=0.7,
                            colsample_bytree=0.7,
                            min_child_weight=3,
                            reg_lambda=3,
                            random_state=42
                            ),
        num_cols=num_cols,
        cat_cols = categorical_cols,
        scale=False
    )

    pipeline.fit(X_train, y_train)

    
       
    y_train_pred= pipeline.predict(X_train)
    y_pred= pipeline.predict(X_test)
    

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv =kf,
        scoring = "r2"
    )

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))


    print("r2_train", r2_score(y_train, y_train_pred))
    print("MAE_train:", mean_absolute_error(y_train, y_train_pred))


    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    print("CV R2 scores:", cv_scores)
    print("Mean CV R2:", cv_scores.mean())

    
    print("RMSE_train:", rmse_train)
    print("RMSE:", rmse_test)


    joblib.dump(pipeline, "model/model_xgboost.pkl")

    

    return pipeline


if __name__ == "__main__":
    df = pd.read_csv("data/listings_clean_duplicate_final.csv")
    train_xgb_model(df)
