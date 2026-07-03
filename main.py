
import pandas as pd
from src.config import FEATURES, TARGET
from src.train_xgboost import train_xgb_model
from src.train_linear import train_linear_model
from src.Train_dt import train_tree_model

if __name__ == "__main__":

    df = pd.read_csv("data/listings_clean_duplicate_final.csv")

    print("Training Linear Regression...")
    train_linear_model(df)

    print("Training Decision Tree...")
    train_tree_model(df)

    print("Training XGBoost...")
    train_xgb_model(df)
