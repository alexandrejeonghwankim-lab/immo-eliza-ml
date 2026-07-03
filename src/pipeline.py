from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def build_pipeline(model, num_cols, scale=True):
    # Bring the NaN directly to the XGBoost
    if scale:
        num_pipeline = Pipeline([
            # "imputer" Decided not to use imputer.
            ("scaler", StandardScaler())
        ])
    else:
        # if there is no scaling, pass it without pipeline
        num_pipeline = "passthrough"

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols)
    ], remainder="passthrough") # Other than numeric columns then just pass directly to the XGboost

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return pipeline