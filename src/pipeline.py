
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_pipeline(model, num_cols, cat_cols=None, scale=True):

    # numerical pipeline
    if scale:
        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])
    else:
        num_pipeline = "passthrough"

    # categorical pipeline 
    if cat_cols:
        cat_pipeline = OneHotEncoder(handle_unknown="ignore")
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])
    else:
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols)
        ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return pipeline
