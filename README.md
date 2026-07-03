# **Immo Eliza Real Estate Price Prediction**

An end-to-end Machine Learning pipeline developed to predict real estate property prices in Belgium. This project progresses from data cleaning and advanced feature engineering to evaluating multiple regression models, ultimately selecting the most robust model for production.


## Repo structure
This structure ensures a clean separation between data, model, pipeline, and training scripts, making the project reusable and maintainable.

immo-eliza-ml/
│
├── data/
│   └── listings_clean_duplicate_final.csv
│
├── model/
│   └── model_xgboost.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── pipeline.py
│   ├── train_xgboost.py
│   ├── train_linear.py
│   └── train_dt.py
│
├── main.py
├── requirements.txt
└── README.md


## 1. How the Process works

The project is built around a reusable, robust, and automated machine learning pipeline:

* **Data Cleaning (clean_data):** Filters out extreme values (properties priced under €50k or over €5M). It handles local anomalies by applying a dynamic Interquartile Range (IQR) outlier removal grouped by postal_code. Finally, it removes extreme luxury anomalies by clipping data at the 99th percentile.
* **Feature Engineering (features_engineering):** Several transformations were applied to improve the predictive power of the dataset:

  * Ordinal Encoding: Translates qualitative scores into numeric scales (e.g., mapping energy efficiency (epc) and kitchen_equipment quality into 1–4 scales).
  * Geographical Proximity: Uses the Haversine formula to compute the exact physical distance between a property's coordinates and major Belgian metropolitan areas and cities.
  * Distance-Based Features: Features such as distance to Brussles or other urban areas were added. Properties closer to cities tend to have  higher prices.
  * Binary & Count features: Features such as number of rooms, gardens, terraces, etc were included. These directly influence property value and attractiveness.
  * Handiling Missing Values: Instead of imputing values for XGBOOST, missing data (NaN) was left to handle natively. This preserves data structure and improves learning performance.
* **The Reusable Pipeline (pipeline.py):** Standardizes structural transformations. For models requiring it, numerical features are scaled using a StandardScaler within a ColumnTransformer. For advanced models like XGBoost, missing values (NaNs) are seamlessly left un-imputed so the model can natively determine optimal splits.

## 2. Chronological Testing & Progression Results

I approached model building iteratively, testing three distinct setups sequentially to measure incremental improvements:

### Test 1: Linear Regression (The Baseline)

First, I established a traditional statistical baseline, Linear Regression. This step required explicit missing-value imputation and categorical one-hot encoding before passing features through a standard scaler.

    r2_train: 0.6607

    R2 (Test): 0.6346

    MAE: €92,935.45

### Test 2: Decision Tree Regressor (Capturing Non-Linearity)

Next, I tested a tree-based approach to capture non-linear market patterns. This model also utilized explicit imputation and encoding features but required no feature scaling.

    r2_train: 0.8280

    R2 (Test): 0.6157

    MAE: €92,890.44

### Test 3: XGBoost Regressor (The Optimized Champion)

Finally, I implemented an advanced gradient boosting algorithm (train_xgboost.py) leveraging native missing value handling and extensive hyperparameter tuning. A 5-fold cross-validation strategy is introduced to prove model stability across different market data splits.

    r2_train: 0.9202 | MAE_train: €47,417.96

    R2 (Test): 0.8006

    MAE: €66,031.84

    CV R2 scores: [0.80165517 0.80618298 0.82002449 0.79629767 0.82127404]

    Mean CV R2: 0.8091

# 3. Model Performance Evaluation

Three distinct regression models are tested to find the best between complexity and predictive power.

## Model Performance Comparison

| Metric                    | Linear Regression | Decision Tree | XGBoost (Final)       |
| ------------------------- | ----------------- | ------------- | --------------------- |
| Training R² (r2_train)   | 0.6607            | 0.8280        | **0.9202**      |
| Testing R² (R²)         | 0.6346            | 0.6157        | **0.8006**      |
| Mean Absolute Error (MAE) | €92,935.45       | €92,890.44   | **€66,031.84** |

## Cross Validation Metrics (XGBoost only)

* 5-Fold CV $R^2$ Scores: [0.80165517 0.80618298 0.82002449 0.79629767 0.82127404]
* Mean CV $R^2$: 0.8090

# 4, Performance Comaprison & Why XGBoost was selected

Analyzing the journey, Test 1 to Test 3, XGBoost was overwhelmingly selected for production deployment:

1. **Drastic Error Reduction**: Moving from Linear Regression/Decision Trees to XGBoost sliced our average prediction error (MAE) down from ~€92.9k to €66k, saving nearly €27,000 in pricing inaccuracy per property.
2. **Overfitting Control**: The Decision Tree suffered severe overfitting—performing well on training data (0.8280) but collapsing on real test data (0.6157). XGBoost successfully maintained generalized power, holding an outstanding 0.8006 Test $R^2$ score.
3. **Proven Reliability**: The 5-fold Cross-Validation verified that XGBoost's high accuracy wasn’t a random fluke. Its performance remained tightly uniform around an 80.90% average across multiple random sub-segments of the dataset.

# 5. Business Translation

For Business stakeholders, this is what the performance metrics actually mean for operations:

**$
R^2$ (R-Squared) & r2_train**

* Data Science Term: The proportion of variance in the dependent variable that is predictable from the independent variables.
* Business Interpretation: Think of this as our "Market Trends Explanation Score". Our final test score of 0.8006 means our model successfully decodes and explains 80.1% of the reasons why property prices fluctuate in Belgium. The r2_train (92.02%) simply shows how well the model memorized historical files during training.

**MAE(Mean Absolute Error)**

* Data Science Term: The average of the absolute differences between forecasted targets and true targets.
* Business Interpretation: This is our "Average Mispricing Window". On average, when our model stamps a price valuation on a property, its estimate is off by €66,031.84. Shrinking this error margin down from the initial €92.9k baseline dramatically insulates the business from costly over-valuations and under-valuations.

***CV $R^2$ Scores & Mean CV $R^2$***

* Data Science Term: Evaluation metrics derived from K-Fold cross-validation to assess model generalization.
* Business Interpretation: This is our "Model Consistency Insurance Policy". Instead of relying on a single test, we simulated launching this model 5 distinct times across completely different sectors of the Belgian housing market. The fact that all 5 scores hovered securely around 80.6% guarantees that our system is highly stable and will predict prices with uniform accuracy on real-world market listings.
