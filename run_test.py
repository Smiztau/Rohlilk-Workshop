import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils import get_train_val_masks
import matplotlib.pyplot as plt

# Read the entire dataset
df_train = pd.read_csv('merged_data_train.csv')
df_test = pd.read_csv('merged_data_test.csv')

# Prepare the feature set (X) and target (y)
X = df_train.drop(['sales', 'weight'], axis=1)
y = df_train['sales']
sample_weights = df_train['weight']
X_test = df_test

# Set parameters for the models
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': 12,
    'silent': 1
}

dtrain = xgb.DMatrix(X, label=y, weight=sample_weights)
dtest = xgb.DMatrix(X_test)
evals = [(dtrain, 'train')]
booster = xgb.train(params, dtrain, num_boost_round=150, evals=evals, verbose_eval=True)

# Predict on test set
y_pred_test = booster.predict(dtest)
id_series = (
    df_test["unique_id"].astype(str)
    + "_"
    + df_test["year"].astype(str)
    + "-"
    + df_test["month"].astype(str).str.zfill(2)
    + "-"
    + df_test["day"].astype(str).str.zfill(2)
)

# 2. Create a DataFrame with two columns: "id" as first, "sales_nhat" as second
df_sales = pd.DataFrame({
    "id": id_series,        # first column
    "sales_hat": y_pred_test    # second column
})
df_sales.to_csv("submission4.csv", index=False)

booster.save_model("xgboost_model_test.json")