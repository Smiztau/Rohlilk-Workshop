import xgboost as xgb
import pandas as pd
from datetime import datetime
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# === Parse argument for availability option ===
parser = argparse.ArgumentParser()
parser.add_argument("--use-availability", type=str, default="false")
args = parser.parse_args()
use_availability = args.use_availability.lower() == "true"

# Read the entire dataset
df_train = pd.read_csv(merged_data_train)
df_test = pd.read_csv(merged_data_test)
df_availability = pd.read_csv(availabilities)

# Warehouses one hot encoding
df_train = pd.get_dummies(df_train, columns=["warehouse"])
df_test = pd.get_dummies(df_test, columns=["warehouse"])

# Prepare the feature set (X) and target (y)
y = df_train['sales']
sample_weights = df_train['weight']
unique_ids = df_test['unique_id'].unique()
train_unique_id = df_train['unique_id']
test_unique_id = df_test['unique_id']
if use_availability:
    df_train = df_train.drop(['sales', 'weight'], axis=1)
    df_test = pd.merge(df_test, df_availability, on=["unique_id", "year", "month", "day"], how='left')

else:
    df_train = df_train.drop(['sales', 'weight', 'availability'], axis=1)

# Set parameters for the models
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': depth,
}

all_predictions = []
for unique_id in unique_ids:
    unique_id_train_mask = (train_unique_id == unique_id)
    unique_id_test_mask = (test_unique_id == unique_id)
    
    # Subset your features, target, and weights accordingly
    X_train = df_train[unique_id_train_mask].drop('unique_id', axis=1).copy()
    y_train = y[unique_id_train_mask]
    w_train = sample_weights[unique_id_train_mask]
    X_test = df_test[unique_id_test_mask].drop('unique_id', axis=1).copy()
    X_test = X_test[X_train.columns]

    # Check if we have enough data to train and validate
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping unique_id {unique_id}.")
        continue
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test)
    evals = [(dtrain, f'train_{unique_id}')]

    booster = xgb.train(params, dtrain, num_boost_round=max_iter, evals=evals, verbose_eval=True)

    # Predict on test set
    y_pred_test = booster.predict(dtest)
    
    # Create an ID column for the test predictions
    unique_id_test_ids = (
        df_test.loc[unique_id_test_mask, "unique_id"].astype(str)
        + "_"
        + df_test.loc[unique_id_test_mask, "year"].astype(str)
        + "-"
        + df_test.loc[unique_id_test_mask, "month"].astype(str).str.zfill(2)
        + "-"
        + df_test.loc[unique_id_test_mask, "day"].astype(str).str.zfill(2)
    )

    # Create a DataFrame for the current unique_id's predictions
    unique_id_predictions = pd.DataFrame({
        "id": unique_id_test_ids,  # Unique IDs for the current unique_id
        "sales_hat": y_pred_test  # Predicted sales
    })

    # Append to the list of all predictions
    all_predictions.append(unique_id_predictions)

    # # Save the model for the current unique_id
    # booster.save_model(f"models/xgboost_model_unique_id_{unique_id}_{current_time}.json")

# Combine all predictions into a single DataFrame
final_submission = pd.concat(all_predictions, ignore_index=True)


# Save the final submission to a single CSV file
final_submission.to_csv(f"submissions/submission-{current_time}.csv", index=False)
