import xgboost as xgb
import pandas as pd
from datetime import datetime
import argparse
from utils import *

# === Parse argument for availability option ===
parser = argparse.ArgumentParser()
parser.add_argument("--use-availability", type=str, default="false")
args = parser.parse_args()
use_availability = args.use_availability.lower() == "true"

# Read the entire dataset
df_train = pd.read_csv(merged_data_train)
df_test = pd.read_csv(merged_data_test)
df_availability = pd.read_csv(availabilities)

# Prepare the feature set (X) and target (y)
y = df_train['sales']
sample_weights = df_train['weight']
warehouses = df_test['warehouse'].unique()
train_warehouse = df_train['warehouse']
test_warehouse = df_test['warehouse']
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
for warehouse in warehouses:
    warehouse_train_mask = (train_warehouse == warehouse)
    warehouse_test_mask = (test_warehouse == warehouse)
    
    # Subset your features, target, and weights accordingly
    X_train = df_train[warehouse_train_mask].drop('warehouse', axis=1).copy()
    y_train = y[warehouse_train_mask]
    w_train = sample_weights[warehouse_train_mask]
    X_test = df_test[warehouse_test_mask].drop('warehouse', axis=1).copy()
    X_test = X_test[X_train.columns]

    # Check if we have enough data to train and validate
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping warehouse {warehouse}.")
        continue
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test)
    evals = [(dtrain, f'train_{warehouse}')]

    booster = xgb.train(params, dtrain, num_boost_round=max_iter, evals=evals, verbose_eval=True)

    # Predict on test set
    y_pred_test = booster.predict(dtest)
    
    # Create an ID column for the test predictions
    warehouse_test_ids = (
        df_test.loc[warehouse_test_mask, "unique_id"].astype(str)
        + "_"
        + df_test.loc[warehouse_test_mask, "year"].astype(str)
        + "-"
        + df_test.loc[warehouse_test_mask, "month"].astype(str).str.zfill(2)
        + "-"
        + df_test.loc[warehouse_test_mask, "day"].astype(str).str.zfill(2)
    )

    # Create a DataFrame for the current warehouse's predictions
    warehouse_predictions = pd.DataFrame({
        "id": warehouse_test_ids,  # Unique IDs for the current warehouse
        "sales_hat": y_pred_test  # Predicted sales
    })

    # Append to the list of all predictions
    all_predictions.append(warehouse_predictions)

    # Save the model for the current warehouse
    booster.save_model(f"models/xgboost_model_warehouse_{warehouse}.json")
    print(f"Completed predictions and model saving for warehouse {warehouse}.")

# Combine all predictions into a single DataFrame
final_submission = pd.concat(all_predictions, ignore_index=True)


# Save the final submission to a single CSV file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
final_submission.to_csv(f"submissions/submission-{current_time}.csv", index=False)
