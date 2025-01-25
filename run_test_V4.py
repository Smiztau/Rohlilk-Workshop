import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils import get_train_val_masks
import matplotlib.pyplot as plt
import os
import utils

print(utils.depth)
print(utils.max_iter)

# Read the entire dataset
df_train = pd.read_csv('csv_junk/merged_data_train.csv')
df_test = pd.read_csv('csv_junk/merged_data_test.csv')

# Prepare the feature set (X) and target (y)
X = df_train.drop(['sales', 'weight'], axis=1)
y = df_train['sales']
sample_weights = df_train['weight']
categories = df_test['L1_category_name_en'].unique()

# Set parameters for the models
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': utils.depth,
    'silent': 1
}

os.makedirs("models_L1_category", exist_ok=True)
os.makedirs("submissions", exist_ok=True)
all_predictions = []
for category in categories:
    
    category_train_mask = (X['L1_category_name_en'] == category)
    category_test_mask = (df_test['L1_category_name_en'] == category)
    # Subset your features, target, and weights accordingly
    X_train = X[category_train_mask].drop('L1_category_name_en', axis=1).copy()

    y_train = y[category_train_mask]
    w_train = sample_weights[category_train_mask]
    X_test = df_test[category_test_mask].drop('L1_category_name_en', axis=1).copy()
    print(X_train.head(10))
    print(y_train.head(10))
    print(w_train.head(10))
    print(X_test.head(10))
    # Check if we have enough data to train and validate
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping category {category}.")
        continue
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test)
    evals = [(dtrain, f'train_{category}')]

    booster = xgb.train(params, dtrain, num_boost_round=utils.max_iter, evals=evals, verbose_eval=True)

    # Predict on test set
    y_pred_test = booster.predict(dtest)
    
    # Create an ID column for the test predictions
    category_test_ids = (
        df_test.loc[category_test_mask, "unique_id"].astype(str)
        + "_"
        + df_test.loc[category_test_mask, "year"].astype(str)
        + "-"
        + df_test.loc[category_test_mask, "month"].astype(str).str.zfill(2)
        + "-"
        + df_test.loc[category_test_mask, "day"].astype(str).str.zfill(2)
    )

    # Create a DataFrame for the current warehouse's predictions
    category_predictions = pd.DataFrame({
        "id": category_test_ids,  # Unique IDs for the current warehouse
        "sales_hat": y_pred_test  # Predicted sales
    })

    # Append to the list of all predictions
    all_predictions.append(category_predictions)

    # Save the model for the current warehouse
    booster.save_model(f"models_L1_category/xgboost_model_category_{category}.json")
    print(f"Completed predictions and model saving for category {category}.")

# Combine all predictions into a single DataFrame
final_submission = pd.concat(all_predictions, ignore_index=True)


# Save the final submission to a single CSV file
final_submission.to_csv("submissions/submissionV3.csv", index=False)
print("Final submission saved as 'submissions/final_submission.csv'.")