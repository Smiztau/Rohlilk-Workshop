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

common_features = ['unique_id', 'warehouse', 'product_unique_id', 'L2_category_name_en','L3_category_name_en','L4_category_name_en','name_number','Embedding Component 1','Embedding Component 2','L1_category_name_en_Bakery','L1_category_name_en_Fruit and vegetable','L1_category_name_en_Meat and fish']
# Read the entire dataset
df_train = pd.read_csv('csv_junk/merged_data_train.csv')
df_test = pd.read_csv('csv_junk/merged_data_test.csv')

# Prepare the feature set (X) and target (y)
X = df_train.drop(['sales', 'weight'], axis=1)
y = df_train['sales']
sample_weights = df_train['weight']
unique_ids = np.sort(df_test['unique_id'].unique())

# Set parameters for the models
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': utils.depth,
}

os.makedirs("models", exist_ok=True)
os.makedirs("submissions", exist_ok=True)
all_predictions = []
for id in unique_ids:
    unique_id_train_mask = (X['unique_id'] == id)
    unique_id_test_mask = (df_test['unique_id'] == id)
    # Subset your features, target, and weights accordingly
    X_train = X[unique_id_train_mask].drop(common_features, axis=1)

    y_train = y[unique_id_train_mask]
    w_train = sample_weights[unique_id_train_mask]

    X_test = df_test[unique_id_test_mask].drop(common_features, axis=1)
    
    # Check if we have enough data to train and validate
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping id {id}.")
        continue
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    evals = [(dtrain, f'train_{id}')]

    booster = xgb.train(params, dtrain, num_boost_round=utils.max_iter,evals=evals,early_stopping_rounds=10, verbose_eval=False)

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

    # Create a DataFrame for the current warehouse's predictions
    id_predictions = pd.DataFrame({
        "id": unique_id_test_ids,  # Unique IDs for the current warehouse
        "sales_hat": y_pred_test  # Predicted sales
    })

    # Append to the list of all predictions
    all_predictions.append(id_predictions)

    # Save the model for the current warehouse
    # booster.save_model(f"model_by_id/xgboost_model_unique_id_{id}.json")
    print(f"Completed predictions and model saving for unique_id {id}.")

# Combine all predictions into a single DataFrame
final_submission = pd.concat(all_predictions, ignore_index=True)


# Save the final submission to a single CSV file
final_submission.to_csv("submissions/submissionV2.csv", index=False)
print("Final submission saved as 'submissions/final_submission.csv'.")