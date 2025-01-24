import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils import get_train_val_masks

total_wmae = 0
cnt = 0

# Read the entire dataset
df_train = pd.read_csv('csv_junk/merged_data_train.csv')

# Prepare the feature set (X) and target (y)
X = df_train.drop(['sales', 'weight'], axis=1)
y = df_train['sales']
sample_weights = df_train['weight']
warehouses = X['warehouse'].unique()

# Set parameters for the models
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': 12,
    'silent': 1
}

splits = [
    ((2024, 3, 2),  (2024, 3, 3),  (2024, 3, 16)),  # Iteration 1
    ((2024, 4, 2),  (2024, 4, 3),  (2024, 4, 16)),  # Iteration 2
    ((2024, 5, 2),  (2024, 5, 3),  (2024, 5, 16)),  # Iteration 3
]

results = []
for (train_end, val_start, val_end) in splits:
    # Get boolean masks for train and validation
    train_mask, val_mask = get_train_val_masks(X, train_end, val_start, val_end)
    for warehouse in warehouses:
        warehouse_train_mask = train_mask & (X['warehouse'] == warehouse)
        warehouse_val_mask = val_mask & (X['warehouse'] == warehouse)
        # Subset your features, target, and weights accordingly
        X_train = X[warehouse_train_mask]
        X_train.drop('warehouse', axis=1, inplace=True)
        y_train = y[warehouse_train_mask]
        w_train = sample_weights[warehouse_train_mask]
    
        X_val = X[warehouse_val_mask]
        X_val.drop('warehouse', axis=1, inplace=True)
        y_val = y[warehouse_val_mask]
        w_val = sample_weights[warehouse_val_mask]

        # Check if we have enough data to train and validate
        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            print(f"Skipping warehouse {warehouse} for split ending {val_end} due to insufficient data.")
            continue
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dvalidation = xgb.DMatrix(X_val, label=y_val, weight=w_val)
        evals = [(dtrain, f'train_{warehouse}_{val_end[1]}'), (dvalidation, f'validation_{warehouse}_{val_end[1]}')]

        booster = xgb.train(params, dtrain, num_boost_round=150, evals=evals, verbose_eval=True)
        # Predict on test set
        y_pred_validation = booster.predict(dvalidation)
        wmae = mean_absolute_error(y_val, y_pred_validation, sample_weight=w_val)
        results.append({
            'month': val_end[1],
            'warehouse': warehouse,
            'wmae': wmae
        })
        print("VALIDATION of warhouse " + warehouse + " of month " + str(val_end[1]) + " Set WMAE:", wmae)


avg_wmae = np.mean([res['wmae'] for res in results])
print(f"Average WMAE across all warehouses and splits: {avg_wmae}")



    # print("X shape:", X.shape)
    # print("X_train shape:", X_train.shape)
    # print("X_validation shape:", X_val.shape)
    # print("y shape:", y.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_validation shape:", y_val.shape)