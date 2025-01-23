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
# Set parameters for the models
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': 12,
    'silent': 1
}

# Each item in 'splits' is ((train_end), (val_start), (val_end))
# For example, if you want:
#   - Iteration 1: train up to Mar 2,   validate Mar 3 - Mar 16
#   - Iteration 2: train up to Apr 2,  validate Apr 3 - Apr 16
#   - Iteration 3: train up to May 2,  validate May 3 - May 16
# just adjust the tuples below.
splits = [
    ((2024, 3, 2),  (2024, 3, 3),  (2024, 3, 16)),  # Iteration 1
    ((2024, 4, 2),  (2024, 4, 3),  (2024, 4, 16)),  # Iteration 2
    ((2024, 5, 2),  (2024, 5, 3),  (2024, 5, 16)),  # Iteration 3
]

train_val_sets = []
for (train_end, val_start, val_end) in splits:
    # Get boolean masks for train and validation
    train_mask, val_mask = get_train_val_masks(X, train_end, val_start, val_end)
    
    # Subset your features, target, and weights accordingly
    X_train = X[train_mask]
    y_train = y[train_mask]
    w_train = sample_weights[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    w_val = sample_weights[val_mask]
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dvalidation = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    evals = [(dtrain, 'train_' + str(val_end[1])), (dvalidation, 'validation' + str(val_end[1]))]

    booster = xgb.train(params, dtrain, num_boost_round=150, evals=evals, verbose_eval=True)
    # Predict on test set
    y_pred_validation = booster.predict(dvalidation)
    wmae = mean_absolute_error(y_val, y_pred_validation, sample_weight=w_val)
    total_wmae += wmae
    cnt +=1
    print("VALIDATION " + str(val_end[1]) + " Set WMAE:", wmae)
    print("X shape:", X.shape)
    print("X_train shape:", X_train.shape)
    print("X_validation shape:", X_val.shape)
    print("y shape:", y.shape)
    print("y_train shape:", y_train.shape)
    print("y_validation shape:", y_val.shape)

print("AVG WMAE:", total_wmae/cnt)

# booster.save_model("xgboost_model_validation.json")

# # Example usage: you could loop over them to train/evaluate
# for i, (X_tr, y_tr, w_tr, X_val, y_val, w_val) in enumerate(train_val_sets, 1):
#     print(f"---- Iteration {i} ----")
#     print(f"Train set size: {len(X_tr)}")
#     print(f"Validation set size: {len(X_val)}")
#     # train a model, for example:
#     # model = SomeRegressor().fit(X_tr, y_tr, sample_weight=w_tr)
#     # val_predictions = model.predict(X_val)
#     # evaluate performance, etc.

# 3741570
# 3847199
# 3950082
