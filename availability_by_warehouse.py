import xgboost as xgb
import pandas as pd
from datetime import datetime
from utils import *

# Read the entire dataset
df_train = pd.read_csv(merged_data_train)
df_test = pd.read_csv(merged_data_test)

# Prepare the feature set (X) and target (y)
X = df_train.drop(['availability', 'weight', 'sales'], axis=1)
y = df_train['availability']
sample_weights = df_train['weight']
warehouses = df_test['warehouse'].unique()

# Set parameters for the models
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': 10,
}

all_predictions = []
for warehouse in warehouses:
    warehouse_train_mask = (X['warehouse'] == warehouse)
    warehouse_test_mask = (df_test['warehouse'] == warehouse)
    
    # Subset your features, target, and weights accordingly
    X_train = X[warehouse_train_mask].drop('warehouse', axis=1).copy()
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

    booster = xgb.train(params, dtrain, num_boost_round=50, evals=evals, verbose_eval=True)

    # Predict on test set
    y_pred_test = booster.predict(dtest)
    # Post-process predictions: force high predictions to 1, and round the rest to 2 decimals
    y_pred_test = np.where(y_pred_test > 0.9, 1.0, np.round(y_pred_test, 2))

    # Create a DataFrame for the current warehouse's predictions
    warehouse_predictions = pd.DataFrame({
        "unique_id": df_test.loc[warehouse_test_mask, "unique_id"],
        "year": df_test.loc[warehouse_test_mask, "year"],
        "month": df_test.loc[warehouse_test_mask, "month"],
        "day": df_test.loc[warehouse_test_mask, "day"],
        "availability": y_pred_test  # Predicted availability
    })

    # Append to the list of all predictions
    all_predictions.append(warehouse_predictions)

# Combine all predictions into a single DataFrame
df_availabilities = pd.concat(all_predictions, ignore_index=True)


# Save the final submission to a single CSV file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
df_availabilities.to_csv(availabilities, index=False)
