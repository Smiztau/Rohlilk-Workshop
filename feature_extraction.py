import xgboost as xgb
import pandas as pd
import os
import utils
import tsfresh
from tsfresh.feature_extraction import MinimalFCParameters
import matplotlib.pyplot as plt
from tsfresh.feature_selection import significance_tests
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np


create_data = 1

if __name__ == "__main__":
    final_data = []
    if create_data:
        custom_features = {
            # Rolling Window Features
            "mean_change": None,
            "mean_second_derivative_central": None,

            # Lag-Based Features
            "autocorrelation": [{"lag": lag} for lag in [1, 5, 10]],
            "partial_autocorrelation": [{"lag": lag} for lag in [1, 5, 10]],
            "c3": [{"lag": lag} for lag in [1, 5, 10]],

            # Trend & Seasonality Features
            "linear_trend": [{"attr": "slope"}],
            "agg_linear_trend": [{"f_agg": "mean", "attr": "slope", "chunk_len": 5}],  # ✅ Fix here

            "fft_coefficient": [{"coeff": coeff, "attr": "real"} for coeff in range(5)],
            "spkt_welch_density": [{"coeff": coeff} for coeff in range(5)],

            # Basic Statistics
            "mean": None,
            "standard_deviation": None,
            "skewness": None,
            "kurtosis": None,
            "median": None,
            "maximum": None,
            "minimum": None,
            "variance": None,
        }


        fresh_work_features = ['year', 'month', 'day','unique_id','total_orders', 'sell_price_main','max_discount','final_price']
        
        # Read the entire dataset
        df_train = pd.read_csv('csv_junk/merged_data_train.csv')
        df_test = pd.read_csv('csv_junk/merged_data_test.csv')
        
        df_train["test"] = 0
        df_test["test"] = 1
        
        df_test["weight"] = 0
        df_test["sales"] = 0
        
        df_combined = pd.concat([df_train, df_test], ignore_index=True)

        os.makedirs("models", exist_ok=True)
        os.makedirs("submissions", exist_ok=True)
        
        final_combined_features = [] 
        
        for year in range(2016, 2025, 1):
            for month in range(1, 13, 1):
                for day in range(1, 32, 1):
                    mask = (
                    (df_combined["year"] == year) &
                    (df_combined["month"] == month) &
                    (df_combined["day"] == day)
                    )
                    
                    if df_combined[mask].empty:
                        continue  
                    
                    unique_date_train_mask = (
                        (df_combined["year"] <= year) &
                        ((df_combined["year"] < year) | (df_combined["month"] < month) | (df_combined["day"] <= day))
                    )

                    if df_combined[unique_date_train_mask].empty:
                        continue
                    
                    curr_features = df_combined[mask]
                    fresh_features = df_combined[unique_date_train_mask]

                    valid_unique_ids = curr_features["unique_id"].unique()
                    fresh_features = fresh_features[fresh_features["unique_id"].isin(valid_unique_ids)]

                    fresh_features = fresh_features[fresh_work_features]                
                    fresh_features = fresh_features[fresh_features["unique_id"].isin(valid_unique_ids)]

                    fresh_features = fresh_features.sort_values(["unique_id", "year", "month", "day"])  
                    fresh_features = fresh_features.groupby("unique_id").tail(60)  

                    fresh_features = fresh_features.astype(float)  
                    fresh_features['time'] = pd.to_datetime(fresh_features[['year', 'month', 'day']])
                    fresh_features = fresh_features.drop(columns=['year', 'month', 'day'])

                    fresh_features = tsfresh.extract_features(
                        fresh_features, 
                        column_id="unique_id",  
                        column_sort="time",  
                        n_jobs= os.cpu_count(),
                        default_fc_parameters=custom_features  # Extracts fewer, faster features
                    )

                    fresh_features["unique_id"] = fresh_features.index.astype(int)

                    combined_features = fresh_features.merge(curr_features, on="unique_id", how="inner")
                    print(combined_features.shape)
                    final_combined_features.append(combined_features)
                        

        
        final_data = pd.concat(final_combined_features, ignore_index=True)
        final_data.to_csv("csv_junk/tsfresh_data.csv", index=False)
    else:
        final_data = pd.read_csv("csv_junk/tsfresh_data.csv")
    
    final_data = pd.get_dummies(final_data, columns=['warehouse'])

    # Get columns that are not numeric
    non_numeric_columns = final_data.select_dtypes(exclude=['number']).columns
    # Convert non-numeric columns to int
    final_data[non_numeric_columns] = final_data[non_numeric_columns].astype(int)

    X_train = final_data[final_data["test"] == 0]
    sample_weights = X_train['weight']
    y_train = X_train["sales"]
    X_train = X_train.drop(columns=["sales", "weight"])

    X_test = final_data[final_data["test"] == 1].drop(columns=["sales","weight"])
    print(X_train.isnull().sum().sum())  # Total number of NaN values in the dataset
    missing_values = X_train.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    print(missing_columns)  # Show columns with NaN values and their count
    rows_with_nan = X_train[X_train.isnull().any(axis=1)]
    print(rows_with_nan)  # Display rows with NaN values

    # Filter the features based on p-value threshold (e.g., 0.05)
    print("The shape of data before impute is", X_train.shape)
    X_train = impute(X_train)
    print("The shape of data after impute is", X_train.shape)

    selected_features = select_features(X_train, y_train, fdr_level=0.001)
    selected_columns = selected_features.columns
    print("The amount of selected features is", selected_features.shape[1])
    X_train = X_train[selected_columns]
    X_test = X_test[selected_columns]

    # Set parameters for the models
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'learning_rate': 0.1,
        'max_depth': utils.depth,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(X_test)
    evals = [(dtrain, "train")]

    booster = xgb.train(params, dtrain, num_boost_round=utils.max_iter,evals=evals,early_stopping_rounds=10, verbose_eval=False)

    y_pred_test = booster.predict(dtest)
    
    # Create an ID column for the test predictions
    id_series = (
    X_test["unique_id"].astype(str)
    + "_"
    + X_test["year"].astype(str)
    + "-"
    + X_test["month"].astype(str).str.zfill(2)
    + "-"
    + X_test["day"].astype(str).str.zfill(2)
    )

    # Create a DataFrame for the current warehouse's predictions
    id_predictions = pd.DataFrame({
        "id": id_series,  # Unique IDs for the current warehouse
        "sales_hat": y_pred_test  # Predicted sales
    })

    xgb.plot_importance(booster, importance_type='gain')
    plt.show()

   # Save the model for the current warehouse
    booster.save_model(f"models/xgboost_model_tsfresh.json")
    print(f"Completed predictions and model saving.")

    # Save the final submission to a single CSV file
    id_predictions.to_csv("submissions/submission_fresh.csv", index=False)
    print("Final submission saved as 'submissions/final_submission_fresh.csv'.")