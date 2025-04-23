import os
import pandas as pd
import numpy as np
import tsfresh
from tsfresh.feature_extraction import MinimalFCParameters  # type: ignore
from tsfresh.utilities.dataframe_functions import impute
from utils import *

def generate_tsfresh_features():
    print("Starting TSFresh feature extraction...")

    # TSFresh configuration
    custom_features = {
        "mean_change": None,
        "mean_second_derivative_central": None,
        "autocorrelation": [{"lag": lag} for lag in [1, 5, 10]],
        "partial_autocorrelation": [{"lag": lag} for lag in [1, 5, 10]],
        "c3": [{"lag": lag} for lag in [1, 5, 10]],
        "linear_trend": [{"attr": "slope"}],
        "agg_linear_trend": [{"f_agg": "mean", "attr": "slope", "chunk_len": 5}],
        "fft_coefficient": [{"coeff": coeff, "attr": "real"} for coeff in range(5)],
        "spkt_welch_density": [{"coeff": coeff} for coeff in range(5)],
        "mean": None,
        "standard_deviation": None,
        "skewness": None,
        "kurtosis": None,
        "median": None,
        "maximum": None,
        "minimum": None,
        "variance": None,
    }

    fresh_work_features = ['unique_id', 'day', 'month', 'year', 'max_discount', 'final_price']

    # Read original datasets
    df_train = pd.read_csv(merged_data_train)
    df_test = pd.read_csv(merged_data_test)

    df_train["test"] = 0
    df_test["test"] = 1
    df_test["weight"] = 0
    df_test["sales"] = 0

    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    os.makedirs("csv_junk", exist_ok=True)

    extracted_features_list = []

    for year in range(2016, 2025):
        for month in range(1, 13):
            for day in range(1, 32):
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

                fresh_features = fresh_features.sort_values(["unique_id", "year", "month", "day"])
                fresh_features = fresh_features.groupby("unique_id").tail(60)

                fresh_features = fresh_features.astype(float)
                fresh_features['time'] = pd.to_datetime(fresh_features[['year', 'month', 'day']])
                fresh_features = fresh_features.drop(columns=['year', 'month', 'day'])

                extracted = tsfresh.extract_features(
                    fresh_features,
                    column_id="unique_id",
                    column_sort="time",
                    n_jobs=os.cpu_count(),
                    default_fc_parameters=custom_features
                )

                extracted["unique_id"] = extracted.index.astype(int)
                curr_with_features = extracted.merge(curr_features, on="unique_id", how="inner")
                extracted_features_list.append(curr_with_features)

    # Combine all day-level features
    enriched_df = pd.concat(extracted_features_list, ignore_index=True)

    # Restore test flag
    enriched_df["test"] = enriched_df["test"].fillna(0).astype(int)

    # Separate and save
    enriched_train = enriched_df[enriched_df["test"] == 0].drop(columns=["test"])
    enriched_test = enriched_df[enriched_df["test"] == 1].drop(columns=["test"])

    enriched_train.to_csv(merged_data_train, index=False)
    enriched_test.to_csv(merged_data_test, index=False)


generate_tsfresh_features()
