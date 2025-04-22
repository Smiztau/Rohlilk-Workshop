import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from utils import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process data based on selected options.")
parser.add_argument("--rolling_avg", action="store_true", help="Use rolling average in training")
parser.add_argument("--train_by_warehouse", action="store_true", help="Train by warehouse")
parser.add_argument("--train_by_unique_ids", action="store_true", help="Train by unique IDs")
args = parser.parse_args()

# Load CSV files
df_sales_train = pd.read_csv(sales_train)
df_sales_test = pd.read_csv(sales_test)
df_inventory = pd.read_csv(inventory)
df_calendar = pd.read_csv(calendar)
df_weights = pd.read_csv(weights)
df_food_embeddings = pd.read_csv(food_embeddings)
df_availabilities = pd.read_csv(availabilities)

# Function to process data
def process_data(df_sales, is_train):
    df = pd.merge(df_sales, df_inventory.drop(columns=['warehouse'], errors='ignore'), on='unique_id', how='left')

    if is_train:
        labels = df_sales["sales"]
        labels.to_csv(train_labels, index=False)
        df = pd.merge(df, df_weights, on='unique_id', how='left')
        df = df.dropna(subset=['sales'])
        df = df.dropna(subset=['availability'])
        
    # Process categorical features
    df[['name_only', 'name_number']] = df['name'].str.split('_', expand=True)
    df.drop('name', axis=1, inplace=True)
    for col in ['L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']:
        df[col] = df[col].str.split('_').str[-1].astype(float)

    # Compute discounts
    df['max_discount'] = df[[f'type_{i}_discount' for i in range(7)]].max(axis=1)
    df['final_price'] = df['sell_price_main'] * (1 - df['max_discount'])
    df.drop(columns=[f'type_{i}_discount' for i in range(7)], inplace=True)

    # Process date-related features
    df['date'] = pd.to_datetime(df['date'], format="mixed", dayfirst=True)
    df["week_of_the_month"] = ((df["date"].dt.day - 1) // 7) + 1
    df['weekday'] = df['date'].dt.weekday
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Encode seasonal and time features
    df['season'] = df['month'].apply(get_season)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['sin_dayofyear'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Apply rolling average if selected
    if args.rolling_avg:
        df['rolling_avg'] = df['sales'].rolling(window=7, min_periods=1).mean()

    df.drop('date', axis=1, inplace=True)

    # Merge calendar data
    df = pd.merge(df, df_calendar, on=['day', 'month', 'year', 'warehouse'], how='left')

    # Merge word embeddings
    df = pd.merge(df, df_food_embeddings, on=["name_only"], how='left')
    df.drop('name_only', axis=1, inplace=True)

    # One-hot encode L1 categories
    df = pd.get_dummies(df, columns=['L1_category_name_en'])

    # Train by warehouse or unique IDs if selected
    # if args.train_by_warehouse:
    #     df = df.groupby('warehouse').mean().reset_index()
    # elif args.train_by_unique_ids:
        # df = df.groupby('unique_id').mean().reset_index()

    # Drop unwanted columns
    df.drop('holiday_name', axis=1, errors='ignore', inplace=True)

    # Save processed data
    fileOutName = merged_data_train if is_train else merged_data_test
    df.to_csv(fileOutName, index=False)

    return df

# Process train and test datasets
df_train_processed = process_data(df_sales_train, is_train=True)
df_test_processed = process_data(df_sales_test, is_train=False)

print("<<<<<<<<<<<<<<<<<<<< FINISH PROCESSING BOTH TRAIN AND TEST DATA >>>>>>>>>>>>>>>>>>")
