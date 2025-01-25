import pandas as pd
import numpy as np

depth = 15
max_iter = 150
merged_data_train = "csv_junk/merged_data_train.csv"
merged_data_test = "csv_junk/merged_data_test.csv"
sales_train = "csv_input/sales_train.csv"
sales_test = "csv_input/sales_test.csv"
inventory = "csv_input/inventory.csv"
weights = "csv_input/test_weights.csv"
calendar = "csv_junk/calendar_enriched.csv"
food_embedings = "csv_junk/food_embeddings.csv"
labels_csv="csv_input/train_labels.csv"
csv_junk = "csv_junk"
rolling = 'csv_junk/rolling.csv'

def get_train_val_masks(X, train_end, val_start, val_end):
    """
    Generate boolean masks for training and validation based on date cutoffs.
    
    Arguments:
        X          : Features DataFrame containing 'year', 'month', 'day'
        train_end  : Tuple (year, month, day) inclusive cutoff for training
        val_start  : Tuple (year, month, day) inclusive start for validation
        val_end    : Tuple (year, month, day) inclusive end for validation
        
    Returns:
        train_mask : Boolean Series (True = row belongs in training)
        val_mask   : Boolean Series (True = row belongs in validation)
    """
    # Unpack the cutoff tuples
    y_te, m_te, d_te = train_end   # Train end date
    y_vs, m_vs, d_vs = val_start   # Validation start date
    y_ve, m_ve, d_ve = val_end     # Validation end date
    
    # Training mask: up to train_end (inclusive)
    train_mask = (
        (X['year'] < y_te)
        | (
            (X['year'] == y_te)
            & (
                (X['month'] < m_te)
                | (
                    (X['month'] == m_te) 
                    & (X['day'] <= d_te)
                )
            )
        )
    )
    
    # Validation mask: between val_start and val_end (inclusive)
    val_mask = (
        (X['year'] == y_vs)
        & (X['month'] == m_vs)
        & (X['day'] >= d_vs)
        & (X['day'] <= d_ve)   # same month & year assumption in this example
    )
    
    return train_mask, val_mask

def get_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4
    
def calculate_previous_month_avg(group):
    group['prev_month_avg_sales'] = None  # Initialize the column

    for i, row in group.iterrows():
        # Get the year and month of the current row
        current_year = row['year']
        current_month = row['month']

        # Determine the previous month and year
        if current_month == 1:  # Handle January (previous month is December of the previous year)
            prev_month = 12
            prev_year = current_year - 1
        else:   
            prev_month = current_month - 1
            prev_year = current_year

        # Calculate the average sales in the previous month
        prev_month_data = group[
            (group['year'] == prev_year) & (group['month'] == prev_month) & (group['sales'].notnull())
        ]

        avg_sales = prev_month_data['sales'].mean() if not prev_month_data.empty else 0
        group.at[i, 'prev_month_avg_sales'] = avg_sales  # Assign the calculated value

    return group


def shift_to_next_month(row):
    if row['month'] == 12:  # If January, go to December of the previous year
        return pd.Series({'month': 1, 'year': row['year'] + 1})
    else:  # Otherwise, simply decrement the month
        return pd.Series({'month': row['month'] + 1, 'year': row['year']})