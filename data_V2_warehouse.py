import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import os
from utils import *

encoder = LabelEncoder()

# Load CSV file
df_sales_train = pd.read_csv(sales_train)
df_sales_test = pd.read_csv(sales_test)
df_inventory = pd.read_csv(inventory)
df_calender = pd.read_csv(calendar)
df_weights = pd.read_csv(weights)
df_word2vec = pd.read_csv(food_embedings)
df_avg_sales_combined = pd.read_csv(rolling)

labels = df_sales_train["sales"]
labels.to_csv(labels_csv, index= False)

df_inventory.drop('warehouse', axis=1, inplace=True)

df_calender['date'] = pd.to_datetime(df_calender['date'], format="mixed", dayfirst=True)
df_calender['day'] = df_calender['date'].dt.day
df_calender['month'] = df_calender['date'].dt.month
df_calender['year'] = df_calender['date'].dt.year
df_calender.drop('date', axis=1, inplace=True)
df_calender['is_holiday'] = (df_calender['holiday'] | df_calender['winter_school_holidays'] | df_calender['school_holidays']).astype(int)

df_sales_train = pd.merge(df_sales_train, df_weights, on='unique_id', how = 'left')
df_sales_train.dropna(subset=['sales'], inplace= True)
df_sales_train.drop('availability', axis=1, inplace=True)


for df_name, df_sales in [('df_sales_train', df_sales_train), ('df_sales_test', df_sales_test)]:
    
    df_sales = pd.merge(df_sales, df_inventory, on='unique_id', how='left')
    df_sales[['name_only', 'name_number']] = df_sales['name'].str.split('_', expand=True)
    df_sales.drop('name', axis=1, inplace=True)
    df_sales['L2_category_name_en'] = df_sales['L2_category_name_en'].str.split('_').str[-1]
    df_sales['L2_category_name_en'] = pd.to_numeric(df_sales['L2_category_name_en'])
    df_sales['L3_category_name_en'] = df_sales['L3_category_name_en'].str.split('_').str[-1]
    df_sales['L3_category_name_en'] = pd.to_numeric(df_sales['L3_category_name_en'])
    df_sales['L4_category_name_en'] = df_sales['L4_category_name_en'].str.split('_').str[-1]
    df_sales['L4_category_name_en'] = pd.to_numeric(df_sales['L4_category_name_en'])
    df_sales['max_discount'] = df_sales[['type_0_discount', 'type_1_discount', 'type_2_discount',
                            'type_3_discount', 'type_4_discount', 'type_5_discount',
                            'type_6_discount']].max(axis=1)
    df_sales['final_price'] = df_sales['sell_price_main'] * (1 - df_sales['max_discount'])
    df_sales.drop('type_0_discount', axis=1, inplace=True)
    df_sales.drop('type_1_discount', axis=1, inplace=True)
    df_sales.drop('type_2_discount', axis=1, inplace=True)
    df_sales.drop('type_3_discount', axis=1, inplace=True)
    df_sales.drop('type_4_discount', axis=1, inplace=True)
    df_sales.drop('type_5_discount', axis=1, inplace=True)
    df_sales.drop('type_6_discount', axis=1, inplace=True)


    df_sales['date'] = pd.to_datetime(df_sales['date'], format="mixed", dayfirst=True)
    df_sales['day'] = df_sales['date'].dt.day
    df_sales['month'] = df_sales['date'].dt.month
    df_sales['year'] = df_sales['date'].dt.year
    df_sales['weekday'] = df_sales['date'].dt.weekday
    df_sales["week_of_the_month"] = ((df_sales["date"].dt.day - 1) // 7) + 1
    corona_start_date = '01-01-2021'
    corona_end_date   = '01-06-2022'

    df_sales['corona_virus'] = df_sales['date'].between(corona_start_date, corona_end_date)
    df_sales['month_cos'] = np.cos(2 * np.pi * df_sales['month']/12)
    df_sales['month_sin'] = np.sin(2 * np.pi * df_sales['month']/12)
    df_sales['day_cos'] = np.cos(2 * np.pi * df_sales['day']/12)
    df_sales['day_sin'] = np.sin(2 * np.pi * df_sales['day']/12)
    # df_sales.drop('date', axis=1, inplace=True)

    df_sales = pd.merge(df_sales, df_calender, on=['day', 'month', 'year','warehouse'], how='left')

    df_sales = pd.merge(df_sales, df_word2vec, on=["name_only"], how='left')
    df_sales.drop('name_only', axis=1, inplace=True)

    df_sales = pd.get_dummies(df_sales, columns=['L1_category_name_en'])

    df_sales.drop('holiday_name', axis=1, inplace=True)
    
    # Save the processed dataframe back to the correct variable
    if df_name == 'df_sales_train':
        df_sales_train = df_sales
    elif df_name == 'df_sales_test':
        df_sales_test = df_sales
    

print("<<<<<<<<<<<<<<<<<<<<FINISH FOR LOOP>>>>>>>>>>>>>>>>>>>")    

df_sales_train["isTrain"] = True
df_sales_test["isTrain"] = False
df_combined = pd.concat([df_sales_train, df_sales_test], ignore_index=True)

avg_sales_prev_month = (
    df_combined
    .groupby(['unique_id', 'year', 'month'], as_index=False)['sales']
    .mean()
    .rename(columns={'sales': 'avg_sales_unique_id_month'})
)

avg_sales_prev_month[['month', 'year']] = avg_sales_prev_month.apply(shift_to_next_month, axis=1)
# Step 3: Create a full calendar of months for all unique_ids
# Get the unique IDs
unique_ids = df_combined['unique_id'].unique()

# Generate all months from 2020 to 2024
date_range = pd.date_range(start='2020-01-01', end='2024-12-31', freq='MS')  # MS = Month Start
calendar = pd.DataFrame({
    'year': date_range.year,
    'month': date_range.month
})

# Create a cartesian product of unique_ids and the calendar
full_calendar = (
    pd.DataFrame({'unique_id': unique_ids})
    .merge(calendar, how='cross')
)

# Step 4: Merge the full calendar with the precomputed data
full_data = full_calendar.merge(
    avg_sales_prev_month,
    on=['unique_id', 'year', 'month'],
    how='left'
)

# # Step 5: Forward-fill missing values within each unique_id
full_data = full_data.sort_values(by=['unique_id', 'year', 'month'])
# Forward-fill missing avg_sales_unique_id_month values within each unique_id
full_data['avg_sales_unique_id_month'] = (
    full_data.groupby('unique_id', group_keys=False)['avg_sales_unique_id_month']
    .apply(lambda group: group.ffill())
)
full_data['avg_sales_unique_id_month'] = (
    full_data.groupby('unique_id', group_keys=False)['avg_sales_unique_id_month']
    .apply(lambda group: group.bfill())
)

df_combined = df_combined.merge(
    full_data,
    on=['unique_id', 'year', 'month'],
    how='left'
)

# Step 7: Remove rows where avg_sales_unique_id_month is empty
df_combined = df_combined[df_combined['avg_sales_unique_id_month'].notnull()]

# Reset index for cleanliness
df_combined.reset_index(drop=True, inplace=True)
print(df_combined["date"].size)
df_combined.drop('date', axis=1, inplace=True)

print("<<<<<<<<<<<<<<<<<<<<DF_COMBINED CREATED>>>>>>>>>>>>>>>>>>>")    
df_sales_train = df_combined[df_combined['isTrain']==True].drop('isTrain', axis=1)
df_sales_test = df_combined[df_combined['isTrain']==False].drop('isTrain', axis=1).drop('sales', axis=1).drop("weight", axis=1)
df_sales_train.reset_index(drop=True, inplace=True)
df_sales_test.reset_index(drop=True, inplace=True)

os.makedirs(csv_junk, exist_ok=True)
df_sales_test.to_csv(merged_data_test, index=False)
df_sales_train.to_csv(merged_data_train, index=False)
df_combined.to_csv("csv_junk/abc.csv") 
print("<<<<<<<<<<<<<<<<<<<<FINITO>>>>>>>>>>>>>>>>>>>")    