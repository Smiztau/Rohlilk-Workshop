import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import os
from utils import get_season

input_folder = "csv_input"
output_folder = "csv_junk"
isTrain = 1
if(isTrain):
    fileName = 'sales_train.csv'
    fileOutName = 'merged_data_train.csv'
else:
    fileName = 'sales_test.csv'
    fileOutName = 'merged_data_test.csv'
    

encoder = LabelEncoder()

# Load CSV file
df_sales = pd.read_csv(input_folder+"/"+fileName)
df_inventory = pd.read_csv(input_folder+'/inventory.csv')
df_calender = pd.read_csv(input_folder+'/calendar.csv')
df_weights = pd.read_csv(input_folder+'/test_weights.csv')
df_word2vec = pd.read_csv(output_folder+'/food_embeddings.csv')

# df_calender['holiday_name'].fillna("-", inplace = True)
df_inventory.drop('warehouse', axis=1, inplace=True)
df = pd.merge(df_sales, df_inventory, on='unique_id', how='left')

if(isTrain):
    labels = df_sales["sales"]
    labels.to_csv(input_folder+'/train_labels.csv', index= False)
    df = pd.merge(df, df_weights, on='unique_id', how = 'left')
    df = df.dropna(subset=['sales'])
    df.drop('availability', axis=1, inplace=True)

df[['name_only', 'name_number']] = df['name'].str.split('_', expand=True)
df.drop('name', axis=1, inplace=True)
df['L2_category_name_en'] = df['L2_category_name_en'].str.split('_').str[-1]
df['L2_category_name_en'] = pd.to_numeric(df['L2_category_name_en'])
df['L3_category_name_en'] = df['L3_category_name_en'].str.split('_').str[-1]
df['L3_category_name_en'] = pd.to_numeric(df['L3_category_name_en'])
df['L4_category_name_en'] = df['L4_category_name_en'].str.split('_').str[-1]
df['L4_category_name_en'] = pd.to_numeric(df['L4_category_name_en'])
df['max_discount'] = df[['type_0_discount', 'type_1_discount', 'type_2_discount',
                         'type_3_discount', 'type_4_discount', 'type_5_discount',
                         'type_6_discount']].max(axis=1)
df['final_price'] = df['sell_price_main'] * (1 - df['max_discount'])
df.drop('type_0_discount', axis=1, inplace=True)
df.drop('type_1_discount', axis=1, inplace=True)
df.drop('type_2_discount', axis=1, inplace=True)
df.drop('type_3_discount', axis=1, inplace=True)
df.drop('type_4_discount', axis=1, inplace=True)
df.drop('type_5_discount', axis=1, inplace=True)
df.drop('type_6_discount', axis=1, inplace=True)


df['date'] = pd.to_datetime(df['date'], format="mixed", dayfirst=True)
df['weekday'] = df['date'].dt.weekday
df["week_of_the_month"] = ((df["date"].dt.day - 1) // 7) + 1
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
corona_start_date = '01-01-2021'
corona_end_date   = '01-06-2022'

df['corona_virus'] = df['date'].between(corona_start_date, corona_end_date)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['day_cos'] = np.cos(2 * np.pi * df['day']/12)
df['day_sin'] = np.sin(2 * np.pi * df['day']/12)
# df['season'] = pd.to_datetime(df['date']).dt.month.apply(get_season)
# df['days_since_first_sale'] = (pd.to_datetime(df['date']) - df.groupby('unique_id')['date'].transform('min')).dt.days

# ##########################################
df.sort_values(['unique_id', 'date'], inplace=True)

# # Rolling Average Price (Last 2 Days) with fallback to Window=1
# df['rolling_avg_price'] = (
#     df.groupby('unique_id')['sell_price_main']
#     .transform(lambda x: x.rolling(window=2).mean())
#     .fillna(df['sell_price_main'])  # Fill NaN with Window=1 (current value)
# )

# # Rolling Total Orders (Last 2 Days) with fallback to Window=1
# df['rolling_total_orders'] = (
#     df.groupby('unique_id')['total_orders']
#     .transform(lambda x: x.rolling(window=2).sum())
#     .fillna(df['total_orders'])  # Fill NaN with Window=1 (current value)
# )

# # Cumulative Total Orders
# df['cumulative_total_orders'] = df.groupby('unique_id')['total_orders'].cumsum()

# # Create a lagged feature for sell_price
# df.sort_values(['unique_id', 'date'], inplace=True)
# df['lag_final_price'] = df.groupby('unique_id')['final_price'].shift(1)
# df['lag_final_price'].fillna(0, inplace=True)  # Replace NaNs with 0 or another default value

df.drop('date', axis=1, inplace=True)

df_calender['date'] = pd.to_datetime(df_calender['date'], format="mixed", dayfirst=True)
df_calender['day'] = df_calender['date'].dt.day
df_calender['month'] = df_calender['date'].dt.month
df_calender['year'] = df_calender['date'].dt.year
df_calender.drop('date', axis=1, inplace=True)
df_calender['is_holiday'] = (df_calender['holiday'] | df_calender['winter_school_holidays'] | df_calender['school_holidays']).astype(int)
df = pd.merge(df, df_calender, on=['day', 'month', 'year','warehouse'], how='left')

df = pd.merge(df, df_word2vec, on=["name_only"], how='left')
df.drop('name_only', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['L1_category_name_en'])


##### to remove
df.drop('holiday_name', axis=1, inplace=True)
##### 
os.makedirs(output_folder, exist_ok=True)
df.to_csv(output_folder+"/"+fileOutName, index=False)
print("<<<<<<<<<<<<<<<<<<<<FINISH>>>>>>>>>>>>>>>>>>>")