import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

    
isTrain = 0
if(isTrain):
    fileName = 'sales_train.csv'
    fileOutName = 'merged_data_train.csv'
else:
    fileName = 'sales_test.csv'
    fileOutName = 'merged_data_test.csv'
    

encoder = LabelEncoder()

# Load CSV file
df_sales = pd.read_csv(fileName)
df_inventory = pd.read_csv('inventory.csv')
df_calender = pd.read_csv('calendar.csv')
df_weights = pd.read_csv('test_weights.csv')

# df_calender['holiday_name'].fillna("-", inplace = True)
df_inventory.drop('warehouse', axis=1, inplace=True)
df = pd.merge(df_sales, df_inventory, on='unique_id', how='left')

if(isTrain):
    labels = df_sales["sales"]
    labels.to_csv('train_labels.csv', index= False)
    df = pd.merge(df, df_weights, on='unique_id', how = 'left')
    df = df.dropna(subset=['sales'])
    df.drop('availability', axis=1, inplace=True)

df['warehouse_city'] = df['warehouse'].str.split('_').str[-1]
# df.drop('warehouse', axis=1, inplace=True)
# df['warehouse_number'] = pd.to_numeric(df['warehouse_number'])
df[['name_only', 'name_number']] = df['name'].str.split('_', expand=True)
df.drop('name', axis=1, inplace=True)
df['name_number'] = pd.to_numeric(df['name_number'])
df['name_only'] = encoder.fit_transform(df['name_only'])
# df['L1_category_name_en'] = encoder.fit_transform(df['L1_category_name_en'])
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
df.drop('date', axis=1, inplace=True)


# df_calender['warehouse_city'] = df_calender['warehouse'].str.split('_').str[-1]
# df_calender['warehouse_number'] = pd.to_numeric(df_calender['warehouse_number'])
df_calender['date'] = pd.to_datetime(df_calender['date'], format="mixed", dayfirst=True)
df_calender['day'] = df_calender['date'].dt.day
df_calender['month'] = df_calender['date'].dt.month
df_calender['year'] = df_calender['date'].dt.year
df_calender.drop('date', axis=1, inplace=True)
df = pd.merge(df, df_calender, on=['day', 'month', 'year','warehouse'], how='left')

df = pd.get_dummies(df, columns=['warehouse_city', 'L1_category_name_en'])
df['warehouse'] = encoder.fit_transform(df['warehouse'])


##### to remove
df.drop('holiday_name', axis=1, inplace=True)
##### 

df.to_csv(fileOutName, index=False)
print("<<<<<<<<<<<<<<<<<<<<FINISH>>>>>>>>>>>>>>>>>>>")