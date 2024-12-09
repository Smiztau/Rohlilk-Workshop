import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load CSV file
df_sales_train = pd.read_csv('sales_train.csv')
df_inventory = pd.read_csv('inventory.csv')
df_calender = pd.read_csv('calendar.csv')
df_weights = pd.read_csv('test_weights.csv')

df_calender['holiday_name'].fillna("-", inplace = True)

df_inventory.drop('warehouse', axis=1, inplace=True)

labels = df_sales_train["sales"]
labels.to_csv('train_labels.csv', index= False)

df_sales_train1 = pd.merge(df_sales_train, df_inventory, on='unique_id', how='left')

df_sales_train2 = pd.merge(df_sales_train1, df_calender, on=['date','warehouse'], how = 'left')

df_sales_train3 = pd.merge(df_sales_train2, df_weights, on='unique_id', how = 'left')

df_sales_train3 = df_sales_train3.dropna(subset=['sales'])

df_sales_train3.to_csv('merged_data.csv', index = False)




