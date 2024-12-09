import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


print("start")
# Read the data
df_features = pd.read_csv('merged_data.csv')
df_features = df_features.dropna(subset=['sales'])

labels = df_features['sales']
print("begin")

# Prepare features
df_encoded = pd.get_dummies(df_features, columns=['warehouse', 'name','L1_category_name_en','L2_category_name_en','L3_category_name_en',
                                                  'L4_category_name_en','holiday_name'])
print(1)
# Convert date to datetime and extract day, month, year
df_encoded['date'] = pd.to_datetime(df_features['date'])
df_encoded['day'] = df_encoded['date'].dt.day
df_encoded['month'] = df_encoded['date'].dt.month
df_encoded['year'] = df_encoded['date'].dt.year
print(2)

# Drop the original 'date' column
df_encoded.drop('date', axis= 1, inplace=True)
print(3)

# Features and target variable
X = df_encoded.drop(['sales', 'weight'], axis=1)
y = df_features['sales']
sample_weights = df_features['weight']
print(5)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)
print(6)

# Initialize the XGBRegressor

model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='mae',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,  # Only 1 estimator per update, we'll update iteratively
    random_state=42
)
print(7)

# Batch training: split the data into smaller batches
batch_size = 1500000  # Reduce the batch size to manage memory
num_batches = len(X_train) // batch_size
print(8)

# Train in batches
for batch_idx in range(num_batches + 1):
    print(batch_idx)
    # Define the current batch
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size
    if end_idx > len(X_train):
        end_idx = len(X_train)
    
    # Get the current batch of data
    batch_X_train = X_train.iloc[start_idx:end_idx]
    batch_y_train = y_train.iloc[start_idx:end_idx]
    batch_sample_weights_train = sample_weights_train.iloc[start_idx:end_idx]
    
    # Fit the model on the current batch
    model.fit(batch_X_train, batch_y_train, sample_weight=batch_sample_weights_train, 
              eval_set=[(batch_X_train, batch_y_train), (X_test, y_test)], 
              verbose=True)

    # Optionally, you could store the trained model at each step for later use:
    # model.save_model(f"model_batch_{batch_idx}.json")

print("Batch training completed")

# Make predictions on the test set
ypred = model.predict(X_test)

# Calculate the Weighted Mean Absolute Error (WMAE)
absolute_error = abs(y_test - ypred)
wmae = (absolute_error * sample_weights_test).sum() / sample_weights_test.sum()

print("WMAE:", wmae)

