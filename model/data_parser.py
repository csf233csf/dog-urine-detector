import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_table_name(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table = cursor.fetchone()
    conn.close()
    return table[0] if table else None

def read_table_to_df(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def preprocess_sensor_data(df):
    df_pivot = df.pivot(index='timestamp', columns='sensor_type', values='value')
    df_pivot.dropna(inplace=True)  # Drop rows with NaN values
    return df_pivot

def group_and_flatten(df, group_size=6):
    return df.groupby(np.arange(len(df)) // group_size).apply(lambda x: x.values.flatten()).apply(pd.Series)

def align_data(df1, df2):
    df1.index = pd.to_datetime(df1.index)
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    df1 = df1[~df1.index.duplicated(keep='first')]
    df2 = df2[~df2['timestamp'].duplicated(keep='first')]
    combined_df = pd.merge_asof(df1.sort_index(), df2.sort_values('timestamp'), left_index=True, right_on='timestamp', direction='nearest')
    combined_df.dropna(inplace=True)  # Drop rows with NaN values after merging
    return combined_df

# Get table name and read data from the first database (sensor data)
db1_path = 'model/data/sensor_data.db'
table1 = get_table_name(db1_path)

if table1:
    df1 = read_table_to_df(db1_path, table1)
    print("\nDataframe from the first database:")
    print(df1.head())
    df1_pivot = preprocess_sensor_data(df1)
    print("\nPivoted Dataframe from the first database:")
    print(df1_pivot.head())

# Get table name and read data from the second database (water sensor data)
db2_path = 'model/data/water_sensor.db'
table2 = get_table_name(db2_path)

if table2:
    df2 = read_table_to_df(db2_path, table2)
    print("\nDataframe from the second database:")
    print(df2.head())

# Align data based on timestamps
combined_df = align_data(df1_pivot, df2)
print("\nCombined Dataframe:")
print(combined_df.head())

# Group and flatten the sensor data
group_size = 6
grouped_sensor_data = group_and_flatten(combined_df.drop(columns=['timestamp', 'water_level']), group_size=group_size)
print("\nGrouped and Flattened Sensor Data:")
print(grouped_sensor_data.head())

# Check for NaN values in the grouped sensor data
print("\nChecking for NaN values in grouped sensor data...")
nan_rows = grouped_sensor_data[grouped_sensor_data.isna().any(axis=1)]
print(f"Rows with NaN values:\n{nan_rows}")

# Drop rows with NaN values
grouped_sensor_data.dropna(inplace=True)

# Randomly label a realistic percentage of the data (e.g., 5%)
urinating_percentage = 5 / 100
np.random.seed(42)
num_labels = int(urinating_percentage * len(grouped_sensor_data))
label_indices = np.random.choice(len(grouped_sensor_data), num_labels, replace=False)
labels = np.zeros(len(grouped_sensor_data), dtype=int)
labels[label_indices] = 1  # Label 1 for urinating, 0 for not urinating

# Prepare features
features = grouped_sensor_data.values

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Save the processed data to CSV files
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train
train_df.to_csv('train_data.csv', index=False)

test_df = pd.DataFrame(X_test)
test_df['label'] = y_test
test_df.to_csv('test_data.csv', index=False)

print("\nTraining Features Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Testing Labels Shape:", y_test.shape)
