import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.preprocessing import StandardScaler

print("="*60)
print("STLGRU DATA PREPARATION FOR PIOMBINO")
print("="*60)

# Configuration
HISTORY = 12  # 12 steps = 3 hours (15-min intervals)
HORIZON = 3   # 3 steps = 45 minutes
THRESHOLD = 0.5  # Correlation threshold for adjacency matrix

# Step 1: Load data from the correct path
data_path = "data/Traffic_data_piombino_2025 -1.parquet"
print(f"\n Loading data from: {data_path}")

if not os.path.exists(data_path):
    print(f" ERROR: File not found at {data_path}")
    print("\nPlease check that the file exists:")
    os.system("ls -la data/")
    exit(1)

df = pd.read_parquet(data_path)
print(f" Data loaded successfully!")
print(f"Raw data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Step 2: Select relevant columns
df = df[['Date_out', 'Sensor', 'Flux']].copy()
df['Date_out'] = pd.to_datetime(df['Date_out'])
df['Flux'] = df['Flux'].fillna(0)

print(f"\nUnique sensors: {df['Sensor'].unique()}")
print(f"Date range: {df['Date_out'].min()} to {df['Date_out'].max()}")
print(f"Total rows: {len(df)}")

# Step 3: Pivot to matrix format
print("\n Pivoting data...")
pivot = df.pivot_table(
    index='Date_out',
    columns='Sensor',
    values='Flux'
).sort_index().fillna(0)

print(f"Pivot shape: {pivot.shape}")
print(f"Sensors: {list(pivot.columns)}")
print(f"Time steps: {len(pivot)}")

# Step 4: Chronological split (NO RANDOM SHUFFLE!)
T = len(pivot)
train_end = int(T * 0.7)
val_end = int(T * 0.85)  # 70% train, 15% val, 15% test

train_df = pivot.iloc[:train_end]
val_df = pivot.iloc[train_end:val_end]
test_df = pivot.iloc[val_end:]

print(f"\n Split sizes:")
print(f"  Train: {train_df.shape} ({len(train_df)/T:.1%})")
print(f"  Val: {val_df.shape} ({len(val_df)/T:.1%})")
print(f"  Test: {test_df.shape} ({len(test_df)/T:.1%})")

# Step 5: Normalization (fit ONLY on training data)
print("\n Normalizing data...")
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(" Scaler saved")

# Step 6: Build adjacency matrix from TRAINING data correlation
print("\ Building graph adjacency matrix...")
corr = np.corrcoef(train_scaled.T)
adj_matrix = (np.abs(corr) > THRESHOLD).astype(float)
np.fill_diagonal(adj_matrix, 1)  # Add self-loops

print(f"Adjacency matrix:\n{adj_matrix}")
print(f"Graph density: {np.sum(adj_matrix)/(adj_matrix.shape[0]*adj_matrix.shape[1]):.2%}")
np.save("adj_matrix.npy", adj_matrix)

# Step 7: Create sliding windows
print("\n Creating sliding windows...")
def create_windows(data, history=HISTORY, horizon=HORIZON):
    X, Y = [], []
    for i in range(len(data) - history - horizon):
        X.append(data[i:i+history])
        Y.append(data[i+history:i+history+horizon])
    return np.array(X), np.array(Y)

X_train, Y_train = create_windows(train_scaled)
X_val, Y_val = create_windows(val_scaled)
X_test, Y_test = create_windows(test_scaled)

print(f"\n Dataset shapes:")
print(f"  X_train: {X_train.shape} (samples, time_steps, sensors)")
print(f"  Y_train: {Y_train.shape} (samples, horizon, sensors)")
print(f"  X_val: {X_val.shape}")
print(f"  Y_val: {Y_val.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  Y_test: {Y_test.shape}")

# Step 8: Save all arrays
print("\n Saving prepared data...")
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

# Step 9: Save metadata
metadata = {
    'num_sensors': len(pivot.columns),
    'sensor_names': list(pivot.columns),
    'history_length': HISTORY,
    'prediction_horizon': HORIZON,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'total_time_steps': len(pivot),
    'train_date_range': [str(train_df.index[0]), str(train_df.index[-1])],
    'val_date_range': [str(val_df.index[0]), str(val_df.index[-1])],
    'test_date_range': [str(test_df.index[0]), str(test_df.index[-1])],
    'adjacency_matrix': adj_matrix.tolist()
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print(" DATA PREPARATION COMPLETE!")
print("="*60)
print("\nFiles created:")
for f in ["X_train.npy", "Y_train.npy", "X_val.npy", "Y_val.npy", 
          "X_test.npy", "Y_test.npy", "adj_matrix.npy", "scaler.pkl", "metadata.json"]:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)  # Size in MB
        print(f"   {f} ({size:.2f} MB)")

print("\n Next step: Train the model with:")
print("    python train.py")
