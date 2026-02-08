"""
Step 4: Intrusion Detection Model Training
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from pathlib import Path

print("=" * 80)
print("STEP 4: MODEL TRAINING")
print("=" * 80)


print("\n1. Loading cleaned dataset...")



df = pd.read_csv('../data/dataset_clean.csv')

print(f"✓ Dataset loaded: {df.shape}")
print(f"✓ Columns: {len(df.columns)}")
print(f"Sample column names: {list(df.columns[:5])}")


# 2. Prepare features and target

selected_features = [
    'Average Packet Size',
    'Fwd Packet Length Max',
    'Subflow Fwd Bytes',
    'Fwd Packet Length Mean',
    'PSH Flag Count',
    'Avg Fwd Segment Size',
    'Packet Length Std',
    'Total Length of Fwd Packets',
    'Max Packet Length',
    'Bwd Packet Length Mean',
    'Init_Win_bytes_forward',
    'Bwd Packet Length Std',
    'Packet Length Variance',
    'Packet Length Mean',
    'Bwd Packets/s',
    'Bwd Packet Length Max',
    'ACK Flag Count',
    'Avg Bwd Segment Size',
    'Total Length of Bwd Packets',
    'Init_Win_bytes_backward',
    'Total Fwd Packets',
    'act_data_pkt_fwd',
    'Fwd Packet Length Min',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd IAT Total',
    'Subflow Bwd Bytes',
    'Fwd Header Length.1',
    'Flow Bytes/s',
    'Flow IAT Max',
    'Subflow Fwd Packets',
    'Bwd Packet Length Min',
    'Fwd IAT Std'
]

# 3. Prepare DATA

X = df[selected_features]
y = df['is_attack']
 
# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]:,} samples")
print(f" Test set:     {X_test.shape[0]:,} samples")

# 5. Normalization
scaler =StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

    # Save Scaler
with open('../data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 6. Decision Tree Classifier
print("\nTraining Decision Tree Classifier...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print("Decision Tree Performance:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_dt):.4f}")

# 7. Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest Performance:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_rf):.4f}")


# 9. Save models 
with open('../data/model_decision_tree.pkl', 'wb') as f:
    pickle.dump(dt, f)

with open('../data/model_random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)

print(" Models saved to ../data/")

print("\n" + "=" * 80)
print(" TRAINING COMPLETED!")
print("=" * 80)

