"""
STEP 3: FEATURE SELECTION FOR INTRUSION DETECTION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import os

print("=" * 80)
print("STEP 3: FEATURE SELECTION")
print("=" * 80)


# 1. LOAD CLEANED DATASET

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load the CLEANED dataset (not the raw ones!)
cleaned_file = DATA_DIR / "dataset_clean.csv"

if not cleaned_file.exists():
    print(f" File not found: {cleaned_file}")
    print("Please run preprocessing first!")
    exit(1)

df = pd.read_csv(cleaned_file)
print(f" Cleaned dataset loaded: {df.shape}")


# 2. CHECK TARGET COLUMN

if 'is_attack' in df.columns:
    label_col = 'is_attack'
elif 'Attack' in df.columns:
    label_col = 'Attack'
else:
    print(" Target column not found!")
    print("Available columns:", df.columns.tolist())
    exit(1)

print(f"Target column: '{label_col}'")
print(f"Label distribution: {df[label_col].value_counts().to_dict()}")


# 3. SEPARATE FEATURES AND TARGET

X = df.drop(label_col, axis=1)
y = df[label_col]

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")


# 4. CALCULATE FEATURE IMPORTANCE

print("\n" + "=" * 80)
print("4. Calculating feature importance using Random Forest...")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  
rf.fit(X, y)

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

# Create DataFrame with sorted importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_importances.index,
    'Importance': feature_importances.values
}).sort_values('Importance', ascending=False)

print("✓ Random Forest trained")
print(f"\n Top 20 Most Important Features:")
print(feature_importance_df.head(20).to_string(index=False))

# 5. STATISTICAL ANALYSIS

print("\n" + "=" * 80)
print("5. Feature importance statistics:")

print(f"  Total features: {len(feature_importances)}")
print(f"  Average importance: {feature_importances.mean():.6f}")
print(f"  Max importance: {feature_importances.max():.6f}")
print(f"  Min importance: {feature_importances.min():.6f}")
print(f"  Standard deviation: {feature_importances.std():.6f}")

# Check how many features have meaningful importance
thresholds = [0.01, 0.005, 0.001]
for threshold in thresholds:
    n_features = (feature_importances > threshold).sum()
    percentage = (n_features / len(feature_importances)) * 100
    print(f"  Features > {threshold}: {n_features} ({percentage:.1f}%)")


# 6. SELECT BEST FEATURES

print("\n" + "=" * 80)
print("6. Selecting best features for IDS...")

selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature'].tolist()

print(f"\n Selected features: {len(selected_features)}")
print("\nList of selected features:")
for i, feat in enumerate(selected_features, 1):
    imp = feature_importance_df[feature_importance_df['Feature'] == feat]['Importance'].values[0]
    print(f"  {i:2d}. {feat:<40} (Importance: {imp:.4f})")

# 7. VISUALIZATION

print("\n" + "=" * 80)
print("7. Visualizing feature importance...")

# Create results directory
os.makedirs(BASE_DIR / "results", exist_ok=True)

top_n = min(20, len(feature_importance_df))
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(top_n)

# Create horizontal bar chart
bars = plt.barh(range(top_n), top_features['Importance'], color='steelblue', edgecolor='black')
plt.yticks(range(top_n), top_features['Feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.title(f'Top {top_n} Most Important Features for Intrusion Detection', 
          fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()  # Most important at top
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()

# Save figure
output_png = BASE_DIR / "results" / f"top_{top_n}_features.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"✓ Graph saved: {output_png}")
plt.show()

# 8. SAVE RESULTS
print("\n" + "=" * 80)
print("8. Saving results...")

# Save to CSV
output_csv = BASE_DIR / "results" / "feature_importances.csv"
feature_importance_df.to_csv(output_csv, index=False)
print(f"✓ CSV saved: {output_csv}")

# Save selected features list
import pickle
with open(DATA_DIR / "selected_features.pkl", 'wb') as f:
    pickle.dump(selected_features, f)
print(f"✓ Pickle file saved: {DATA_DIR / 'selected_features.pkl'}")

# Save to text file
os.makedirs(BASE_DIR / "docs", exist_ok=True)
with open(BASE_DIR / 'docs' / "04_feature_selection.txt", 'w', encoding='utf-8') as f:
    f.write(f"SELECTED FEATURES FOR INTRUSION DETECTION ({len(selected_features)})\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Importance threshold: > 0.01\n")
    f.write(f"Dataset: {df.shape}\n")
    f.write(f"Target column: '{label_col}'\n")
    f.write(f"Label distribution: {dict(y.value_counts())}\n\n")
    
    f.write("TOP 20 FEATURES:\n")
    f.write("-" * 80 + "\n")
    for i, row in feature_importance_df.head(20).iterrows():
        marker = " [SELECTED]" if row['Feature'] in selected_features else ""
        f.write(f"{i+1:2d}. {row['Feature']:<45} Importance: {row['Importance']:.4f}{marker}\n")
    
    f.write("\n\nALL SELECTED FEATURES:\n")
    f.write("-" * 80 + "\n")
    for i, feat in enumerate(selected_features, 1):
        imp = feature_importance_df[feature_importance_df['Feature'] == feat]['Importance'].values[0]
        f.write(f"{i:3d}. {feat:<45} Importance: {imp:.4f}\n")

print(f"Text report saved: {BASE_DIR / 'docs' / '04_feature_selection.txt'}")

