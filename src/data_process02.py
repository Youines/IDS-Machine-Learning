"""
Step 2:  Data Preprocessing
"""


import pandas as pd
import numpy as np
from pathlib import Path


#PHASE 1: DATA PREPROCESSING
print("=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


df_monday = pd.read_csv(DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv")
df_ddos = pd.read_csv(DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df_portscan = pd.read_csv(DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

# 1 ) Combine all datasets
df_combined = pd.concat([df_monday, df_ddos, df_portscan], ignore_index=True)
print(f"\n✓ Combined dataset: {len(df_combined):,} rows, {df_combined.shape[1]} columns")

df_combined.columns =df_combined.columns.str.strip()
if 'Label' not in df_combined.columns:
    for col in df_combined.columns:
        if 'label' in col.lower():
            df_combined.rename(columns={col: 'Label'}, inplace=True)
            print(f" Renamed column '{col}' to 'Label'")
            break

#  2 ) Handle infinite values
print("\nHandling infinite values...")
numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
inf_counts = (df_combined[numeric_cols] == np.inf).sum()
# replace infinite values with NaN
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
#count after 
inf_after = np.isinf(df_combined.select_dtypes(include=[np.number])).sum().sum()
print(f"Infinite values after: {inf_after}")
#  3 ) Handle missing values
missing_values=df_combined.isnull().sum().sum()
print(f"\nMissing values before: {missing_values}")
df_cleaned = df_combined.dropna()
missing_after=df_cleaned.isnull().sum().sum()
print(f"Missing values after: {missing_after}")
# 4 ) Encode labels
print("\nEncoding labels...")
df_clean = df_cleaned.copy()
df_clean['is_attack'] = (df_clean['Label'] != 'BENIGN').astype(int)
attack_counts = df_clean['is_attack'].value_counts()
print(attack_counts)
#  5 ) remove unnecessary columns
columns_to_remove = [
    'Flow ID',      # Unique identifier (not useful )
    'Source IP',    # IP addresses (don't generalize well)
    'Destination IP',
    'Timestamp',    # Timestamp (time-based, not useful)
    'Label'         # Original label (we have 'is_attack')
]
removed_count = 0
for col in columns_to_remove:
    if col in df_clean.columns:
        df_clean.drop(col, axis=1, inplace=True)
        print(f"   Removed: {col}")
        removed_count += 1
    else:
        print(f"   Column '{col}' not found")

print(f"\nColumns removed: {removed_count}")
# 6 ) Save cleaned dataset
# Save full cleaned dataset
output_file = DATA_DIR / "dataset_clean.csv"
df_clean.to_csv(output_file, index=False)
print(f"✓ Cleaned dataset saved: {output_file}")
print(f"  Rows: {len(df_clean):,}, Columns: {len(df_clean.columns)}")