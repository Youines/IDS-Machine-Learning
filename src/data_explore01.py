"""
Step 1:  Data Exploration
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Root directory configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load datasets
df_monday = pd.read_csv(DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv")
df_ddos = pd.read_csv(DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df_portscan = pd.read_csv(DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

print("MONDAY:", df_monday.shape)
print("DDOS:", df_ddos.shape)
print("PORTSCAN:", df_portscan.shape)


print("\n" + "=" * 80)
print("GENERAL INFORMATION")
print("=" * 80)

print(f"\nMonday (Normal Traffic):")
print(f"  Rows: {len(df_monday):,}")
print(f"  Columns: {len(df_monday.columns)}")


print(f"\nFriday DDoS:")
print(f"  Rows: {len(df_ddos):,}")
print(f"  Columns: {len(df_ddos.columns)}")


print(f"\nFriday PortScan:")
print(f"  Rows: {len(df_portscan):,}")
print(f"  Columns: {len(df_portscan.columns)}")


print("\n" + "=" * 80)
print("ACTUAL COLUMN NAMES (First 15)")
print("=" * 80)

print("\nMonday dataset column names (showing spaces):")
for i, col in enumerate(df_monday.columns[:15], 1):
    print(f"  {i:2d}. '{col}'")
  

# Columns information
print("\n" + "=" * 80)
print("COLUMN INFORMATION")
print("=" * 80)

print(f"\nTotal number of columns: {len(df_monday.columns)}")

# Check if columns are identical across datasets
print(f"\nColumns identical across all datasets?")
monday_cols = set(df_monday.columns)
ddos_cols = set(df_ddos.columns)
portscan_cols = set(df_portscan.columns)
print(f"  Monday vs DDoS: {monday_cols == ddos_cols}")
print(f"  Monday vs PortScan: {monday_cols == portscan_cols}")
print(f"  All identical: {monday_cols == ddos_cols == portscan_cols}")


if not (monday_cols == ddos_cols == portscan_cols):
    print("\nColumn differences found:")
    all_cols = monday_cols.union(ddos_cols).union(portscan_cols)
    for col in sorted(all_cols):
        in_monday = "true" if col in monday_cols else "false"
        in_ddos = "true" if col in ddos_cols else "false"
        in_portscan = "true" if col in portscan_cols else "false"
        print(f"  '{col}': Monday{in_monday} DDoS{in_ddos} PortScan{in_portscan}")

# Label distribution - using the exact column name from your output
print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

# Find the exact label column name
label_column = None
for col in df_monday.columns:
    if 'label' in col.lower() or 'Label' in col:
        label_column = col
        print(f"Found label column: '{label_column}'")
        break

if label_column:
    print(f"\nMonday (Normal Traffic):")
    label_counts = df_monday[label_column].value_counts()
    print(label_counts)
    total = len(df_monday)
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")

    print(f"\nFriday DDoS:")
    label_counts = df_ddos[label_column].value_counts()
    print(label_counts)
    total = len(df_ddos)
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")

    print(f"\nFriday PortScan:")
    label_counts = df_portscan[label_column].value_counts()
    print(label_counts)
    total = len(df_portscan)
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
else:
    print("Label column not found!")

# Missing values
print("\n" + "=" * 80)
print(" MISSING VALUES ANALYSIS")
print("=" * 80)

print(f"\nMonday: {df_monday.isnull().sum().sum()} missing values")
print(f"DDoS: {df_ddos.isnull().sum().sum()} missing values")
print(f"PortScan: {df_portscan.isnull().sum().sum()} missing values")

# Check for columns with missing values
print("\nColumns with missing values:")
for df_name, df in [("Monday", df_monday), ("DDoS", df_ddos), ("PortScan", df_portscan)]:
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"\n  {df_name}: {len(missing_cols)} columns with missing values")
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"    - '{col}': {missing_count:,} missing ({missing_pct:.4f}%)")
    else:
        print(f"\n  {df_name}: No missing values ✓")

# Statistical analysis - using safe column selection
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

# First, let's find what columns we actually have
print("\nAvailable columns that match our search:")
search_terms = ['Flow Duration', 'Fwd Packets', 'Backward Packets', 
                'Length of Fwd', 'Length of Bwd', 'Packet Length']

available_columns = []
for col in df_monday.columns:
    for term in search_terms:
        if term.lower() in col.lower():
            available_columns.append(col)
            print(f"  Found: '{col}'")
            break

# Take the first few available columns for analysis
if available_columns:
    key_columns = available_columns[:7]  # Take first 7 available columns
    print(f"\nUsing these columns for statistical analysis:")
    for i, col in enumerate(key_columns, 1):
        print(f"  {i}. '{col}'")
    
    print("\n" + "-" * 40)
    print("Statistics for DDoS attack traffic:")
    print("-" * 40)
    print(df_ddos[key_columns].describe().round(2))
    
    print("\n" + "-" * 40)
    print("Statistics for PortScan attack traffic:")
    print("-" * 40)
    print(df_portscan[key_columns].describe().round(2))
    
    print("\n" + "-" * 40)
    print("Statistics for Normal traffic (Monday):")
    print("-" * 40)
    print(df_monday[key_columns].describe().round(2))
else:
    print("No matching columns found for statistical analysis")

# Additional analysis: Data types
print("\n" + "=" * 80)
print(" DATA TYPES INFORMATION")
print("=" * 80)

print("\nData type distribution (Monday dataset):")
dtype_counts = df_monday.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# Check column names with spaces issues
print("\n" + "=" * 80)
print(" COLUMN NAME CLEANING CHECK")
print("=" * 80)

print("\nColumns with leading/trailing spaces:")
columns_with_spaces = []
for col in df_monday.columns:
    if col != col.strip():
        columns_with_spaces.append(col)
        print(f"  '{col}' -> should be '{col.strip()}'")

if not columns_with_spaces:
    print("  No columns with extra spaces found ✓")

# Show sample of actual data
print("\n" + "=" * 80)
print(" DATA PREVIEW ")
print("=" * 80)

print("\nMonday dataset (normal traffic) - first 2 rows:")
print(df_monday.head(2))

print("\nColumn names in Monday dataset:")
print([f"'{col}'" for col in df_monday.columns[:10]])

# Summary
print("\n" + "=" * 80)
print("EXPLORATION SUMMARY")
print("=" * 80)

print("\nKey findings:")
print(f"1. Dataset sizes: Monday={len(df_monday):,}, DDoS={len(df_ddos):,}, PortScan={len(df_portscan):,}")
print(f"2. All datasets have {len(df_monday.columns)} columns")
print(f"3. Label distribution:")
print(f"   - Monday: 100% BENIGN")
print(f"   - DDoS: {len(df_ddos[df_ddos[label_column] == 'DDoS']):,} DDoS ({len(df_ddos[df_ddos[label_column] == 'DDoS']) / len(df_ddos) * 100:.1f}%)")
print(f"   - PortScan: {len(df_portscan[df_portscan[label_column] == 'PortScan']):,} PortScan ({len(df_portscan[df_portscan[label_column] == 'PortScan']) / len(df_portscan) * 100:.1f}%)")
print(f"4. Missing values: Mostly in 'Flow Bytes/s' column")
print(f"5. Memory usage: {df_monday.memory_usage(deep=True).sum() / 1024**2:.1f} MB (Monday)")

print(f"\nTotal records across all datasets: {(len(df_monday) + len(df_ddos) + len(df_portscan)):,}")
print(f"Attack records: {len(df_ddos[df_ddos[label_column] != 'BENIGN']) + len(df_portscan[df_portscan[label_column] != 'BENIGN']):,}")
print(f"Normal records: {len(df_monday[df_monday[label_column] == 'BENIGN']) + len(df_ddos[df_ddos[label_column] == 'BENIGN']) + len(df_portscan[df_portscan[label_column] == 'BENIGN']):,}")

