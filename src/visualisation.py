"""
Step 6: Final  Visualizations 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

print("=" * 80)
print("STEP 7: FINAL VISUALIZATIONS")
print("=" * 80)


# 1. LOAD DATA AND MODELS

print("\n1. Loading data and models...")

required_files = [
    '../data/dataset_clean.csv',
    '../data/selected_features.pkl',
    '../data/scaler.pkl',
    '../data/model_decision_tree.pkl',
    '../data/model_random_forest.pkl'
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Missing file: {file}")
df = pd.read_csv('../data/dataset_clean.csv')


with open('../data/selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

with open('../data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../data/model_decision_tree.pkl', 'rb') as f:
    dt = pickle.load(f)

with open('../data/model_random_forest.pkl', 'rb') as f:
    rf = pickle.load(f)

print(f"✓ Dataset: {df.shape}")
print(f"✓ Features: {len(selected_features)}")
print("✓ Models loaded")

# 2. PREPARE DATA

print("\n2. Preparing data...")

X = df[selected_features]
y = df['is_attack']

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred_dt = dt.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)

# 3. CALCULATE METRICS

print("\n3. Calculating metrics...")

# Calculate metrics
dt_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt),
    'recall': recall_score(y_test, y_pred_dt),
    'f1': f1_score(y_test, y_pred_dt)
}

rf_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf)
}

# Print metrics
print("\n MODEL METRICS:")
print("-" * 50)
print(f"Decision Tree:")
for metric, value in dt_metrics.items():
    print(f"  {metric.capitalize():10}: {value:.6f}")

print(f"\nRandom Forest:")
for metric, value in rf_metrics.items():
    print(f"  {metric.capitalize():10}: {value:.6f}")


# 4. CREATE VISUALIZATION - MODEL COMPARISON

print("\n4. Creating model comparison visualization...")

# Create DataFrame for visualization
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Decision Tree': [dt_metrics['accuracy'], dt_metrics['precision'], 
                     dt_metrics['recall'], dt_metrics['f1']],
    'Random Forest': [rf_metrics['accuracy'], rf_metrics['precision'], 
                     rf_metrics['recall'], rf_metrics['f1']]
}

metrics_df = pd.DataFrame(metrics_data)

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(metrics_df))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], metrics_df['Decision Tree'], 
               width, label='Decision Tree', color='steelblue', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], metrics_df['Random Forest'], 
               width, label='Random Forest', color='forestgreen', alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('IDS Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Metric'])
ax.legend()


ax.set_ylim(0.9995, 1.0001)  # Zoom approprié pour scores ~99.99%
ax.grid(axis='y', alpha=0.3, linestyle='--')


for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.00001,
                f'{height:.5f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ../results/model_comparison.png")
plt.show()


# 5. CREATE VISUALIZATION - ERROR ANALYSIS

print("\n5. Creating error analysis...")

# Calculate confusion matrices
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)


dt_tn, dt_fp, dt_fn, dt_tp = cm_dt.ravel()
rf_tn, rf_fp, rf_fn, rf_tp = cm_rf.ravel()

print(f"\nDecision Tree Errors:")
print(f"  False Positives: {dt_fp}")
print(f"  False Negatives: {dt_fn}")

print(f"\nRandom Forest Errors:")
print(f"  False Positives: {rf_fp}")
print(f"  False Negatives: {rf_fn}")

# Create error comparison
error_data = {
    'Error Type': ['False Positives', 'False Negatives'],
    'Decision Tree': [dt_fp, dt_fn],
    'Random Forest': [rf_fp, rf_fn]
}

error_df = pd.DataFrame(error_data)

# Create error comparison chart
fig, ax = plt.subplots(figsize=(8, 5))

x = range(len(error_df))
bars1 = ax.bar([i - 0.2 for i in x], error_df['Decision Tree'], 
               0.4, label='Decision Tree', color='lightcoral', edgecolor='darkred')
bars2 = ax.bar([i + 0.2 for i in x], error_df['Random Forest'], 
               0.4, label='Random Forest', color='lightgreen', edgecolor='darkgreen')

ax.set_xlabel('Error Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Error Comparison - Lower is Better', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(error_df['Error Type'])
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add values on bars with offset
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        # FIXED: Add offset to prevent text overlap
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/error_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ../results/error_comparison.png")
plt.show()


# 6. SAVE METRICS TO FILE



metrics = {
    'decision_tree': dt_metrics,
    'random_forest': rf_metrics,
    'confusion_matrices': {
        'decision_tree': cm_dt.tolist(),
        'random_forest': cm_rf.tolist()
    },
    'error_counts': {
        'decision_tree': {'fp': int(dt_fp), 'fn': int(dt_fn)},
        'random_forest': {'fp': int(rf_fp), 'fn': int(rf_fn)}
    }
}

with open('../data/model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# Save readable report
with open('../docs/model_comparison_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("IDS MODEL COMPARISON REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Test Set Information:\n")
    f.write(f"  Total Samples: {len(y_test):,}\n")
    f.write(f"  Normal Traffic: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)\n")
    f.write(f"  Attack Traffic: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)\n\n")
    
    f.write("DECISION TREE CLASSIFIER:\n")
    f.write("-" * 60 + "\n")
    for metric, value in dt_metrics.items():
        f.write(f"  {metric.capitalize():12}: {value:.6f} ({value*100:.4f}%)\n")
    f.write(f"\n  Error Analysis:\n")
    f.write(f"    False Positives: {dt_fp:,}\n")
    f.write(f"    False Negatives: {dt_fn:,}\n")
    f.write(f"    Total Errors:    {dt_fp + dt_fn:,}\n\n")
    
    f.write("RANDOM FOREST CLASSIFIER:\n")
    f.write("-" * 60 + "\n")
    for metric, value in rf_metrics.items():
        f.write(f"  {metric.capitalize():12}: {value:.6f} ({value*100:.4f}%)\n")
    f.write(f"\n  Error Analysis:\n")
    f.write(f"    False Positives: {rf_fp:,}\n")
    f.write(f"    False Negatives: {rf_fn:,}\n")
    f.write(f"    Total Errors:    {rf_fp + rf_fn:,}\n\n")
    
    f.write("=" * 60 + "\n")
    f.write("COMPARISON & RECOMMENDATION:\n")
    f.write("=" * 60 + "\n")
    
    # Determine best model
    if rf_metrics['f1'] > dt_metrics['f1']:
        winner = "Random Forest"
        f.write(f" WINNER: Random Forest\n\n")
        f.write(f"Advantages:\n")
        f.write(f"  ✓ Higher F1-Score ({rf_metrics['f1']:.6f} vs {dt_metrics['f1']:.6f})\n")
        f.write(f"  ✓ Better overall performance\n")
    elif dt_metrics['f1'] > rf_metrics['f1']:
        winner = "Decision Tree"
        f.write(f" WINNER: Decision Tree\n\n")
        f.write(f"Advantages:\n")
        f.write(f"  ✓ Higher F1-Score ({dt_metrics['f1']:.6f} vs {rf_metrics['f1']:.6f})\n")
        f.write(f"  ✓ Fewer errors ({dt_fp + dt_fn} vs {rf_fp + rf_fn})\n")
    else:
        f.write(f"ℹ Both models perform equally well\n")
        winner = "Both (Equal)"
    
    f.write(f"\nRECOMMENDATION: Deploy {winner} for production IDS\n")

print("✓ Metrics saved to: ../data/model_metrics.pkl")
print("✓ Report saved to: ../docs/model_comparison_report.txt")

print("\n" + "=" * 80)
print("✓ FINAL VISUALIZATIONS COMPLETED!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  1. ../results/model_comparison.png")
print(f"  2. ../results/error_comparison.png")
print(f"  3. ../data/model_metrics.pkl")
print(f"  4. ../docs/model_comparison_report.txt")