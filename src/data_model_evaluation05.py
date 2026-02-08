"""
Step 5:  Data Evaluation  
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

print("=" * 80)
print("STEP 5: MODEL EVALUATION")
print("=" * 80)

# Load data
df = pd.read_csv('../data/dataset_clean.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Load saved models and files
with open('../data/selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)
print(f"✓ Selected features: {len(selected_features)}")

with open('../data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../data/model_decision_tree.pkl', 'rb') as f:
    dt = pickle.load(f)

with open('../data/model_random_forest.pkl', 'rb') as f:
    rf = pickle.load(f)

# 1. Prepare Data
X = df[selected_features]
y = df['is_attack']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # ✅ Ajout stratify
)

X_test_scaled = scaler.transform(X_test)

# 2. Evaluate Decision Tree
print("\n" + "=" * 80)
print("Evaluating Decision Tree Classifier...")
print("=" * 80)

y_pred_dt = dt.predict(X_test_scaled)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Create figure with percentages
fig, ax = plt.subplots(figsize=(8, 6))
disp_dt = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt, 
    display_labels=['Normal', 'Attack']
)
disp_dt.plot(cmap='Blues', values_format='d', ax=ax)

# Add percentages
for i in range(2):
    for j in range(2):
        pct = cm_dt[i, j] / cm_dt[i].sum() * 100
        ax.text(j, i + 0.3, f'({pct:.1f}%)', 
                ha='center', va='center', 
                fontsize=10, color='darkred', fontweight='bold')

plt.title('Confusion Matrix - Decision Tree', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/confusion_matrix_dt.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ../results/confusion_matrix_dt.png")
plt.show()

# 3. Evaluate Random Forest
print("\n" + "=" * 80)
print("Evaluating Random Forest Classifier...")
print("=" * 80)

y_pred_rf = rf.predict(X_test_scaled)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Create figure with percentages
fig, ax = plt.subplots(figsize=(8, 6))
disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=['Normal', 'Attack']
)
disp_rf.plot(cmap='Greens', values_format='d', ax=ax)

# Add percentages
for i in range(2):
    for j in range(2):
        pct = cm_rf[i, j] / cm_rf[i].sum() * 100
        ax.text(j, i + 0.3, f'({pct:.1f}%)', 
                ha='center', va='center', 
                fontsize=10, color='darkgreen', fontweight='bold')

plt.title('Confusion Matrix - Random Forest', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ../results/confusion_matrix_rf.png")
plt.show()

# 4. Classification Reports
print("\n" + "=" * 80)
print("CLASSIFICATION REPORTS")
print("=" * 80)

print("\n[1] Decision Tree:")
print("-" * 80)
print(classification_report(y_test, y_pred_dt, target_names=['Normal', 'Attack']))

print("\n[2] Random Forest:")
print("-" * 80)
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Attack']))

# 5. Save detailed report
print("\n" + "=" * 80)
print("Saving evaluation report...")

with open('../docs/evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write("EVALUATION REPORT - IDS ML\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DECISION TREE:\n")
    f.write("-" * 80 + "\n")
    f.write(classification_report(y_test, y_pred_dt, target_names=['Normal', 'Attack']))
    f.write("\n\n")
    
    f.write("RANDOM FOREST:\n")
    f.write("-" * 80 + "\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Attack']))
    f.write("\n\n")
    
    # ✅ Ajout des matrices de confusion
    f.write("CONFUSION MATRICES:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Decision Tree:\n{cm_dt}\n\n")
    f.write(f"Random Forest:\n{cm_rf}\n")

print("✓ Report saved: ../docs/evaluation_report.txt")

print("\n" + "=" * 80)
print("✓ EVALUATION COMPLETED!")
print("=" * 80)