import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n_legit = 9500
n_fraud = 500
n_total = n_legit + n_fraud

# Legitimate transactions
legit = pd.DataFrame({
    'amount':          np.random.lognormal(mean=4.5, sigma=1.2, size=n_legit),
    'hour':            np.random.randint(0, 24, n_legit),
    'v1':              np.random.normal(0.1,  1.0, n_legit),
    'v2':              np.random.normal(0.2,  1.0, n_legit),
    'v3':              np.random.normal(-0.1, 1.0, n_legit),
    'v4':              np.random.normal(0.05, 0.8, n_legit),
    'v5':              np.random.normal(0.0,  1.0, n_legit),
    'v6':              np.random.normal(0.1,  0.9, n_legit),
    'v7':              np.random.normal(-0.05,1.0, n_legit),
    'v8':              np.random.normal(0.0,  1.0, n_legit),
    'merchant_risk':   np.random.beta(2, 8, n_legit),
    'distance_home':   np.abs(np.random.normal(10, 15, n_legit)),
    'Class':           0
})

# Fraudulent transactions — moderate overlap for realistic ~98% accuracy
fraud = pd.DataFrame({
    'amount':          np.random.lognormal(mean=5.0, sigma=1.8, size=n_fraud),
    'hour':            np.random.choice([0,1,2,3,22,23], n_fraud),
    'v1':              np.random.normal(-2.0, 2.0, n_fraud),
    'v2':              np.random.normal(1.8,  2.0, n_fraud),
    'v3':              np.random.normal(-2.5, 2.0, n_fraud),
    'v4':              np.random.normal(2.2,  1.8, n_fraud),
    'v5':              np.random.normal(-1.5, 2.0, n_fraud),
    'v6':              np.random.normal(1.5,  2.0, n_fraud),
    'v7':              np.random.normal(-2.0, 2.0, n_fraud),
    'v8':              np.random.normal(1.8,  2.0, n_fraud),
    'merchant_risk':   np.random.beta(5, 3, n_fraud),
    'distance_home':   np.abs(np.random.normal(50, 40, n_fraud)),
    'Class':           1
})

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print("=" * 60)
print("  CREDIT CARD FRAUD DETECTION")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Legitimate:   {(df.Class == 0).sum()} ({(df.Class==0).mean()*100:.1f}%)")
print(f"Fraudulent:   {(df.Class == 1).sum()} ({(df.Class==1).mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────
features = [c for c in df.columns if c != 'Class']
X = df[features]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Manual oversampling of minority class (SMOTE-like)
train_df = pd.concat([X_train, y_train], axis=1)
majority = train_df[train_df.Class == 0]
minority = train_df[train_df.Class == 1]
minority_up = resample(minority, replace=True,
                        n_samples=len(majority), random_state=42)
train_balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
X_train_bal = train_balanced[features]
y_train_bal  = train_balanced['Class']

scaler = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_bal)
X_test_sc   = scaler.transform(X_test)
X_train_raw = X_train_bal.values
X_test_raw  = X_test.values

# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=0.3, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=6,
                                                   min_samples_leaf=15, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == 'Logistic Regression' else X_train_raw
    X_te = X_test_sc  if name == 'Logistic Regression' else X_test_raw
    model.fit(X_tr, y_train_bal)
    y_pred  = model.predict(X_te)
    y_prob  = model.predict_proba(X_te)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
                     'acc': acc, 'f1': f1, 'auc': auc,
                     'X_te': X_te, 'cm': confusion_matrix(y_test, y_pred)}
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

best_name = max(results, key=lambda k: results[k]['auc'])
print(f"\n[Best Model]: {best_name} (AUC={results[best_name]['auc']:.4f})")

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS  (4 figures, 10 plots)
# ─────────────────────────────────────────────
palette = {'Legit': '#2ecc71', 'Fraud': '#e74c3c'}
model_colors = ['#3498db', '#e67e22', '#9b59b6']
model_names  = list(results.keys())

# ── Figure 1: EDA ──────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold', y=1.01)

# 1a – Class distribution
ax = axes[0, 0]
counts = df['Class'].value_counts()
bars = ax.bar(['Legitimate', 'Fraudulent'], counts.values,
              color=[palette['Legit'], palette['Fraud']], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count'); ax.set_ylim(0, counts.max() * 1.2)

# 1b – Transaction amount distribution
ax = axes[0, 1]
for cls, label, color in [(0,'Legitimate', palette['Legit']), (1,'Fraudulent', palette['Fraud'])]:
    vals = df[df.Class==cls]['amount'].clip(upper=1000)
    ax.hist(vals, bins=40, alpha=0.65, label=label, color=color, edgecolor='white')
ax.set_title('Transaction Amount Distribution', fontweight='bold')
ax.set_xlabel('Amount ($)'); ax.set_ylabel('Frequency')
ax.legend()

# 1c – Hourly fraud rate
ax = axes[0, 2]
hourly = df.groupby('hour')['Class'].agg(['sum','count'])
hourly['rate'] = hourly['sum'] / hourly['count'] * 100
ax.bar(hourly.index, hourly['rate'], color='#e74c3c', alpha=0.8)
ax.set_title('Fraud Rate by Hour of Day', fontweight='bold')
ax.set_xlabel('Hour'); ax.set_ylabel('Fraud Rate (%)')

# 1d – Distance from home
ax = axes[1, 0]
for cls, label, color in [(0,'Legitimate', palette['Legit']), (1,'Fraudulent', palette['Fraud'])]:
    ax.hist(df[df.Class==cls]['distance_home'].clip(upper=200), bins=35,
            alpha=0.65, label=label, color=color, edgecolor='white')
ax.set_title('Distance from Home', fontweight='bold')
ax.set_xlabel('Distance (km)'); ax.set_ylabel('Frequency')
ax.legend()

# 1e – Merchant risk
ax = axes[1, 1]
for cls, label, color in [(0,'Legitimate', palette['Legit']), (1,'Fraudulent', palette['Fraud'])]:
    ax.hist(df[df.Class==cls]['merchant_risk'], bins=30,
            alpha=0.65, label=label, color=color, edgecolor='white')
ax.set_title('Merchant Risk Score', fontweight='bold')
ax.set_xlabel('Risk Score'); ax.set_ylabel('Frequency')
ax.legend()

# 1f – Correlation heatmap
ax = axes[1, 2]
corr = df[features + ['Class']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdBu_r', center=0,
            square=True, linewidths=0.3, cbar_kws={'shrink': 0.7},
            annot=False, fmt='.1f')
ax.set_title('Feature Correlation', fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Done] Figure 1 saved: EDA")

# ── Figure 2: Confusion Matrices ───────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
for ax, name, color in zip(axes, model_names, model_colors):
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'],
                linewidths=1, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    acc = results[name]['acc']
    ax.set_title(f'{name}\nAccuracy: {acc:.4f}', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('fig2_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Done] Figure 2 saved: Confusion Matrices")

# ── Figure 3: ROC + PR Curves ──────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('Model Performance Curves', fontsize=16, fontweight='bold')

# ROC
ax = axes[0]
ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random (AUC=0.50)')
for name, color in zip(model_names, model_colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['auc']
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC={auc:.3f})')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.fill_between([0,1],[0,1], alpha=0.05, color='grey')

# Precision-Recall
ax = axes[1]
baseline = y_test.mean()
ax.axhline(baseline, color='k', linestyle='--', alpha=0.4,
           label=f'Baseline (P={baseline:.3f})')
for name, color in zip(model_names, model_colors):
    prec, rec, _ = precision_recall_curve(y_test, results[name]['y_prob'])
    ax.plot(rec, prec, color=color, lw=2.5, label=name)
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curve', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

plt.tight_layout()
plt.savefig('fig3_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Done] Figure 3 saved: ROC + PR Curves")

# ── Figure 4: Model Comparison + Feature Importance ──
fig4, axes = plt.subplots(1, 2, figsize=(15, 6))
fig4.suptitle('Model Comparison & Feature Importance', fontsize=16, fontweight='bold')

# 4a – Metric comparison bar chart
ax = axes[0]
metrics = ['acc', 'f1', 'auc']
metric_labels = ['Accuracy', 'F1 Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25
for i, (name, color) in enumerate(zip(model_names, model_colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel('Score'); ax.set_ylim([0, 1.12])
ax.set_title('Metric Comparison Across Models', fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0.85, color='red', linestyle=':', alpha=0.5, label='Target 0.85')

# 4b – Random Forest feature importance
ax = axes[1]
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'feature': features, 'importance': importances})
feat_df = feat_df.sort_values('importance', ascending=True).tail(10)
colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))
bars = ax.barh(feat_df['feature'], feat_df['importance'],
               color=colors_fi, edgecolor='white')
for bar, val in zip(bars, feat_df['importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_title('Random Forest - Feature Importance', fontweight='bold')
ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('fig4_comparison_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Done] Figure 4 saved: Comparison + Feature Importance")

# ─────────────────────────────────────────────
# 5. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'ROC-AUC':>10}")
print("-" * 60)
for name in model_names:
    r = results[name]
    print(f"{name:<25} {r['acc']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}")
print("=" * 60)
print(f"\n[Best Model] : {best_name}")
print(f"   Accuracy   : {results[best_name]['acc']:.4f}")
print(f"   F1 Score   : {results[best_name]['f1']:.4f}")
print(f"   ROC-AUC    : {results[best_name]['auc']:.4f}")
print("\nFull Classification Report - Best Model:")
print(classification_report(y_test, results[best_name]['y_pred'],
                             target_names=['Legitimate', 'Fraudulent']))
print("\n[Done] All 4 figures saved to the current directory.")