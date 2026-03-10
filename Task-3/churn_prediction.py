import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import os

# Create folder for graphs
os.makedirs('graphs', exist_ok=True)

print("Loading data...")
df = pd.read_csv('Churn_Modelling.csv')

# Drop irrelevant columns
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']

categorical_cols = ['Geography', 'Gender']
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
y_preds = {}
y_probs = {}

print("Training models...")
for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    
    y_preds[name] = pipeline.predict(X_test)
    y_probs[name] = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_preds[name])
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

print("Generating graphs...")
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Model Accuracy Comparison
plt.figure(figsize=(10, 6))
bars = sns.barplot(x=list(results.keys()), y=list(results.values()), hue=list(results.keys()), legend=False, palette='viridis')
plt.title('1. Model Accuracy Comparison', fontsize=14)
plt.ylabel('Accuracy Score', fontsize=12)
plt.ylim(0.7, 0.95)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/1_accuracy_comparison.png')
plt.close()

# 2. ROC Curves
plt.figure(figsize=(10, 8))
for name, y_prob in y_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('2. Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig('graphs/2_roc_curves.png')
plt.close()

# 3. Precision-Recall Curves
plt.figure(figsize=(10, 8))
for name, y_prob in y_probs.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'{name}', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('3. Precision-Recall Curve', fontsize=14)
plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()
plt.savefig('graphs/3_precision_recall_curves.png')
plt.close()

# Confusion Matrices
cmap_dict = {'Logistic Regression': 'Blues', 'Random Forest': 'Greens', 'Gradient Boosting': 'Oranges'}
for i, (name, y_pred) in enumerate(y_preds.items(), 4):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_dict[name], annot_kws={'size': 14})
    plt.title(f'{i}. Confusion Matrix - {name}', fontsize=14)
    plt.ylabel('Actual label (0 = Retained, 1 = Churned)', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'graphs/{i}_confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

print("All 6 graphs successfully generated and saved in the 'graphs' folder.")
