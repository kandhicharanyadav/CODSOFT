import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Load Data ---
def load_data(file_path, is_test=False):
    print(f"Loading {file_path}...")
    try:
        if is_test:
            df = pd.read_csv(file_path, sep=':::', names=['ID', 'TITLE', 'DESCRIPTION'], engine='python')
        else:
            df = pd.read_csv(file_path, sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')
            df['GENRE'] = df['GENRE'].str.strip()
        
        df['TITLE'] = df['TITLE'].str.strip()
        df['DESCRIPTION'] = df['DESCRIPTION'].str.strip()
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

train_file = "c:/Users/chara/OneDrive/Desktop/Intenship/Codsoft/Task-1/Genre Classification Dataset/train_data.txt"
test_file = "c:/Users/chara/OneDrive/Desktop/Intenship/Codsoft/Task-1/Genre Classification Dataset/test_data.txt"
test_solution_file = "c:/Users/chara/OneDrive/Desktop/Intenship/Codsoft/Task-1/Genre Classification Dataset/test_data_solution.txt"

train_df = load_data(train_file)
# Load test features
test_features_df = load_data(test_file, is_test=True)
# Load test labels (solutions)
test_solution_df = load_data(test_solution_file)

if train_df is None or test_features_df is None or test_solution_df is None:
    print("Failed to load datasets. Exiting.")
    exit()

# The test_data_solution.txt contains ID, TITLE, GENRE, DESCRIPTION. We merge it with test_data to ensure order.
# Or realistically we can just use test_solution_df as our test set directly since it has the true labels and text.
test_df = test_solution_df.copy()

print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")


# --- 2. Data Preprocessing & Feature Extraction ---
print("\nExtracting TF-IDF features...")
# Using TfidfVectorizer with n-grams, sublinear tf to scale frequency, and max_features to capture more context
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=150000, ngram_range=(1, 2), sublinear_tf=True)

train_df['TEXT'] = train_df['TITLE'] + " " + train_df['DESCRIPTION']
test_df['TEXT'] = test_df['TITLE'] + " " + test_df['DESCRIPTION']

X_train = tfidf_vectorizer.fit_transform(train_df['TEXT'])
y_train = train_df['GENRE']

X_test = tfidf_vectorizer.transform(test_df['TEXT'])
y_test = test_df['GENRE']

print(f"Feature matrix shape: {X_train.shape}")


# --- 3. Modeling ---
print("\nTraining Logistic Regression model...")
# Removed class_weight='balanced' to improve overall accuracy and added C=5.0 for better fitting
model = LogisticRegression(C=5.0, max_iter=2000, n_jobs=-1, solver='lbfgs')
model.fit(X_train, y_train)


# --- 4. Evaluation ---
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)


# --- 5. Visualizations ---
print("\nGenerating visualisations...")
output_dir = "c:/Users/chara/OneDrive/Desktop/Intenship/Codsoft/Task-1/Genre Classification Dataset/results"
os.makedirs(output_dir, exist_ok=True)

# Save text metrics
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")

classes = model.classes_

# 5a. Confusion Matrix
plt.figure(figsize=(16, 12))
cm = confusion_matrix(y_test, y_pred, labels=classes)
sns.heatmap(cm, annot=False, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Movie Genre Classification')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()
print(f"Saved confusion matrix to {output_dir}/confusion_matrix.png")


# 5b. Multiclass ROC AUC Curves
print("Computing ROC AUC metrics...")
# Binarize labels for One-vs-Rest ROC curve calculations
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(classes)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Plot all ROC curves
plt.figure(figsize=(12, 10))

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# We have too many classes (27), plotting all lines will be messy. We'll plot top 10 most common or random few
top_genres = y_train.value_counts().head(10).index.tolist()

# Use a colormap to get colors
colormap = plt.get_cmap('tab10')
colors = [colormap(i) for i in np.linspace(0, 1, 10)]

# Plot ROC curves for the top 10 genres
color_idx = 0
for i in range(n_classes): 
    genre_name = classes[i]
    if genre_name in top_genres:
        color = colors[color_idx % len(colors)]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {genre_name} (area = {roc_auc[i]:0.2f})')
        color_idx += 1

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) to Multi-Class - Top 10 Genres')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_auc_curves.png'))
plt.close()
print(f"Saved ROC AUC curves to {output_dir}/roc_auc_curves.png")

roc_out = f"\nMicro-Averaged ROC AUC Score: {roc_auc['micro']:.4f}"
print(roc_out)

with open(os.path.join(output_dir, "classification_report.txt"), "a") as f:
    f.write(roc_out + "\n")

print("\nExecution Complete!")
