import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import os

def main():
    # Create directory for plots
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading dataset...")
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # 1. Plot Class Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='label', palette='viridis', hue='label', legend=False)
    plt.title('Class Distribution (Ham vs Spam)')
    plt.savefig(os.path.join(output_dir, '1_class_distribution.png'))
    plt.close()

    # Feature engineering for EDA: Message Length
    df['message_length'] = df['message'].apply(len)
    
    # 2. Plot Message Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, palette='viridis')
    plt.title('Message Length Distribution by Class')
    plt.xlim(0, 250) # Limit x-axis to zoom in on the bulk of data
    plt.savefig(os.path.join(output_dir, '2_message_length_distribution.png'))
    plt.close()

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna()

    print(f"Dataset loaded successfully with {len(df)} records.")

    # Split the dataset
    X = df['message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize models
    # Set probability=True for SVM to get prediction probabilities for ROC/PR curves
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel='linear', probability=True)
    }

    print("\nTraining models and evaluating accuracy...")
    print("-" * 40)
    
    best_model_name = ""
    best_accuracy = 0
    accuracies = {}
    
    roc_fig, roc_ax = plt.subplots(figsize=(10, 8))
    pr_fig, pr_ax = plt.subplots(figsize=(10, 8))

    plot_idx = 3 # Start index for confusion matrices
    
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        
        print(f"{name} Results:")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
        print("-" * 40)
        
        # 3, 4, 5. Confusion Matrix Plots
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{plot_idx}_confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()
        plot_idx += 1
        
        # ROC Curve Data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        # Precision-Recall Curve Data
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_ax.plot(recall, precision, label=name)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    # 6. ROC Curve Plot
    roc_ax.plot([0, 1], [0, 1], 'k--')
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.05])
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    roc_ax.legend(loc="lower right")
    roc_fig.tight_layout()
    roc_fig.savefig(os.path.join(output_dir, f'{plot_idx}_roc_curve.png'))
    plt.close(roc_fig)
    plot_idx += 1

    # 7. Precision-Recall Curve Plot
    pr_ax.set_xlim([0.0, 1.0])
    pr_ax.set_ylim([0.0, 1.05])
    pr_ax.set_xlabel('Recall')
    pr_ax.set_ylabel('Precision')
    pr_ax.set_title('Precision-Recall Curve')
    pr_ax.legend(loc="lower left")
    pr_fig.tight_layout()
    pr_fig.savefig(os.path.join(output_dir, f'{plot_idx}_precision_recall_curve.png'))
    plt.close(pr_fig)
    plot_idx += 1

    # 8. Model Accuracy Comparison
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('coolwarm', len(accuracies))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors, hue=list(accuracies.keys()), legend=False)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0.9, 1.0) # Zoom in on the top values
    plt.ylabel('Accuracy')
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{plot_idx}_model_accuracy_comparison.png'))
    plt.close()
            
    print(f"\nBest Model: {best_model_name} with an Accuracy of {best_accuracy:.4f}")
    if best_accuracy >= 0.70:
        print("\nTarget accuracy (0.70 to 0.85+) successfully met!")
    else:
        print("\nTarget accuracy was NOT met.")
        
    print(f"\nSuccessfully generated 8 plots and saved them in the '{os.path.abspath(output_dir)}' directory.")

if __name__ == "__main__":
    main()
