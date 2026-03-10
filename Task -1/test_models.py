import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score
import time

def load_data(file_path):
    df = pd.read_csv(file_path, sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')
    df['GENRE'] = df['GENRE'].str.strip()
    return df

print("Loading data...")
train_df = load_data("c:/Users/chara/OneDrive/Desktop/Intenship/Codsoft/Genre Classification Dataset/train_data.txt")
test_df = load_data("c:/Users/chara/OneDrive/Desktop/Intenship/Codsoft/Genre Classification Dataset/test_data_solution.txt")

train_df['TEXT'] = train_df['TITLE'] + " " + train_df['DESCRIPTION']
test_df['TEXT'] = test_df['TITLE'] + " " + test_df['DESCRIPTION']

print("Vectorizing...")
tfidf = TfidfVectorizer(stop_words='english', max_features=150000, ngram_range=(1, 2), min_df=2)
X_train = tfidf.fit_transform(train_df['TEXT'])
X_test = tfidf.transform(test_df['TEXT'])
y_train = train_df['GENRE']
y_test = test_df['GENRE']

models = {
    "LogisticRegression(C=10)": LogisticRegression(C=10.0, max_iter=2000, n_jobs=-1, solver='lbfgs'),
    "LinearSVC(C=1.0)": LinearSVC(C=1.0, max_iter=2000),
    "LinearSVC(C=0.5)": LinearSVC(C=0.5, max_iter=2000),
    "ComplementNB(alpha=0.1)": ComplementNB(alpha=0.1),
    "ComplementNB(alpha=1.0)": ComplementNB(alpha=1.0),
    "MultinomialNB(alpha=0.1)": MultinomialNB(alpha=0.1)
}

print("Testing models...")
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    t1 = time.time()
    print(f"{name}: Accuracy = {acc:.4f} (Time: {t1-t0:.2f}s)")
