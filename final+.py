import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle, class_weight
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load dan preprocessing
df = pd.read_csv("PRDECT-ID Dataset.csv")
print(df['Emotion'].value_counts())

factory = StopWordRemoverFactory()
stopwords_id = set(factory.get_stop_words())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_id]
    return ' '.join(tokens)

df['Customer Review'] = df['Customer Review'].apply(clean_text)
df = shuffle(df, random_state=42).reset_index(drop=True)

X = df['Customer Review']
y = df['Emotion']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_val_vectorized = vectorizer.fit_transform(X_train_val)
X_test_vectorized = vectorizer.transform(X_test)

os.makedirs('saved_models', exist_ok=True)
with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
print("Vectorizer saved as 'vectorizer.pkl'")

class CustomSVM:
    def __init__(self, lr=0.005, lambda_param=0.01, n_iters=3000, class_weights=None):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.class_weights = class_weights
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in tqdm(range(self.n_iters), desc="Training SVM", leave=False):
            for idx, x_i in enumerate(X):
                x_i_array = x_i.toarray().flatten()
                condition = y_[idx] * (np.dot(x_i_array, self.w) - self.b) >= 1

                weight = 1.0
                if self.class_weights is not None:
                    label = 1 if y[idx] == 1 else 0
                    weight = self.class_weights.get(label, 1.0)

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * weight * (2 * self.lambda_param * self.w - np.dot(x_i_array, y_[idx]))
                    self.b -= self.lr * weight * y_[idx]

    def decision_function(self, X):
        return X.dot(self.w) - self.b

def predict(X, models):
    preds = []
    for i in range(X.shape[0]):
        scores = {label: model.decision_function(X[i]) for label, model in models.items()}
        preds.append(max(scores, key=scores.get))
    return np.array(preds)

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump({
            'w': model.w,
            'b': model.b,
            'lr': model.lr,
            'lambda_param': model.lambda_param,
            'n_iters': model.n_iters
        }, f)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_y_true = []
all_y_pred = []
best_models_per_label = {}
best_params_per_label = {}

for fold, (train_index, val_index) in enumerate(kf.split(X_train_val_vectorized)):
    print(f"\n==== Fold {fold + 1} ====")
    X_train, X_val = X_train_val_vectorized[train_index], X_train_val_vectorized[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    unique_labels = np.unique(y_train)
    
    for label in unique_labels:
        if label not in best_models_per_label:
            best_model = None
            best_score = -np.inf
            best_params = None
            param_grid = [
                {'lr': 0.001, 'lambda_param': 0.01, 'n_iters': 10},
                {'lr': 0.005, 'lambda_param': 0.01, 'n_iters': 10},
                {'lr': 0.01, 'lambda_param': 0.001, 'n_iters': 10}
            ]
            y_binary = np.where(y_train == label, 1, 0)
            weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_binary)
            class_weights_dict = {0: weights[0], 1: weights[1]}

            for params in param_grid:
                clf = CustomSVM(lr=params['lr'], lambda_param=params['lambda_param'], n_iters=params['n_iters'], class_weights=class_weights_dict)
                clf.fit(X_train, y_binary)
                val_preds = np.where(clf.decision_function(X_val) >= 0, 1, 0)
                acc = np.mean(val_preds == np.where(y_val == label, 1, 0))
                if acc > best_score:
                    best_score = acc
                    best_model = clf
                    best_params = params

            best_models_per_label[label] = best_model
            best_params_per_label[label] = best_params

    y_val_pred = predict(X_val, best_models_per_label)
    all_y_true.extend(y_val)
    all_y_pred.extend(y_val_pred)

print("\n=== Final Classification Report from K-Fold (Train/Validation Data) ===")
print(classification_report(all_y_true, all_y_pred))

cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(y_train_val))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (K-Fold Train/Validation)')
plt.show()

# Simpan semua model terbaik hasil tuning K-Fold
os.makedirs('saved_models', exist_ok=True)
for label, model in best_models_per_label.items():
    save_model(model, f'saved_models/svm_model_label_{label}.pkl')

print("\n=== All best models (from K-Fold) saved in 'saved_models/' ===")

# print("\n==== FINAL EVALUATION ON TEST SET ====")
y_test_pred = predict(X_test_vectorized, best_models_per_label)

print("\n=== Final Test Classification Report ===")
print(classification_report(y_test, y_test_pred))

cm_test = confusion_matrix(y_test, y_test_pred, labels=np.unique(y))
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Final Test Set)')
plt.show()