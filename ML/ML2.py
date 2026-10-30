import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove missing values
df = df.dropna()

print(df.head())
print("\nClass Distribution:")
print(df["label"].value_counts())


# --------------------------------------------------
# 2. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# --------------------------------------------------
# 3. TF-IDF Feature Extraction
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# 4. K-Nearest Neighbors
# --------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_tfidf, y_train)

knn_pred = knn.predict(X_test_tfidf)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", knn_accuracy)
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))

print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))


# --------------------------------------------------
# 5. Support Vector Machine
# --------------------------------------------------

svm = SVC(kernel="linear")

svm.fit(X_train_tfidf, y_train)

svm_pred = svm.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Accuracy:", svm_accuracy)
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))

print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))


# --------------------------------------------------
# 6. Performance Comparison
# --------------------------------------------------

print("\n-------------------------------")
print("Performance Comparison")
print("-------------------------------")

print(f"KNN Accuracy : {knn_accuracy:.4f}")
print(f"SVM Accuracy : {svm_accuracy:.4f}")