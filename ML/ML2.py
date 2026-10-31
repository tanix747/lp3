import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
df.columns = ["label","message"]
df.label = df.label.map({"ham":0, "spam":1})

x = df.message
y = df.label

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel="linear")

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

print("KNN Accuracy:", accuracy_score(y_test, knn.predict(X_test)))
print("SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))