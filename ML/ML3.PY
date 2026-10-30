import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
data = pd.read_csv("Churn_Modelling.csv")

# Select features
X = data[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
          "HasCrCard", "IsActiveMember", "EstimatedSalary"]]
y = data["Exited"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train neural network
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))