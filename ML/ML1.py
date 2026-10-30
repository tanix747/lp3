"""
setup

sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
mkdir ~/ML
cd ~/ML
python3.12 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow nltk

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("uber.csv")

print("Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")

# Remove missing values
df = df.dropna()

# Convert datetime column
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# Extract useful features
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["hour"] = df["pickup_datetime"].dt.hour

# Drop original datetime
df = df.drop(columns=["pickup_datetime"])

print("\nAfter Preprocessing:")
print(df.head())


# --------------------------------------------------
# 3. Identify Outliers
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare_amount"])
plt.title("Fare Amount - Outliers")
plt.show()

# Remove invalid/outlier fare values
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["fare_amount"] >= lower) &
        (df["fare_amount"] <= upper)]

print("\nDataset after removing outliers:")
print(df.shape)


# --------------------------------------------------
# 4. Correlation
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()


# --------------------------------------------------
# 5. Features and Target
# --------------------------------------------------

X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 6. Linear Regression
# --------------------------------------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))


# --------------------------------------------------
# 7. Random Forest Regression
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))


# --------------------------------------------------
# 8. Compare Models
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "R2 Score": [
        linear_r2,
        rf_r2
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ]
})

print("\nModel Comparison:")
print(results)