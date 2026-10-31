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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("uber.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore").dropna()

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["hour"] = df["pickup_datetime"].dt.hour
df = df.drop(columns=["pickup_datetime"])

sns.boxplot(x=df["fare_amount"])
plt.show()

Q1 = df.fare_amount.quantile(.25)
Q3 = df.fare_amount.quantile(.75)
IQR = Q3 - Q1

df = df[(df.fare_amount >= Q1 - 1.5*IQR) &
        (df.fare_amount <= Q3 + 1.5*IQR)]

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

p1 = lr.predict(X_test)
p2 = rf.predict(X_test)

print("Linear Regression:", r2_score(y_test,p1),
      np.sqrt(mean_squared_error(y_test,p1)))

print("Random Forest:", r2_score(y_test,p2),
      np.sqrt(mean_squared_error(y_test,p2)))