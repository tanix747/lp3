"""
setup

sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
mkdir ~/ML
cd ~/ML
python3.12 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn 

"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


# 1. Load data
df = pd.read_csv("uber.csv")

# 2. Remove unnecessary columns and missing values
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")
df = df.dropna()


# 3. Convert date into useful features
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["hour"] = df["pickup_datetime"].dt.hour

df = df.drop(columns=["pickup_datetime"])


# 4. Identify and remove outliers using IQR
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df["fare_amount"] >= Q1 - 1.5 * IQR) &
    (df["fare_amount"] <= Q3 + 1.5 * IQR)
]


# 5. Check correlation
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()


# 6. Separate input and output
X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]


# 7. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print (r2_score(y_test, pred))
print (root_mean_squared_error(y_test, pred))

rf=RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print (r2_score(y_test, pred))
print (root_mean_squared_error(y_test, pred))