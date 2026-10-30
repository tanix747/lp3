import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("sales_data_sample.csv")

# Select numerical columns
X = data.select_dtypes(include="number").dropna()

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Elbow method
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Apply K-Means
model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X)

print("Clusters:")
print(clusters)