import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("sales_data_sample.csv", encoding="latin-1")

X = data.select_dtypes("number").dropna()
X = StandardScaler().fit_transform(X)

wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Choose K from the elbow graph
model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X)

print(clusters)