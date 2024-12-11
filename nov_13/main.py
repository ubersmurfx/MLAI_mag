import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


X = np.random.rand(100, 2)

k = 3

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Кластеризация k-средних')
plt.show()

# Вывод метрики качества
print('SSE:', kmeans.inertia_)