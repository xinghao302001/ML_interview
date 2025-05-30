import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成模拟数据
n_samples = 1000
n_features = 2
n_clusters = 4
random_state = 42
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# 使用KMeans聚类
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 绘制损失曲线（惯性指标）
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Loss)')
plt.title('K-means Loss Curve')
plt.xticks(k_values)
plt.grid(True)
plt.show()