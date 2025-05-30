import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 生成模拟数据
n_samples = 1000
n_features = 10
n_clusters = 4
random_state = 42
X, _ = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=random_state,
)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA进行降维
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 计算不同主成分数量下的累积方差解释率
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 绘制累积方差解释率曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Loss Curve (Cumulative Explained Variance)")
plt.grid(True)
plt.show()
