import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

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

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=random_state, n_iter=1000, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)

# 绘制t-SNE结果
plt.figure(figsize=(8, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c="blue", alpha=0.5)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization")
plt.grid(True)
plt.show()

# 绘制损失曲线（KL散度）
loss = tsne.kl_divergence_
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(tsne.embedding_) + 1), [loss] * len(tsne.embedding_), marker="o")
plt.xlabel("Iteration")
plt.ylabel("KL Divergence (Loss)")
plt.title("t-SNE Loss Curve (KL Divergence)")
plt.grid(True)
plt.show()
