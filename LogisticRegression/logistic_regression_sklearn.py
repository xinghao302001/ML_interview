import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# 加载 Iris 数据集
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练逻辑回归模型
log_reg = LogisticRegression(max_iter=100, solver="lbfgs", multi_class="multinomial")
losses = []

# 每次迭代后记录损失值
for i in range(1, 101):
    log_reg.max_iter = i
    log_reg.fit(X_train, y_train)
    y_prob = log_reg.predict_proba(X_train)
    loss = log_loss(y_train, y_prob)
    losses.append(loss)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, 201), losses, marker="o", linestyle="-", color="g")
plt.xlabel("Number of Iterations")
plt.ylabel("Log Loss")
plt.title("Log Loss vs. Number of Iterations for Logistic Regression")
plt.grid()
plt.show()
