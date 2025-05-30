import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载 Iris 数据集
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 定义 k 值范围
k_values = range(1, 31)
train_errors = []
test_errors = []

# 训练 KNN 并计算每个 k 值对应的训练和测试集错误率
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)

# 绘制训练和测试集错误率曲线
plt.figure(figsize=(10, 6))
plt.plot(
    k_values, train_errors, marker="o", linestyle="-", color="r", label="Training Error"
)
plt.plot(
    k_values, test_errors, marker="o", linestyle="--", color="b", label="Test Error"
)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Error Rate")
plt.title("Training and Test Error Rate vs. K Value for KNN")
plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()
