import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 加载 Iris 数据集
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练决策树模型
train_errors = []
test_errors = []

# 定义最大深度范围
max_depths = range(1, 21)

for depth in max_depths:
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_clf.fit(X_train, y_train)

    # 计算训练集和测试集错误率
    y_train_pred = tree_clf.predict(X_train)
    y_test_pred = tree_clf.predict(X_test)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)

# 绘制训练和测试集错误率曲线
plt.figure(figsize=(10, 6))
plt.plot(
    max_depths,
    train_errors,
    marker="o",
    linestyle="-",
    color="r",
    label="Training Error",
)
plt.plot(
    max_depths, test_errors, marker="o", linestyle="--", color="b", label="Test Error"
)
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Error Rate")
plt.title("Training and Test Error Rate vs. Max Depth for Decision Tree")
plt.legend()
plt.grid()
plt.show()

# 绘制决策树
plt.figure(figsize=(20, 10))
final_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
final_tree.fit(X_train, y_train)
plot_tree(
    final_tree,
    filled=True,
    feature_names=data.feature_names,
    class_names=data.target_names,
)
plt.title("Decision Tree Structure (Max Depth = 3)")
plt.show()
