import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.linear_model import Perceptron

"""感知机模型"""
# 数据线性可分，二分类数据

class Model:
    def __init__(self):

        # 初始w/b的值
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0.0
        self.l_rate = 0.1
      
    # 求y的值
    def sign(self, x, w, b):
        y = np.dot(x, w) + b  
        return y

    
    # 梯度下降法（GD），根据损失函数的梯度，对w，b进行更新（请补全）
    def fit(self, X_train, y_train):

        x = [self.sign(i, self.w, self.b) for i in X_train]
        x = np.asarray(x, dtype=np.float32)

        for j in range(len(x)):

            if x[j] * y_train[j] < 0:
                self.w = self.w + X_train[j] * y_train[j] * self.l_rate
                self.b = self.b + self.l_rate * y_train[j]
                break

        return 0

    # 检查是否有误分类点
    def check(self, X, y):
        x = [self.sign(i, self.w, self.b) for i in X]
        x = np.asarray(x)

        flag = 0
        for j in range(len(x)):
            if x[j] * y[j] < 0:
                flag = 1
                break

        return flag

    # 训练
    def train(self, X_train, y_train):

        while self.check(X_train, y_train) == 1:

            self.fit(X_train, y_train)

        return 0

    # 得分
    def score(self):
        pass


# 导入数据集
df = pd.read_csv('Iris.csv', usecols=[1, 2, 3, 4, 5])

setosa = df[df['Species'] == 'Iris-setosa']
versicolor = df[df['Species'] == 'Iris-versicolor']
virginica = df[df['Species'] == 'Iris-virginica']

"""绘制训练集散点图，观察数据集的线性可分性"""
# 绘制图形的画板尺寸为8*5
plt.figure(figsize=(8, 5))

# 散点图的x坐标、y坐标、标签（请补全三种标签散点分布图可视化）
plt.scatter(setosa['SepalLength'], setosa['SepalWidth'], label='Iris-setosa')
plt.scatter(versicolor['SepalLength'], versicolor['SepalWidth'], label='Iris-versicolor')
plt.scatter(virginica['SepalLength'], virginica['SepalWidth'], label='Iris-virginica')

plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
#  '鸢尾花萼片的长度与宽度的散点分布'
plt.title('Scattered distribution of length and width of iris sepals.')
# 显示标签
plt.legend()
plt.show()
"""
由散点图发现，Iris-setosa与Iris-versicolor和Iris-virginica的混合体线性可分
"""

# 取前100条数据中的：前2个特征+标签用于训练
data = np.array(df.iloc[:100, [0, 1, -1]])
# 数据类型转换，为了后面的数学计算
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 'Iris-setosa' else -1 for i in y])

"""感知机模型,开始训练"""
perceptron = Model()
perceptron.train(X, y)

# 最终参数
print(f'w={perceptron.w}')
print(f'b={perceptron.b}')

# 绘图
x_points = np.linspace(4, 7, 10)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)
plt.scatter(df[:50]['SepalLength'], df[:50]['SepalWidth'], label='Iris-setosa')
plt.scatter(df[50:100]['SepalLength'], df[50:100]['SepalWidth'], label='Iris-versicolor')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')

# '感知机模型训练结果'
plt.title('Training results of Perceptron.')
plt.legend()
plt.show()
