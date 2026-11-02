# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# from sklearn.neural_network import MLPRegressor

# 定义激活函数，避免溢出
def sigmoid(x):
    # ndarray展平
    ravel = x.ravel()
    length = len(ravel)

    # copy
    y = []

    # 防止 overflow encountered exp
    for idx in range(length):

        if ravel[idx] >= 0:
            y.append(1.0 / (1 + np.exp(-ravel[idx])))

        else:
            y.append(np.exp(ravel[idx]) / (1 + np.exp(ravel[idx])))

    return np.array(y).reshape(x.shape)

# 1.初始化参数
"""
两层神经网络，第一层神经元中:w1为10*4矩阵，第二层神经元中:w2为3*10矩阵
X先与w1矩阵相乘，再与w2矩阵相乘，输出3*150的分类标签矩阵
"""
def initialize(n_x, n_h, n_y):
    np.random.seed(2)

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    init_paras = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return init_paras

initialize(4, 10, 3)

# 2.前向传播（两个输出）
def forward(init_paras, X):

    w1 = init_paras['w1']
    b1 = init_paras['b1']
    w2 = init_paras['w2']
    b2 = init_paras['b2']

    # 第一层
    X_1 = np.dot(w1, X) + b1  # 10 * 150
    Z_1 = sigmoid(X_1)

    # 第二层
    X_2 = np.dot(w2, Z_1) + b2  # 3 * 150
    Z_2 = sigmoid(X_2)

    return Z_2, Z_1

# 3.计算代价函数
def compute_cost(a2, Y):

    # 交叉熵
    m = Y.shape[1]
    logprobs = np.multiply(np.log(a2), Y) + np.multiply(np.log(1 - a2), (1 - Y))
    cost = -np.sum(logprobs) / m

    return cost

# 4.反向传播
def backward(init_paras, forward_paras, X, Y):

    # 前向传播的第一层输出、第二层输出
    a2, Z1 = forward(init_paras, X)
    m = X.shape[1]

    # 最后输出的误差
    error = a2 - Y
    dz2 = error

    # 最后一层 dw2 db2
    dw2 = np.dot(dz2, Z1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    # 向前一层的误差
    dz1 = np.dot(init_paras['w2'].T, dz2) * (Z1 * (1 - Z1))

    # 向前一层 dw1 db1
    dw1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    backward_paras = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return backward_paras

# 5.更新参数
def update(init_paras, backward_paras, lr=0.1):

    w1 = init_paras['w1']
    b1 = init_paras['b1']
    w2 = init_paras['w2']
    b2 = init_paras['b2']

    dw1 = backward_paras['dw1']
    db1 = backward_paras['db1']
    dw2 = backward_paras['dw2']
    db2 = backward_paras['db2']

    # 更新参数
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    update_paras = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return update_paras


# 6.建立神经网络
def nn_model(X, Y, n_h, n_input, n_output, num_iterations=2000, print_cost=False):
    np.random.seed(3)

    n_x = n_input  # 输入层节点数
    n_y = n_output  # 输出层节点数

    # 1.初始化参数
    parameters = initialize(n_x, n_h, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 2.前向传播
        a2, forward_paras = forward(parameters, X)
        # 3.计算代价函数
        cost = compute_cost(a2, Y)
        # 4.反向传播
        backward_paras = backward(parameters, forward_paras, X, Y)
        # 5.更新参数
        parameters = update(parameters, backward_paras)

        # 每100次迭代，输出一次代价函数 （请补全）
        if print_cost and i % 100 == 0:
            print(f'迭代第{i}次: {cost}')

    return parameters


if __name__ == "__main__":
    # 载入数据集
    data_set = pd.read_csv('iris_BP.csv', header=None)
    X = data_set.iloc[:, 0:4].values.T  # 前四列是特征，T表示转置
    Y = data_set.iloc[:, 4:].values.T  # 分类标签

    # 计算参数
    parameters = nn_model(X, Y, 10, 4, 3, print_cost=True)
    print(parameters)  # 输出权重和偏置的值
