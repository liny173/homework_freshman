import numpy as np
import pandas as pd

# 创建一个类
class Logistic_model():

    # 初始化
    def __init__(self, data, learning_rate):

        self.data = data
        self.learning_rate = learning_rate

        # 两个参数
        self.w = np.zeros(30)
        self.b = 0

    # 阶跃函数
    def sig_f(self, x):

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

    # 预测标签
    def label_f(self, x):

        # 标签列表
        # 以0.5为分界
        y_pre_label = [1 if i >= 0.5 else 0 for i in x]

        return y_pre_label

    # 损失函数
    def cost(self, y, y_pre):

        # 交叉熵
        c = np.array(y * self.sig_f(y_pre) + (1 - y) * (1 - self.sig_f(y_pre)))
        c_mean = -(np.mean(c))

        return c_mean

    # 梯度下降函数
    def gradient_descent(self, y):

        # 迭代计算
        for _ in range(2200):

            # l = w*x + b
            l = np.dot(self.data, self.w) + self.b
            # 代入sig函数中，求得概率
            y_pre = self.sig_f(l)

            # 对w, b求偏导
            dw = (1 / 30) * np.dot(self.data.T, (y_pre - y))
            db = (1 / 30) * np.sum(y_pre - y)

            # 向梯度相反方向移动
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if abs(self.cost(y, y_pre)) < 1e-3:
                break

        return 0

    # 预测函数
    def predict(self, x):

        # 代入计算求得的w, b
        l = np.dot(x, self.w) + self.b
        y_pre = self.sig_f(l)
        y_pre_label = self.label_f(y_pre)

        return np.asarray(y_pre), np.asarray(y_pre_label)

    # 准确率
    def accuracy(self, y, y_pre):

        accuracy = np.sum(y == y_pre) / len(y)

        return accuracy

    # 查准率
    def precision(self, y, y_pre):

        # 预测为真的数量
        pre_num = np.sum(y_pre == 1)

        # 预测值中实际为真的数量
        lab = [1 if y[i] == y_pre[i] == 1 else 0 for i in range(len(y))]
        lab = np.asarray(lab)
        pre_fact_num = np.sum(lab)

        # 比值
        precision = pre_fact_num / pre_num

        return precision

    # 查全率
    def recall(self, y, y_pre):

        # 实际为真的数量
        fact_num = np.sum(y == 1)

        # 预测值中实际为真的数量
        lab = [1 if y[i] == y_pre[i] == 1 else 0 for i in range(len(y))]
        lab = np.asarray(lab)
        pre_fact_num = np.sum(lab)

        # 比值
        recall = pre_fact_num / fact_num

        return recall

    # F1 score
    def F1_score(self, y, y_pre):

        # 预测值中实际为真的数量
        lab_true = [1 if y[i] == y_pre[i] == 1 else 0 for i in range(len(y))]
        lab_true = np.asarray(lab_true)
        pre_true_num = np.sum(lab_true)

        # 预测值中实际为假的数量
        lab_false = [1 if y[i] == y_pre[i] == 0 else 0 for i in range(len(y))]
        lab_false = np.asarray(lab_false)
        pre_false_num = np.sum(lab_false)

        # 比值
        upper = 2 * pre_true_num
        downer = len(y) + pre_true_num - pre_false_num
        f1 = upper / downer

        return f1

    def PR_curve(self, y, y_pre_data):

        # 实际为真的数量
        fact = np.sum(y == 1)

        # 以预测值设置阈值区间
        thresholds = np.unique(y_pre_data)
        thresholds = sorted(thresholds, reverse=True)

        # initialization
        precisions = []
        recalls = []

        for threshold in thresholds:

            # 大于等于阈值判断为真
            lab = [1 if y_pre_data[i] >= threshold else 0 for i in range(len(y_pre_data))]
            lab = np.asarray(lab)

            # 判断为真中实际为真
            lab_pre = [1 if lab[i] == y[i] == 1 else 0 for i in range(len(lab))]
            lab_pre = np.asarray(lab_pre)

            # 该阈值下的查准率
            precision = np.sum(lab_pre) / np.sum(lab)
            precisions.append(precision)

            # 该阈值下的查全率
            recall = np.sum(lab_pre) / fact
            recalls.append(recall)

        return precisions, recalls

    # PR曲线下的面积
    def PR_area(self, y, y_pre_data):

        y_axis, x_axis = self.PR_curve(y, y_pre_data)
        y_axis_min = min(y_axis)
        areas = 0

        for i in range(len(x_axis) - 1):

            # 小三角
            triangle = 1/2 * (x_axis[i+1] - x_axis[i]) * abs(y_axis[i] - y_axis[i+1])

            # 矩形
            rectangle = (x_axis[i+1] - x_axis[i]) * y_axis[i+1]

            area = triangle + rectangle
            areas += area

        return areas

    # ROC curve
    def ROC(self, y, y_pre_data):

        # 根据precision值重排label
        data = pd.DataFrame({'testing': y, 'precision': y_pre_data})
        data = data.sort_values(by='precision', ascending=False, ignore_index=True)

        # initialization
        tpr = 0
        fpr = 0
        tprs = [0]
        fprs = [0]

        true_num = np.sum(y == 1)
        false_num = np.sum(y == 0)
        m_0 = 1 / true_num
        m_1 = 1 / false_num

        for i in range(len(data)):

            # 将分类阈值设为每个样例的预测值
            if data['testing'][i] == 1:
                tpr += m_0
                tprs.append(tpr)
                fprs.append(fpr)

            if data['testing'][i] == 0:
                fpr += m_1
                fprs.append(fpr)
                tprs.append(tpr)

        return fprs, tprs

    # Area under ROC curve
    def AUC(self, y, y_pre_data):

        x_axis, y_axis = self.ROC(y, y_pre_data)
        areas = 0

        for i in range(len(x_axis) - 1):

            area = y_axis[i+1] * (x_axis[i+1]-x_axis[i])
            areas += area

        return areas