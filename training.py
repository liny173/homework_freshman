import pandas as pd
import numpy as np
from functions import Logistic_model
# import matplotlib.pyplot as plt

# # 数据读入
# data = pd.read_csv('data.csv')
#
# # 打乱数据集
# data = data.sample(frac=1).reset_index(drop=True)
#
# # 划分训练集与测试集，比例为7:3
# train_data = data.sample(frac=0.7, random_state=0, axis=0)
# text_data = data[~(data.index.isin(train_data.index))]
#
# # 重置index
# train_data = train_data.reset_index(drop=True)
# text_data = text_data.reset_index(drop=True)
#
# # 保存训练集与测试集
# train_data.to_csv('D:\\PycharmProjects\\machine_1\\train_data.csv')
# text_data.to_csv('D:\\PycharmProjects\\machine_1\\text_data.csv')

# 读入训练集
train = pd.read_csv('train_data.csv')

# 参与训练的特征值
train_data = train[train.columns[3:33]]
train_data = np.asarray(train_data)

# 训练集的分类标签
train_label = train['diagnosis']
train_label_data = [1 if i == 'M' else 0 for i in train_label]
train_label_data = np.asarray(train_label_data)

# 读入测试集
text = pd.read_csv('text_data.csv')

# 参与测试的特征值
text_data = text[text.columns[3:33]]
text_data = np.asarray(text_data)

# 测试集的分类标签
text_label = text['diagnosis']
text_label_data = [1 if i == 'M' else 0 for i in text_label]
text_label_data = np.asarray(text_label_data)

if __name__ == '__main__':

    # 模型创建
    logistic = Logistic_model(train_data, 0.0005)

    # 梯度下降
    logistic.gradient_descent(train_label_data)

    # 预测
    text_pre, text_pre_label = logistic.predict(text_data)

    # 损失
    cost = logistic.cost(text_label_data, text_pre_label)

    # 准确率
    accuracy = logistic.accuracy(text_label_data, text_pre_label)

    # 查准率
    precision = logistic.precision(text_label_data, text_pre_label)

    # 查全率
    recall = logistic.recall(text_label_data, text_pre_label)

    # F1 score
    f1_score = logistic.F1_score(text_label_data, text_pre_label)

    # AUPR
    aupr = logistic.PR_area(text_label_data, text_pre)

    # AUC
    auc = logistic.AUC(text_label_data, text_pre)

    # 一些指标的输出
    print(f'cost: {cost}')
    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1 score: {f1_score}')
    print(f'AUPR: {aupr}')
    print(f'AUC: {auc}')

    '''
    分析：
      PR曲线中，以预测值设置阈值，查准率较高时，查全率也比较高，输出发现，
    预测值中，为1.00的数量较多，所以高阈值下的查全率较高。
      ROC曲线中，高阈值下的标签预测比较准确，一开始接近于垂直向上。
      AUC值约为0.9645，接近1，高阈值下的标签预测比较准确。
    '''

    # 绘图
    # PR curve
    # precisions, recalls = logistic.PR_curve(text_label_data, text_pre)
    # fig = plt.figure(figsize=(8, 6))
    # plt.plot(recalls, precisions)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('P-R Curve')
    # plt.savefig('./PR.png')
    # plt.show()

    # ROC curve
    # fpr, tpr = logistic.ROC(text_label_data, text_pre)
    # fig = plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr)
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC Curve')
    # plt.savefig('./ROC.png')
    # plt.show()
