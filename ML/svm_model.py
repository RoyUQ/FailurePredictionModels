import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, hp, tpe
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"
test_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\test_labels.csv"

# 需要先使用xgboost_model.save_data() 保存数据，然后再将npy文件加载到train_x和train_y
train_x = []
train_y = []


def score(params: dict):
    """
    评估函数（基于F1值）
    :param params: 待调参数
    :return: 交叉验证后F1的均值
    """
    my_svm = svm.SVC(C=params['C'],
                     kernel='rbf',
                     gamma=params['gamma'])
    sampled_x = train_x
    sampled_y = train_y
    # 使用F1最为评估值；交叉验证次数为10
    metric = cross_val_score(my_svm, sampled_x, sampled_y, cv=10,
                             scoring="f1").mean()
    print(metric)
    return -metric


def optimize():
    """
    基于评估函数优化参数
    :return: 优化后的参数
    """
    space = {
        'C': hp.loguniform('C', 0, 4),
        'gamma': hp.loguniform('gamma', 2, 5)
    }
    # hyperopt.rand.suggest -- 随机搜索算法
    # hyperopt.anneal.suggest -- 模拟退火算法
    # hyperopt.tpe.suggest -- tpe算法
    # 使用tpe算法优化参数
    best = fmin(score, space, algo=tpe.suggest, max_evals=10)  # 默认搜索10次
    print(best)
    return best


# if __name__ == '__main__':
#     # 加载训练数据
#     train_x = np.load(default_path + "sampled_x.npy")
#     train_y = np.load(default_path + "sampled_y.npy")
#
#     transformer = Normalizer().fit(train_x)
#     print(transformer)
#     train_x = transformer.transform(train_x)  # 标准化训练数据
#
#     # 优化参数并返回最佳参数
#     best_params = optimize()
#     print(best_params)
#
#     # 根据优化后的参数重构模型
#     my_svm = svm.SVC(C=best_params['C'],
#                      kernel='rbf',
#                      gamma=best_params['gamma'])
#
#     # 加载测试数据
#     test_x = np.load(default_path + "test_x.npy")
#     test_y = np.load(default_path + "test_y.npy")
#
#     # 标准化测试数据
#     test_x = transformer.transform(test_x)
#
#     # 模型拟合和预测
#     my_svm.fit(train_x, train_y)
#     y_pred = my_svm.predict(test_x)
#
#     # 储存模型
#     with open(default_path + 'svm_model', 'wb') as f:
#         pickle.dump(my_svm, f)
#
#     # 计算Accuracy 和 F1-score
#     predictions = [round(value) for value in y_pred]
#     accuracy = accuracy_score(test_y, predictions)
#     f1 = f1_score(test_y, predictions)
#
#     # 将结果匹配文件名并输出为csv格式
#     result = pd.DataFrame(columns=['ID', 'Label'])
#     result['ID'] = pd.read_csv(test_label_path)['文件名']
#     result['Label'] = predictions
#     result.to_csv(default_path + "svm_result")
#
#     print("Accuracy: %.2f%%" % (accuracy * 100.0))
#     print("f1: %s" % f1)
