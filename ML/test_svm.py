import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, hp, tpe
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"
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
                     gamma=params['gamma'],
                     degree=params['degree'])
    sampled_x = train_x
    sampled_y = train_y
    # 使用F1最为评估值；交叉验证次数为10
    metric = cross_val_score(my_svm, sampled_x, y=sampled_y, cv=7,
                             scoring="f1").mean()
    print(metric)
    return -metric


def optimize():
    """
    基于评估函数优化参数
    :return: 优化后的参数
    """
    space = {
        'C': hp.choice('C', [0.001, 0.01, 0.1, 1, 10, 100]),
        'gamma': hp.choice('gamma', [0.0001, 0.001, 0.01, 0.1, 1, 10]),
        'degree': hp.choice('degree', [0, 1, 2, 3, 4, 5, 6])
    }
    # hyperopt.rand.suggest -- 随机搜索算法
    # hyperopt.anneal.suggest -- 模拟退火算法
    # hyperopt.tpe.suggest -- tpe算法
    # 使用tpe算法优化参数
    best = fmin(score, space, algo=tpe.suggest, max_evals=7)  # 默认搜索2次
    return best


if __name__ == '__main__':
    # train_x = np.load(default_path + "train_x.npy")
    # train_y = np.load(default_path + "train_y.npy")
    # transformer = Normalizer().fit(train_x)
    # # print(transformer)
    # train_x = transformer.transform(train_x)  # 正则化数据
    # scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)

    # 抽取发动机转速和油泵转速为特征
    train_data = pd.read_csv(default_path + 'train_all_data.csv')
    train_x = np.array(train_data.loc[:, ['发动机转速', '油泵转速']])
    train_y = np.array(train_data['标签'])

    # 将数据标准化
    min_max_scaler = MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    print(train_x)
    print(np.sum(train_y == 1))
    # best_params = optimize()
    # print(best_params)

    # 随机设置svm的参数
    my_svm = svm.SVC(C=10, kernel='rbf', gamma=15)

    # 根据优化后的参数重构模型
    # my_svm = svm.SVC(
    #     C=best_params['C'],
    #     kernel='rbf',
    #     gamma=best_params['gamma'],
    #     degree=best_params['degree']
    # )
    # C = best_params['C'],
    # kernel = 'rbf',
    # gamma = best_params['gamma']

    # 加载测试数据并抽取发动机转速和油泵转速为特征
    test_data = pd.read_csv(default_path + 'test_all_data.csv')
    test_x = np.array(test_data.loc[:, ['发动机转速', '油泵转速']])
    test_y = np.array(test_data['标签'])

    my_svm.fit(train_x, train_y)
    # test_x = transformer.transform(test_x)
    # test_x = scaler.transform(test_x)
    # test_x = min_max_scaler.transform(test_x)
    print(test_x)
    # 此处不对测试数据进行标准化
    y_pred = my_svm.predict(test_x)

    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("f1: %s" % f1)
