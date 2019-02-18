import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, hp, tpe
from ML import process_train_test
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"
train_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_train"
train_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\train_labels.csv"
test_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_test"
test_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\test_labels.csv"

my_columns = ['活塞工作时长', '发动机转速', '油泵转速', '泵送压力', '液压油温', '流量档位', '分配压力',
              '排量电流', '标签']

# 需要先使用save_data 保存数据，然后将npy文件再加载到train_x和train_y
train_x = []
train_y = []


def save_data(train_path, test_path, train_label_path, test_label_path,
              save_path, columns, sample_method=0):
    """
    存储数据
    :param train_path: 训练数据文件夹路径
    :param test_path:  测试数据文件夹路径
    :param train_label_path: 训练数据标签文件路径
    :param test_label_path: 测试数据标签文件路径
    :param save_path: 数组存储路径
    :param columns: 已选特征的列命（包含标签）
    :param sample_method: 向下采样为1；向上采样为2；默认为0--不采样
    """
    test_x, test_y = process_train_test.data_process(test_path, test_label_path,
                                                     columns[:-1])
    raw_x, raw_y = process_train_test.data_process(train_path, train_label_path,
                                                   columns[:-1])
    if sample_method == 1:
        sampled_x, sampled_y = process_train_test.sampling(raw_x, raw_y,
                                                           columns, down=1)
        np.save(save_path + "sampled_x.npy", sampled_x)
        np.save(save_path + "sampled_y.npy", sampled_y)
    elif sample_method == 2:
        sampled_x, sampled_y = process_train_test.sampling(raw_x, raw_y,
                                                           columns, up=1)
        np.save(save_path + "sampled_x.npy", sampled_x)
        np.save(save_path + "sampled_y.npy", sampled_y)
    else:
        np.save(save_path + "train_x.npy", raw_x)
        np.save(save_path + "train_y.npy", raw_y)
    np.save(save_path + "test_x.npy", test_x)
    np.save(save_path + "test_y.npy", test_y)


def score(params: dict):
    """
    评估函数（基于F1值）
    :param params: 待调参数
    :return: 交叉验证后F1的均值
    """
    gbm = xgboost.XGBClassifier(n_estimators=int(params['n_estimators']),
                                max_depth=int(params['max_depth']),
                                learning_rate=params['learning_rate'],
                                min_child_weight=int(
                                    params['min_child_weight']),
                                subsample=params['subsample'],
                                gamma=params['gamma'],
                                colsample_bytree=params['colsample_bytree'],
                                objective='binary:logistic',
                                booster='gbtree',
                                seed=1000)
    sampled_x = train_x
    sampled_y = train_y
    # 使用F1最为评估值；交叉验证次数为10
    metric = cross_val_score(gbm, sampled_x, y=sampled_y, cv=10,
                             scoring="f1").mean()
    print(metric)
    return -metric


def optimize():
    """
    基于评估函数优化参数
    :return: 优化后的参数
    """
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
        'max_depth': hp.randint("max_depth", 15),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.3, 1, 0.025),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.025),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1, 0.025),
    }
    # hyperopt.rand.suggest -- 随机搜索算法
    # hyperopt.anneal.suggest -- 模拟退火算法
    # hyperopt.tpe.suggest -- tpe算法
    # 使用tpe算法优化参数
    best = fmin(score, space, algo=tpe.suggest, max_evals=10)  # 默认搜索10次
    print(best)
    return best


# if __name__ == '__main__':
#     # 将数据存储成npy文件；使用向上采样
#     save_data(train_path, test_path, train_label_path, test_label_path,
#               default_path, my_columns)
#     # 加载训练数据
#     train_x = np.load(default_path + "sampled_x.npy")
#     train_y = np.load(default_path + "sampled_y.npy")
#
#     # 优化参数并返回最佳参数
#     best_params = optimize()
#     print(np.sum(train_y == 1))
#     print(best_params)
#
#     # 根据优化后的参数重构模型
#     gbm = xgboost.XGBClassifier(n_estimators=int(best_params['n_estimators']),
#                                 max_depth=int(best_params['max_depth']),
#                                 learning_rate=best_params['learning_rate'],
#                                 min_child_weight=int(
#                                     best_params['min_child_weight']),
#                                 subsample=best_params['subsample'],
#                                 gamma=best_params['gamma'],
#                                 colsample_bytree=best_params[
#                                     'colsample_bytree'],
#                                 objective='binary:logistic',
#                                 booster='gbtree',
#                                 seed=1000)
#
#     # 加载测试数据
#     test_x = np.load(default_path + "test_x.npy")
#     test_y = np.load(default_path + "test_y.npy")
#
#     # 模型拟合和预测
#     gbm.fit(train_x, train_y)
#     y_pred = gbm.predict(test_x)
#
#     # 储存模型
#     with open(default_path + 'xgboost_model', 'wb') as f:
#         pickle.dump(gbm, f)
#
#     predictions = [round(value) for value in y_pred]
#
#     # 计算Accuracy 和 F1-score
#     accuracy = accuracy_score(test_y, predictions)
#     f1 = f1_score(test_y, predictions)
#
#     # 将结果匹配文件名并输出为csv格式
#     result = pd.DataFrame(columns=['ID', 'Label'])
#     result['ID'] = pd.read_csv(test_label_path)['文件名']
#     result['Label'] = predictions
#     result.to_csv(default_path + "xgboost_result")
#
#     print("Accuracy: %.2f%%" % (accuracy * 100.0))
#     print("f1: %s" % f1)
