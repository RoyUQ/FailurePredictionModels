import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, hp, tpe
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
from sklearn.preprocessing import MinMaxScaler

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"
train_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_train"
train_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\train_labels.csv"
test_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_test"
test_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\test_labels.csv"

train_x = []
train_y = []


def score(params):
    """
    评估函数（基于F1值）
    :param params: 待调参数
    :return: 交叉验证后F1的均值
    """
    clf = RandomForestClassifier(**params)
    # 使用F1最为评估值；交叉验证次数为10
    metric = cross_val_score(clf, train_x, y=train_y, cv=10,
                             scoring="f1").mean()
    print(metric)
    return -metric


def optimize():
    """
    基于评估函数优化参数
    :return: 优化后的参数
    """
    space_rf = {
        'max_depth': hp.choice('max_depth', range(10, 30)),
        'max_features': hp.choice('max_features', range(1, 15)),
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
    }
    # hyperopt.rand.suggest -- 随机搜索算法
    # hyperopt.anneal.suggest -- 模拟退火算法
    # hyperopt.tpe.suggest -- tpe算法
    # 使用tpe算法优化参数
    best = fmin(score, space_rf, algo=tpe.suggest, max_evals=10)  # 默认搜索10次
    print(best)
    return best


def child_optimize(train_data, test_data, my_type):
    """
    基于不同型号进行参数优化，数据格式均为DataFrame
    :param train_data: 训练数据
    :param test_data: 测试数据
    :param my_type: 目标类型
    :return: 包含预测结果和文件名的DataFrame
    """
    # 将DataFrame转换成可输入到模型的array形式
    x = np.array(train_data.drop(['标签', '设备类型', '文件名'], axis=1))
    y = np.array(train_data['标签'])
    global train_x
    global train_y
    train_x = x
    train_y = y
    # 对数据进行标准化处理
    min_max_scaler = MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    # 参数优化
    best_params = optimize()
    # 建立最优参数对应的模型
    criterion = ["gini", "entropy"]
    clf = RandomForestClassifier(
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        n_estimators=best_params['n_estimators'],
        criterion=criterion[best_params['criterion']]
    )
    # 将DataFrame转换成可输入到模型的array形式
    test_x = np.array(test_data.drop(['标签', '设备类型', '文件名'], axis=1))
    test_y = np.array(test_data['标签'])
    clf.fit(x, y)
    # 储存模型
    with open(default_path + "rf_" + my_type, 'wb') as f:
        pickle.dump(clf, f)
    # 对测试数据进行标准化处理
    test_x = min_max_scaler.transform(test_x)
    y_pred = clf.predict(test_x)
    # 计算Accuracy 和 F1-score
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("f1: %s" % f1)
    # 将预测结果和文件名绑定形成DataFrame格式返回
    result = pd.DataFrame(columns=['ID', 'Label'])
    result['ID'] = test_data['文件名']
    result['Label'] = y_pred
    return result


if __name__ == '__main__':
    types = ['ZV55eec', 'ZV75a42', 'ZVe0672', 'ZV41153', 'ZV90b78', 'ZVc1d93',
             'ZV7e8e3']
    final_result = pd.DataFrame()
    for my_type in types:  # 对所有类型逐个进行参数优化并预测
        # 获取该型号的训练数据和测试数据
        train_data = pd.read_csv(default_path + 'train_' + my_type + ".csv")
        test_data = pd.read_csv(default_path + 'test_' + my_type + '.csv')

        # 将不同型号的预测结果进行拼接
        result = child_optimize(train_data, test_data, my_type)
        final_result = pd.concat([final_result, result], axis=0,
                                 ignore_index=True)
    # 将整合的预测结果输出成csv文件
    final_result.to_csv(default_path + 'rf_result.csv')
    # 读取测试数据的结果
    test_result = pd.read_csv(test_label_path)
    print('----------------------最终结果------------------------')
    # 按文件名对DataFrame排序，方便计算F1-score
    final_result = final_result.sort_values('ID')  # 对预测结果排序
    test_result = test_result.sort_values('sample_file_name')  # 对测试数据排序
    final_f1 = f1_score(final_result['Label'].values,
                        test_result['label'].values)
    print(final_f1)
