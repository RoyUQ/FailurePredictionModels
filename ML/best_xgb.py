import pandas as pd
import numpy as np
from ML import xgboost_model
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
from sklearn.preprocessing import MinMaxScaler

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"
train_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_train"
train_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\train_labels.csv"
test_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_test"
test_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\test_labels.csv"


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
    # 对数据进行标准化处理
    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    # 将处理后的数据赋值给模型文件的全局变量
    xgboost_model.train_x = x
    xgboost_model.train_y = y
    # 参数优化
    best_params = xgboost_model.optimize()
    # 建立最优参数对应的模型
    gbm = xgb.XGBClassifier(n_estimators=int(best_params['n_estimators']),
                            max_depth=int(best_params['max_depth']),
                            learning_rate=best_params['learning_rate'],
                            min_child_weight=int(
                                best_params['min_child_weight']),
                            subsample=best_params['subsample'],
                            gamma=best_params['gamma'],
                            colsample_bytree=best_params[
                                'colsample_bytree'],
                            objective='binary:logistic',
                            booster='gbtree',
                            seed=1000)
    # 将DataFrame转换成可输入到模型的array形式
    test_x = np.array(test_data.drop(['标签', '设备类型', '文件名'], axis=1))
    test_y = np.array(test_data['标签'])
    gbm.fit(x, y)
    # 储存模型
    with open(default_path + "xgboost_" + my_type, 'wb') as f:
        pickle.dump(gbm, f)
    # 对测试数据进行标准化处理
    test_x = min_max_scaler.transform(test_x)
    y_pred = gbm.predict(test_x)
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
    types = ['ZV7e8e3', 'ZV55eec', 'ZV75a42', 'ZVe0672', 'ZV41153', 'ZV90b78',
             'ZVc1d93']
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
    final_result.to_csv(default_path + 'xgboost_result.csv')
    # 读取测试数据的结果
    test_result = pd.read_csv(test_label_path)
    print('----------------------最终结果------------------------')
    # 按文件名对DataFrame排序，方便计算F1-score
    final_result = final_result.sort_values('ID')  # 对预测结果排序
    test_result = test_result.sort_values('sample_file_name')  # 对测试数据排序
    final_f1 = f1_score(final_result['Label'].values,
                        test_result['label'].values)
    print(final_f1)

    # train_data = pd.read_csv(default_path + 'train_' + 'ZV90b78' + ".csv")
    # test_data = pd.read_csv(default_path + 'test_' + 'ZV90b78' + '.csv')
    # child_optimize(train_data, test_data, 'ZV90b78')
