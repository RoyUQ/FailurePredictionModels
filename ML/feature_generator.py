import numpy as np
import os
import pandas as pd

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"
train_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_train"
train_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\train_labels.csv"
test_path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_test"
test_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\test_labels.csv"


# my_columns = ['活塞工作时长', '发动机转速', '油泵转速', '泵送压力', '液压油温', '流量档位', '分配压力',
#               '排量电流', '标签']

# def regularit(df):
#     newDataFrame = pd.DataFrame(index=df.index)
#     columns = df.columns.tolist()
#     for c in columns:
#         if c == '设备类型' or c == '文件名':
#             newDataFrame[c] = df[c].tolist()
#             continue
#         d = df[c]
#         MAX = d.max()
#         MIN = d.min()
#         if MAX - MIN == 0:
#             newDataFrame[c] = df[c].tolist()
#             continue
#         newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
#     return newDataFrame


def feature_generate(path, label, type=None):
    """
    产生方差数据并根据不同型号的泵车拆分成不同的csv文件
    :param path: 数据存放位置
    :param label: 标签数据文件位置
    :param type: 泵车型号。默认为None，处理全数据
    :return: 处理完成之后的DataFrame
    """
    raw_Y = pd.read_csv(label)
    raw_Y.index = raw_Y.iloc[:, 0]  # 使文件名作为index，方便标签匹配
    raw_X = pd.DataFrame()

    for info in os.listdir(path):
        domain = os.path.abspath(path)  # 获取文件夹的路径
        data = pd.read_csv(os.path.join(domain, info))
        if type is not None:  # 跳过非目标类型的文件
            if data['设备类型'][0] != type:
                continue
        # 计算文件中对应列的方差
        var_dict = {
            '发动机转速方差': data['发动机转速'].var(),
            '油泵转速方差': data['油泵转速'].var(),
            '泵送压力方差': data['泵送压力'].var(),
            '液压油温方差': data['液压油温'].var(),
            '流量档位方差': data['流量档位'].var(),
            '分配压力方差': data['分配压力'].var(),
            '排量电流方差': data['排量电流'].var()
        }
        if len(data) <= 1:  # 对只有一行数据的文件，默认方差为0
            for key in var_dict.keys():
                var_dict[key] = 0
        columns = data.columns.values.tolist()  # 获取列名
        num_data = data.mean().values  # 获取均值
        num_data = np.append(num_data, raw_Y.loc[info]["label"])  # 获取标签
        columns.append('标签')
        columns.remove("设备类型")
        train_x1 = pd.DataFrame(var_dict, index=[0])  # 方差数据
        train_x2 = pd.DataFrame(columns=columns)  # 均值数据
        train_x2.loc[0] = num_data
        train_x2['设备类型'] = data['设备类型'][0]
        train_x2['文件名'] = info

        train_x = pd.concat([train_x1, train_x2], axis=1, sort=False)  # 合并方差与均值数据
        if train_x.isnull().values.any():  # 检查是否有异常值
            print(info)
            print(train_x)
        raw_X = pd.concat([raw_X, train_x], axis=0, ignore_index=True)  # 拼接同一类型不同文件的数据
    process_data = raw_X
    return process_data


if __name__ == '__main__':
    #  处理全训练数据
    train_data = feature_generate(train_path, train_label_path)
    train_data.to_csv(default_path + 'train_all_data.csv')

    types = ['ZV7e8e3', 'ZV55eec', 'ZV75a42', 'ZVe0672', 'ZV41153', 'ZV90b78',
             'ZVc1d93']

    for my_type in types:  # 根据不同型号处理训练数据
        type_data = feature_generate(train_path, train_label_path, type=my_type)
        type_data.to_csv(default_path + 'train_' + my_type + '.csv')

    #  处理全测试数据
    test_data = feature_generate(test_path, test_label_path)
    test_data.to_csv(default_path + 'test_all_data.csv')

    for my_type in types:  # 根据不同型号处理测试数据
        type_data = feature_generate(test_path, test_label_path, type=my_type)
        type_data.to_csv(default_path + 'test_' + my_type + '.csv')
    # variance = process_data.var()
    # low_key = []
    # for key in variance.keys():
    #     if variance[key]<0.001:
    #         low_key.append(key)
    # process_data.drop()
