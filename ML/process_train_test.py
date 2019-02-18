import os
import pandas as pd
import numpy as np
from sklearn.utils import resample

# path1 = "D:\\zoomlion\\train_test_data\\data_train"
# label_file1 = "D:\\zoomlion\\train_test_data\\train_labels.csv"
# path2 = "D:\\zoomlion\\train_test_data\\data_test"
# label_file2 = "D:\\zoomlion\\train_test_data\\test_labels.csv"


def data_process(path, label,columns):
    """
    数据预处理
    :param path: 数据存储的文件夹路径
    :param label: 数据对应标签的文件路径
    :param columns: 待选特征的列命（不包含标签）
    :return: X 数组，Y 数组
    """
    raw_Y = pd.read_csv(label)
    raw_Y.index = raw_Y.iloc[:, 0]
    processed_X = []
    processed_Y = []
    for info in os.listdir(path):
        domain = os.path.abspath(path)  # 获取文件夹的路径
        data = pd.read_csv(os.path.join(domain, info))
        # data.columns = ['活塞工作时长', '活塞工作方量', '发动机转速', '油泵转速', '泵送压力', '液压油温',
        #                 '流量档位',
        #                 '分配压力', '排量电流', '低压开关', '高压开关', '搅拌超压信号', '正泵', '反泵',
        #                 '设备类型']
        # data.drop(['低压开关', '高压开关', '搅拌超压信号', '正泵', '反泵', '设备类型'], axis=1)
        # data['活塞工作时长'] = data['活塞工作时长'].mean()
        # data['活塞工作方量'] = data['活塞工作方量'].mean()
        # data['发动机转速'] = data['发动机转速'].mean()
        # data['油泵转速'] = data['油泵转速'].mean()
        # data['泵送压力'] = data['泵送压力'].mean()
        # data['液压油温'] = data['液压油温'].mean()
        # data['流量档位'] = data['流量档位'].mean()
        # data['分配压力'] = data['分配压力'].mean()
        # data['排量电流'] = data['排量电流'].mean()
        data = data.mean()
        processed_X.append(data[columns].values)  # 添加输入值
        processed_Y.append(raw_Y.loc[info]["label"])  # 找到对应的文件标签
    processed_X = np.array(processed_X)
    processed_Y = np.array(processed_Y)
    return processed_X, processed_Y


def sampling(x, y, columns, up=0, down=0):
    """
    采样方法
    :param x: X 数组
    :param y: Y（标签） 数组
    :param columns: 已选特征的列命（包含标签）
    :param up: 1 - 向上采样
    :param down: 1 - 向下采样
    :return: 采样后的X 数组和Y 数组
    """
    data = np.concatenate([x, y[:, None]], axis=1)
    df = pd.DataFrame(data, columns=columns)
    df_majority = df[df.标签 == 0]
    df_minority = df[df.标签 == 1]
    if up == 1:
        df_sampled = resample(df_minority, replace=True,
                              n_samples=len(df_majority), random_state=123)
        df_processd = pd.concat([df_majority, df_sampled])
    elif down == 1:
        df_sampled = resample(df_majority, replace=True,
                              n_samples=len(df_minority), random_state=123)
        df_processd = pd.concat([df_sampled, df_minority])
    else:
        raise ValueError("需要选择采样方式")

    processed_Y = np.array(df_processd.标签)
    processed_X = np.array(df_processd.drop(['标签'], axis=1))
    return processed_X, processed_Y
