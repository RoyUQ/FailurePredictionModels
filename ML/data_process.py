# -*- coding: UTF-8 -*-
import pandas as pd

"""
此脚本用于拆分数据
"""


def process(path1: str, path2: str, file: str, interval):
    """
    拆分数据
    :param path1: 读取路径
    :param path2: 存储路径
    :param file: 文件名
    :param interval: 拆分间隔
    """
    df = pd.read_csv(path1 + file)

    # 根据给定的分组长度进行分组
    groups = int(df["活塞剩余时长"].max() / interval)

    file_labels = {}  # 创建字典储存子文件名和对应标签
    df_size = len(df)

    # 确定每组的行数
    nrows = int(df_size / groups)

    for group in range(groups):
        if group == groups - 1:
            df_temp = df.iloc[group * nrows:]  # 剩余数据组成最后的子数据集
        else:
            # 对原始数据集进行分组
            df_temp = df.iloc[group * nrows:(group + 1) * nrows]

        df_temp = df_temp.drop(["活塞剩余时长"], axis=1)
        df_temp.rename(columns={df.columns[0]: ""}, inplace=True)

        # 统计子数据集的标签类别和数量，同时确定子数据集的标签
        temp_counts = df_temp["是否将故障"].value_counts()
        if temp_counts.index.size == 1:
            temp_label = temp_counts.index
        elif temp_counts.index.size == 2:
            if temp_counts[0] > temp_counts[1]:
                temp_label = 0
            elif temp_counts[0] < temp_counts[1]:
                temp_label = 1
            else:
                temp_label = ""
        else:
            raise ValueError("Label不能超过2种")

        # 将子数据集文件名与标签存入字典
        file_name = file[:len(file) - 4] + "_" + str(group + 1) + ".csv"
        file_labels[file_name] = temp_label
        # 导出子数据集
        df_temp.to_csv(path2 + file_name, index=False)

    # 导出统计结果
    result = pd.DataFrame(file_labels).transpose()
    result.to_csv(path2 + file[:len(file) - 4] + "_" + "result" + ".csv",
                  header=None)


if __name__ == '__main__':
    path = "D:\\zoomlion\\"
    file = "201612251612107993_18.csv"
    interval = 2
    process(path, path, file, interval)
