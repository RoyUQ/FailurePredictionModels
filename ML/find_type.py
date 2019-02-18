import pandas as pd
import os

"""此文件用于找出所有的泵车类型"""

path = "D:\\zoomlion\\train_test_data_1h_20190117\\data_test"

if __name__ == '__main__':
    types = []
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        data = pd.read_csv(os.path.join(domain, info))
        if data['设备类型'][0] not in types:
            types.append(data['设备类型'][0])
            continue
    print(types)