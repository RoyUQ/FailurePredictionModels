import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import pickle

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"
test_label_path = "D:\\zoomlion\\train_test_data_1h_20190117\\test_labels.csv"
train_x = []
train_y = []

if __name__ == '__main__':
    best_result = pd.DataFrame()
    type_models = {'ZV55eec': 'xgboost_ZV55eec', 'ZV75a42': 'rf_ZV75a42',
                   'ZVe0672': 'xgboost_ZVe0672', 'ZV41153': 'xgboost_ZV41153',
                   'ZV90b78': 'xgboost_ZV90b78', 'ZVc1d93': 'xgboost_ZVc1d93',
                   'ZV7e8e3': 'rf_ZV7e8e3'}
    for my_type in type_models.keys():
        train_data = pd.read_csv(default_path + 'train_' + my_type + ".csv")
        test_data = pd.read_csv(default_path + 'test_' + my_type + '.csv')
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(
            np.array(train_data.drop(['标签', '设备类型', '文件名'], axis=1)))
        test_x = min_max_scaler.transform(
            np.array(test_data.drop(['标签', '设备类型', '文件名'], axis=1)))
        with open(default_path + type_models[my_type], 'rb') as f2:
            model = pickle.load(f2)
        y_pred = model.predict(test_x)
        result = pd.DataFrame(columns=['ID', 'Label'])
        result['ID'] = test_data['文件名']
        result['Label'] = y_pred
        best_result = pd.concat([best_result, result], axis=0,
                                ignore_index=True)
    best_result.to_csv(default_path + 'best_result.csv')
    test_result = pd.read_csv(test_label_path)
    print('----------------------最终结果------------------------')
    best_result = best_result.sort_values('ID')
    test_result = test_result.sort_values('sample_file_name')
    best_f1 = f1_score(best_result['Label'].values,
                       test_result['label'].values)
    best_accuracy = accuracy_score(best_result['Label'].values,
                                   test_result['label'].values)
    print("Accuracy: %.2f%%" % (best_accuracy * 100.0))
    print("f1: %s" % best_f1)
