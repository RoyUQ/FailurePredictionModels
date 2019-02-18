import numpy as np
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from ML import process_train_test

path1 = "D:\\zoomlion\\train_test_data\\data_train"
label_file1 = "D:\\zoomlion\\train_test_data\\train_labels.csv"
path2 = "D:\\zoomlion\\train_test_data\\data_test"
label_file2 = "D:\\zoomlion\\train_test_data\\test_labels.csv"
columns = ['活塞工作时长', '活塞工作方量', '发动机转速', '油泵转速', '泵送压力', '液压油温', '流量档位', '分配压力',
           '排量电流', '标签']


def xgboost_model(x: np.ndarray, y: np.ndarray):
    model = xgboost.XGBClassifier()
    model.fit(x, y)
    print(model)
    return model


def my_SVM(x: np.ndarray, y: np.ndarray):
    clf = SVC(gamma='auto')
    clf.fit(x, y)
    print(clf)
    return clf


def model_accuracy(model, test_x, test_y):
    y_pred = model.predict(test_x)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    print(predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("f1: %s" % f1)


if __name__ == '__main__':
    train_X, train_Y = process_train_test.data_process(path1, label_file1)
    train_X, train_Y = process_train_test.sampling(train_X, train_Y, columns,
                                                   down=1)
    print(np.sum(train_Y == 0))
    test_X, test_Y = process_train_test.data_process(path2, label_file2)
    xgb = xgboost_model(train_X, train_Y)
    # svm = my_SVM(train_X, train_Y)
    model_accuracy(xgb, test_X, test_Y)
    # model_accuracy(svm, test_X, test_Y)
