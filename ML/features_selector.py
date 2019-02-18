from sklearn.feature_selection import RFECV
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

default_path = "D:\\zoomlion\\train_test_data_1h_20190117\\"

gbm1 = xgb.XGBClassifier(n_estimators=891,
                         max_depth=13,
                         learning_rate=0.225,
                         min_child_weight=1,
                         subsample=0.75,
                         gamma=0.95,
                         colsample_bytree=1,
                         objective='binary:logistic',
                         booster='gbtree',
                         seed=1000)

gbm2 = xgb.XGBClassifier()


def select_features(data, model):
    x = np.array(data.drop(['标签', '设备类型', '文件名'], axis=1))
    y = np.array(data['标签'])
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(3),
                  scoring='f1')
    rfecv.fit(x, y)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Ranking of features : %s" % rfecv.ranking_)


if __name__ == '__main__':
    data = pd.read_csv(default_path + 'ZV55eec.csv')
    select_features(data, gbm2)
