import numpy as np
import pandas as pd
import pickle
from scipy.optimize import curve_fit


class Degradation:
    """ Degradation model """

    def __init__(self, x):
        if isinstance(x, pd.DataFrame):
            if x.isnull().any().any():
                raise ValueError('数据不能包含空值')
            self.x = x.values.transpose()
        elif isinstance(x, np.ndarray):
            self.x = x
        else:
            raise TypeError("数据格式必须为DataFrame或者ndarray")
        xlist = []
        ylist = []
        self.proportion = self.data_process(self.x[0])
        self.x[0] = self.x[0] / self.proportion
        for i in range(1, len(self.x)):
            xlist.append(self.x[0])  # 默认第一列为x
            ylist.append(self.x[i])  # 默认除第一列外为y
        self.xdata = np.concatenate(xlist)
        self.ydata = np.concatenate(ylist)  # 多重样本拟合
        self.a = None  # 第一个参数
        self.b = None  # 第二个参数
        self.model_func = None  # 模型的函数方程

    @staticmethod
    def linear_function(t, a, b):
        return a + b * t

    @staticmethod
    def exponential_function(t, a, b):
        return a * np.exp(b * t)

    @staticmethod
    def logarithmic_function(t, a, b):
        if t.min() < 1:
            raise ValueError('对数模型的t值必须大于等于1')
        return a + b * np.log(t)

    @staticmethod
    def data_process(xdata: np.ndarray):
        """
        将数据等比例放大或缩小至可接受范围
        :param xdata: 一维数组
        :return: 放大或缩小的比例
        """
        max_x = xdata.max()
        max_x = format(max_x)
        max_x = max_x.replace(".", "")
        max_x = float(max_x[0:2] + "." + max_x[2:])
        proportion = xdata.max() / max_x
        return proportion

    def model_evaluation(self, popt, model_func):
        """
        根据R方值和MSE来评估模型的拟合程度
        :param popt: 模型的参数
        :param model_func: 模型的函数
        :return: R方和均方误差
        """
        residuals = self.ydata - model_func(self.xdata, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.ydata - np.mean(self.ydata)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        mse = np.mean((self.ydata - model_func(self.xdata, *popt)) ** 2)
        return r_squared, mse

    def model_fit(self, model=None):
        """
        根据不同的模型函数来估计参数
        :param model: 1-线性模型，2-指数模型，3-对数模型
        :return: 一维数组(popt):参数的估计值，model_func:模型的函数方程
        """
        if model == 1:
            model_func = self.linear_function
        elif model == 2:
            model_func = self.exponential_function
        elif model == 3:
            model_func = self.logarithmic_function
        else:
            raise ValueError('不存在对应模型')
        popt, pcov = curve_fit(model_func, self.xdata, self.ydata)
        return popt, model_func

    def optimize_fit(self):
        """
        根据现有的模型，找出R方值最大的模型
        :return: 一维数组(popt):参数的估计值，model_func:模型的函数方程
        """
        popt, model_func = self.model_fit(1)
        r_squared, mse = self.model_evaluation(popt, model_func)
        for i in range(2, 4):
            temp_popt, temp_func = self.model_fit(i)
            temp_r_squared, temp_mse = self.model_evaluation(temp_popt,
                                                             temp_func)
            if temp_r_squared > r_squared:
                r_squared = temp_r_squared
                popt = temp_popt
                model_func = temp_func
        return popt, model_func

    def save_model(self, popt, model_func, path: str):
        """
        存储模型
        :param popt: 模型的参数
        :param model_func: 模型的函数方程
        :param path: 存储路径
        """
        self.a = popt[0]
        self.b = popt[1]
        self.model_func = model_func
        with open(path, 'wb') as f:
            pickle.dump(self, f)
