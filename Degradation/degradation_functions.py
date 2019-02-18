import pickle
import numpy as np


class Degradation_functions:
    def __init__(self, path: str):
        with open(path, 'rb') as f2:
            self.model = pickle.load(f2)

    def inverse_linear(self, y):
        """线性模型的反函数"""
        return (y - self.model.a) / self.model.b

    def inverse_exponential(self, y):
        """指数模型的反函数"""
        return np.log(y / self.model.a) / self.model.b

    def inverse_logarithmic(self, y):
        """对数模型的反函数"""
        return np.exp((y - self.model.a) / self.model.b)

    def predict_lifetimes(self, threshold_value):
        """
        根据给定的阈值来预测预期寿命
        :param threshold_value: 阈值
        :return: 预期寿命
        """
        if self.model.model_func.__name__ == 'linear_function':
            return self.inverse_linear(threshold_value)*self.model.proportion
        elif self.model.model_func.__name__ == 'exponential_function':
            return self.inverse_exponential(threshold_value)*self.model.proportion
        elif self.model.model_func.__name__ == 'logarithmic_function':
            return self.inverse_logarithmic(threshold_value)*self.model.proportion
        else:
            raise ValueError('不存在该模型的反函数')

    def predict_degradation(self, t):
        """
        根据给定寿命估计退化数据
        :param t: 给定寿命
        :return: 退化数据
        """
        t = t / self.model.proportion
        return self.model.model_func(t, self.model.a, self.model.b)
