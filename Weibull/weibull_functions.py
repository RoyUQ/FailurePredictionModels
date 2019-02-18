import pickle
import numpy as np
from scipy.special import gamma


class Weibull_functions:
    def __init__(self, path: str):
        with open(path, 'rb') as f2:
            self.model = pickle.load(f2)

    def failure_function(self, x):
        """
        :param x: 元件的已使用寿命或期望寿命
        :return: 当前寿命或期望寿命的失效概率
        """
        if x < self.model.location:
            x = self.model.location
        result = (self.model.shape / self.model.scale) * (
                (x - self.model.location) / self.model.scale) ** (
                         self.model.shape - 1)
        if result > 1:
            result = 1
        return result

    def reliability_function(self, x):
        """
        :param x: 元件的期望寿命
        :return: 期望寿命的可靠度
        """
        if x < self.model.location:
            return 1
        else:
            return np.exp(-((x - self.model.location) / self.model.scale) ** self.model.shape)

    def lifetime_reliability(self, x, t):
        """
        :param x: 期望寿命
        :param t: 元件的已使用寿命
        :return: 基于元件的已使用寿命所对应的期望寿命的可靠度
        """
        if t < self.model.location:
            t = self.model.location
        if x + t < self.model.location:
            return 1
        return np.exp(
            ((t - self.model.location) ** self.model.shape - (
                    t + x - self.model.location) ** self.model.shape) / (
                self.model.scale) ** self.model.shape)

    def predict_lifespan(self, prob, t):
        """
        :param prob: 可接受的最低可靠度
        :param t: 元件的已使用寿命
        :return: 剩余寿命的预测值
        """
        if prob <= 0:
            raise ValueError('可靠度不能为负数')
        if t < self.model.location:
            t = self.model.location
        return self.model.location - t + pow(
            (t - self.model.location) ** self.model.shape - (
                self.model.scale) ** self.model.shape * np.log(
                prob), 1 / self.model.shape)

    def mttf(self):
        """计算并返回平均失效时间"""
        return self.model.scale * gamma(
            1.0 / self.model.shape + 1) + self.model.location

    # def show_model(self):
    #     """展示拟合过的weibull模型"""
    #     img = plt.imread("weibull_fig.png")
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.title('Weibull Model')
    #     plt.show()
