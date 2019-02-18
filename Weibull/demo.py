from Weibull import weibull_functions, weibull_model
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def failure_plot(x, shape, scale, location):
    """画出不同寿命时间点的失效概率图像"""
    x.sort()
    x = np.linspace(x.min(), x.max())
    ydata = (shape / scale) * ((x - location) / scale) ** (shape - 1)
    mask = ydata > 1
    ydata[mask] = 1
    plt.plot(x, ydata)
    plt.xlabel("Time")
    plt.ylabel("Failure rate")
    plt.title("Failure rate function")
    plt.show()


def reliability_plot(x, shape, scale, location):
    """画出不同寿命时间点的可靠度图像"""
    x.sort()
    x = np.linspace(x.min(), x.max())
    ydata = np.exp(-((x - location) / scale) ** shape)
    plt.plot(x, ydata)
    plt.xlabel("Time")
    plt.ylabel("Reliability")
    plt.title("Reliability function")
    plt.show()


def lifetime_prob_plot(shape, scale, location, t):
    """基于不同的预期寿命画出对应的可靠度图像"""
    x = np.linspace(0, 2 * scale)
    ydata = np.exp(((t - location) ** shape - (
            t + x - location) ** shape) / scale ** shape)
    plt.plot(x, ydata)
    plt.xlabel("Time")
    plt.ylabel("Reliability")
    plt.title("Reliability function based on current time")
    plt.show()


def predict_lifespan_plot(shape, scale, location, t):
    """基于不同的可靠度画出剩余寿命的图像"""
    prob = np.linspace(0.01, 1)
    ydata = location - t + pow(
        (t - location) ** shape - scale ** shape * np.log(prob), 1 / shape)
    plt.plot(prob, ydata)
    plt.xlabel("Reliability")
    plt.ylabel("Remain life")
    plt.title("Predicting remain life function")
    plt.show()


def plot_model(x, shape, scale, location):
    """拟合weibull分布图形"""
    x.sort()
    ydata = stats.weibull_min.pdf(np.linspace(x.min(), x.max()),
                                  shape, location,
                                  scale)
    plt.plot(np.linspace(x.min(), x.max()), ydata, '-')
    plt.hist(x, bins=np.linspace(0, x.max()), normed=True,
             alpha=0.5)
    plt.xlabel("Failure time")
    plt.ylabel("Proportion")
    plt.title("Weibull distribution")
    plt.show()


if __name__ == '__main__':
    # 模拟100000个符合weibull分布的数据，形状参数为2.2096，尺度参数为5.9264，位置参数为2.8104
    x = stats.weibull_min.rvs(2.2096, loc=2.8104, scale=5.9264, size=100000)
    w = weibull_model.Weibull(x, 2.8104)  # 创建模型对象， 位置参数为2.8104

    shape, scale = w.fit()  # 通过迭代得到估计的形状与尺度参数
    print("形状与尺度参数的估计值 --------------------------:")
    print("形状参数: {}".format(shape) + " 尺度参数: {}".format(scale))

    w.save_model('weibull_model')

    plot_model(x, shape, scale, 2.8104)  # 画出weibull分布图像

    w2 = weibull_functions.Weibull_functions('weibull_model')  # 创建模型功能对象

    print("零件寿命为5年时的失效概率 --------------------------:")
    failure_rate = w2.failure_function(5)  # 估计零件寿命为5年时的失效概率
    print(failure_rate)
    failure_plot(x, shape, scale, 2.8104)
    # print(w2.failure_function(20))

    print("零件寿命为5年时的可靠度 --------------------------:")
    reliability = w2.reliability_function(5)  # 估计零件寿命为5年时的可靠度
    print(reliability)
    reliability_plot(x, shape, scale, 2.8104)
    # print(w2.reliability_function(1))

    print("基于当前寿命等于3年时，期望剩余寿命为一年时的可靠度--------------------------:")
    prob = w2.lifetime_reliability(1, 3)  # 基于当前寿命等于3年时，估计期望剩余寿命为一年时的可靠度
    print(prob)
    lifetime_prob_plot(shape, scale, 2.8104, 3)

    print("基于当前寿命等于3年时，预测剩余寿命(可靠度为prob) --------------------------:")
    print(w2.predict_lifespan(prob, 3))  # 基于当前寿命等于3年时，预测剩余寿命(可靠度为prob)
    predict_lifespan_plot(shape, scale, 2.8104, 3)

    print("平均失效前时间 --------------------------:")
    print(w2.mttf())