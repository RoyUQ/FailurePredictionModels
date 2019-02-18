from Degradation import degradation_functions, degradation_model
import pandas as pd

if __name__ == '__main__':
    # 寿命数据为气缸的运动次数（十的六次方）， 退化数据为缸内气压
    dict = {'time': [200, 400, 600, 800, 1000, 1200, 1600, 1800],
            'mop1': [0.79, 0.927, 0.993, 1.070, 1.098, 1.043, 1.103, 1.143],
            'mop2': [0.847, 0.953, 1.0, 1.027, 1.063, 1.097, 1.097, 1.100],
            'mop3': [0.863, 0.947, 0.963, 0.947, 0.913, 1.037, 1.240, 1.213],
            'mop4': [0.910, 0.910, 0.850, 0.897, 1.010, 1.107, 1.120, 1.220],
            'mop5': [0.740, 0.733, 0.710, 0.880, 0.937, 1.073, 1.070, 1.067],
            'mop6': [0.877, 0.923, 0.900, 0.830, 0.723, 0.767, 1.030, 1.113],
            'mop7': [0.620, 0.623, 0.713, 0.810, 0.977, 0.980, 0.937, 0.900],
            'mop8': [0.857, 0.913, 0.990, 1.027, 1.063, 1.087, 1.073, 1.103]}
    data = pd.DataFrame(data=dict)

    dm = degradation_model.Degradation(data)
    popt, model_func = dm.optimize_fit()  # 使用最优化拟合方法来得到模型参数和函数方程
    print("最优模型的R方值和均方误差--------------------------:")
    print(dm.model_evaluation(popt, model_func))  # 得到模型的R方值和均方误差
    dm.save_model(popt, model_func, "degradation_model")  # 存储模型

    df = degradation_functions.Degradation_functions("degradation_model")  # 创建退化模型方法对象
    print("基于阈值为1.2的预期寿命--------------------------:")
    print(df.predict_lifetimes(1.2))  # 根据给定阈值来预测预期寿命
    print("基于给定寿命为1000的退化数据--------------------------:")
    print(df.predict_degradation(1000))  # 根据给定寿命估计退化数据

    dm = degradation_model.Degradation(data)
    popt, model_func = dm.model_fit(model=3)  # 使用对数模型拟合来得到模型参数和函数方程
    print("对数模型的R方值和均方误差--------------------------:")
    print(dm.model_evaluation(popt, model_func))  # 得到对数模型的R方值和均方误差
    dm.save_model(popt, model_func, "logarithmic_degradation_model")  # 存储模型

    df = degradation_functions.Degradation_functions(
        "logarithmic_degradation_model")  # 创建退化模型方法对象
    print("基于阈值为1.2的预期寿命（对数模型）--------------------------:")
    print(df.predict_lifetimes(1.2))  # 根据给定阈值来预测预期寿命
    print("基于给定寿命为1000的退化数据（对数模型）--------------------------:")
    print(df.predict_degradation(1000))  # 根据给定寿命估计退化数据
