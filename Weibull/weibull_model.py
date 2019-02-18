import numpy as np
import pandas as pd
import pickle


class ParameterError(Exception):
    def __init__(self, *args):
        default_str = 'Values for "shape" and "scale" not found; Run the "fit" method or assign values explicitly.'
        super().__init__(default_str, *args)


class Weibull:
    """ Weibull distribution"""

    def __init__(self, x, loccation=None, iters=None, eps=None, shape=None,
                 scale=None):
        """
        :param x: 1d-ndarray of samples from an (unknown) distribution. Each value must satisfy x > 0.
        :param loccation: Initial location of weibull distribution or failure-free life
        :param iters: Maximum number of iterations
        :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
        :param shape: Shape parameter
        :param scale: Scale parameter
        """
        if isinstance(x, pd.Series):
            data = x.values
        elif isinstance(x, np.ndarray):
            data = x
        else:
            raise TypeError("数据格式必须为DataFrame或者ndarray")

        if data.ndim != 1:
            raise TypeError("数据维度必须为一维")
        if data.min() < 0:
            raise ValueError('数据不能包含负数')
        if data.size < 10:
            raise ValueError('数据规模不得小于10')
        self.x = data
        if loccation is None or loccation < 0:
            self.location = 0
        else:
            self.location = loccation
        if iters is None or iters < 0:
            self.iters = 100000
        else:
            self.iters = iters
        if eps is None or eps < 0:
            self.eps = 1e-6
        else:
            self.eps = eps
        self.shape, self.scale = shape, scale

    def fit(self):
        """
        Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
        :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
            Impossible fits may be due to 0-values in x.
        """
        # fit k via MLE
        ln_x = np.log(self.x - self.location)
        k = 1.
        k_t_1 = k

        for t in range(self.iters):
            x_k = (self.x - self.location) ** k
            x_k_ln_x = x_k * ln_x
            ff = np.sum(x_k_ln_x)
            fg = np.sum(x_k)
            f = ff / fg - np.mean(ln_x) - (1. / k)

            # Calculate second derivative d^2f/dk^2
            ff_prime = np.sum(x_k_ln_x * ln_x)
            fg_prime = ff
            f_prime = (ff_prime / fg - (ff / fg * fg_prime / fg)) + (
                    1. / (k * k))

            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            k -= f / f_prime

            if np.isnan(f):
                return np.nan, np.nan
            if abs(k - k_t_1) < self.eps:
                break

            k_t_1 = k

        lam = np.mean((self.x - self.location) ** k) ** (1.0 / k)

        self.shape = k
        self.scale = lam

        return k, lam

    def save_model(self, path: str):
        if not self.shape or not self.scale:
            raise ParameterError
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        # self.plot_model()

    # def plot_model(self):
    #     if not self.shape or not self.scale:
    #         raise ParameterError
    #     self.x.sort()
    #     ydata = stats.weibull_min.pdf(np.linspace(0, self.x.max()),
    #                                   self.shape, self.location,
    #                                   self.scale)
    #     plt.plot(np.linspace(0, self.x.max()), ydata, '-')
    #     plt.hist(self.x, bins=np.linspace(0, self.x.max()), normed=True,
    #              alpha=0.5)
    #     plt.savefig("weibull_fig.png")
