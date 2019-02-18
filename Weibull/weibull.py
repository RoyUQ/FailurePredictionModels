import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def fit(x, iters=1000, eps=1e-6):
    """
    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x: 1d-ndarray of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    # fit k via MLE
    ln_x = np.log(x)
    k = 1.
    k_t_1 = k

    for t in range(iters):
        x_k = x ** k
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
        if abs(k - k_t_1) < eps:
            break

        k_t_1 = k

    lam = np.mean(x ** k) ** (1.0 / k)

    return k, lam


def my_test():
    weibull = np.random.weibull(2.0, 100000)
    x = 2 * weibull
    mle_shape, mle_scale = fit(x)
    x.sort()
    print(mle_shape)
    print(mle_scale)
    # p0, p1, p2 = stats.weibull_min.fit(x, floc=0)
    # print(p0, p1, p2)
    ydata = stats.weibull_min.pdf(np.linspace(0, x.max(), 10), mle_shape, 0,
                                  mle_scale)
    plt.plot(np.linspace(0, x.max(), 10), ydata, '-')
    plt.hist(x, bins=np.linspace(0, x.max(), 10), normed=True, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    my_test()
