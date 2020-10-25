import numpy as np
from scipy.stats import norm
# Original Sens Estimator
def __preprocessing(x):
    x = np.asarray(x)
    dim = x.ndim

    if dim == 1:
        c = 1

    elif dim == 2:
        (n, c) = x.shape

        if c == 1:
            dim = 1
            x = x.flatten()

    else:
        print('Please check your dataset.')

    return x, c
def __sens_estimator(x):
    idx = 0
    n = len(x)
    d = np.ones(int(n * (n - 1) / 2))

    for i in range(n - 1):
        j = np.arange(i + 1, n)
        d[idx: idx + len(j)] = (x[j] - x[i]) / (j - i)
        idx = idx + len(j)

    return d


def sens_slope(x):
    """
    Examples
    --------
      >>> x = np.random.rand(120)
      >>> slope,intercept = sens_slope(x)
    """

    x, c = __preprocessing(x)
    #     x, n = __missing_values_analysis(x, method = 'skip')
    n = len(x)
    slope = np.nanmedian(__sens_estimator(x))
    intercept = np.nanmedian(x) - np.median(
        np.arange(n)[~np.isnan(x.flatten())]) * slope  # or median(x) - (n-1)/2 *slope

    return [slope,intercept]



def __variance_s(x, n):
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18

    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        demo = np.ones(n)

        for i in range(g):
            tp[i] = np.sum(demo[x == unique_x[i]])

        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    return var_s
def mk_test(x, alpha=0.05):
    """
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (0.05 default)
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p-value of the significance test
        z: normalized test statistics
        Tau: Kendall Tau
        s: Mann-Kendal's score
        var_s: Variance S
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05)
    """
    n = len(x)

    # calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x, tp = np.unique(x, return_counts=True)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else:  # there are some ties in data
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    else: # s == 0:
        z = 0

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    tau=s/(.5*n*(n-1))

    return [trend, h, p, z,s,tau,__variance_s(x,n),sens_slope(x)[0],sens_slope(x)[1]]

