import numpy as np
import statsmodels.api as sm
np.seterr(divide='ignore', invalid='ignore')
def efp(data, mtype="OLS-MOSUM", h=0.15):
    # formula, data = list(),rescale = TRUE
    X = np.arange(0, len(data))
    X = sm.add_constant(X)
    Y = data.values
    n = len(Y)
    h = 0.15
    retval = {}

    model = sm.OLS(Y, X).fit()
    e = model.resid
    sigma = np.sqrt(sum(pow(e, 2)) / model.df_resid)
    nh = int(np.floor(n * h))
    process = np.insert(e, 0, 0, axis=None).cumsum()
    process = process[nh:] - process[:-nh]
    process = process / (sigma * np.sqrt(n))
    start = np.floor(0.5 + nh / 2)
    X_process = np.arange(start, start + len(process))

    retval["nreg"] = X.shape[1]
    retval["par"] = h
    retval["type"] = "OLS-based MOSUM test"
    retval["lim_process"] = "Brownian bridge increments"
    retval["process"] = process
    retval["datatsp"] = [1, n, "m"]

    m_fit = sm.OLS(Y, X).fit()
    retval["coefficients"] = m_fit.params
    retval["sigma"] = np.sqrt(sum(pow(m_fit.resid, 2)) / m_fit.df_resid)
    return retval


def sctest(Vt,h, alt_boundary=False, functional="max"):
    '''

    Parameters
    ----------
    Vt
    h
    alt_boundary
    functional

    Returns
    -------
    dict{
    statistic：
    p: p value
    }
    '''
    process_ret=efp(Vt, h=h)
    x = process_ret["process"]
    h = process_ret["par"]
    k = process_ret["nreg"] - 1
    STAT = np.max(np.abs(x))
    if k > 6:
        pass
    crit_table = np.array([[0.7552, 0.8017, 0.8444, 0.8977],
                           [0.9809, 1.0483, 1.1119, 1.1888],
                           [1.1211, 1.2059, 1.2845, 1.3767],
                           [1.2170, 1.3158, 1.4053, 1.5131],
                           [1.2811, 1.3920, 1.4917, 1.6118],
                           [1.3258, 1.4448, 1.5548, 1.6863],
                           [1.3514, 1.4789, 1.5946, 1.7339],
                           [1.3628, 1.4956, 1.6152, 1.7572],
                           [1.3610, 1.4976, 1.6210, 1.7676],
                           [1.3751, 1.5115, 1.6341, 1.7808]])
    tablen = 4
    tableh = np.arange(1, 11) * 0.05
    tablep = [1, 0.1, 0.05, 0.025, 0.01]
    tableipl = [0, 0, 0, 0, 0]
    for i in range(1, 1 + tablen):
        tableipl[i] = np.interp(h, tableh, crit_table[:, i - 1])
    pval = np.interp(STAT, tableipl, tablep)

    return {"statistic": STAT, "p": pval}
