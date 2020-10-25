import pandas as pd
import numpy as np
from scipy.interpolate import griddata

def aic(n, breakpoints, k, rss):
    #return float aic
    if rss==0:
        return np.nan
    df = (k + 1) * (len([i for i in breakpoints if not np.isnan(i)])) + 1
    logL = -0.5 * n * (np.log(rss) + 1 - np.log(n) + np.log(2 * np.pi))
    return -2 * logL + np.log(n) * df

def recresid(X, y, span=None):
    nobs, nvars = X.shape
    if span is None:
        span = nvars

    recresid = np.nan * np.zeros((nobs))
    recvar = np.nan * np.zeros((nobs))

    X0 = X[:span]
    y0 = y[:span]

    # Initial fit
    XTX_j = np.linalg.inv(np.dot(X0.T, X0))
    XTY = np.dot(X0.T, y0)
    beta = np.dot(XTX_j, XTY)

    yhat_j = np.dot(X[span - 1], beta)
    recresid[span - 1] = y[span - 1] - yhat_j
    recvar[span - 1] = 1 + np.dot(X[span - 1],
                                  np.dot(XTX_j, X[span - 1]))
    for j in range(span, nobs):
        x_j = X[j:j + 1, :]
        y_j = y[j]

        # Prediction with previous beta
        yhat_j = np.dot(x_j, beta)
        resid_j = y_j - yhat_j

        # Update
        XTXx_j = np.dot(XTX_j, x_j.T)
        f_t = 1 + np.dot(x_j, XTXx_j)
        XTX_j = XTX_j - np.dot(XTXx_j, XTXx_j.T) / f_t  # eqn 5.5.15

        beta = beta + (XTXx_j * resid_j / f_t).ravel()  # eqn 5.5.14

        recresid[j] = resid_j
        recvar[j] = f_t

    return recresid / np.sqrt(recvar)


def load_example():
    '''

    Returns
    -------
    NDVI_SERIES float32 with time index
    '''
    # ndvi = "\\GreenBrown\\data\\time_series.npy"
    # ndvi_date = "\\GreenBrown\\data\\time_series_date.txt"
    save = "\\GreenBrown\\data\\time_series.csv"

    ndvi_d = pd.read_csv(save)
    ndvi_date = pd.to_datetime(ndvi_d["time"], format="%Y-%m-%d")
    value = ndvi_d.values[:, 1].astype("float32")
    ndvi_s = pd.Series(value, index=ndvi_date)
    ndvi_s = (ndvi_s - ndvi_s.min()) / (ndvi_s.max() - ndvi_s.min())
    return ndvi_s


def load_tif_sample():

    path="\\GreenBrown\\data\\clip.csv" #20,14
    ndvi = pd.read_csv(path,header=0)
    nd=ndvi.iloc[:,:-4].values

    row=nd[:,0]
    col=nd[:,1]
    rstart=np.min(nd[:,0])
    cstart=np.min(nd[:,1])

    rr=row.max()-row.min()
    cr=col.max()-col.min()

    rn=40
    cn=int((rn*(rr)/(cr)))

    rd=rr/rn
    cd=rr/cn

    dim=len(nd[0,2:])
    rst=np.zeros((rn+1,cn+1,dim))

    def convert(x,y):
        r=int((x-rstart)/rd)
        c=int((y-cstart)/cd)
        return r,c

    for i in range(len(nd)):   #
        nd[i][0],nd[i][1]=convert(nd[i][0],nd[i][1])
        if nd[i][1]<cn and nd[i][0]<rn:
            rst[int(nd[i][0])][int(nd[i][1])]=nd[i,2:]
    rst[rst<=0]=np.nan
    rst[rst>1]=np.nan
    rst=rst[:,:,:16]
    
    return rst


def interpolate_3d(array):
    
    for i in range(array.shape[2]):
        array[:,:,i]=interpolate_2d(array[:,:,i])
    return array

def interpolate_2d(array):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method='cubic',fill_value=0)
    return GD1
