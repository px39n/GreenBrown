import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from .mosum import extracted_by_mosum
from .differential import PiecewiseLinFit
def extract_bps(yt,h,mode="differential",season=None):
    '''

    Parameters
    ----------
    yt
    h
    mode

    Returns
    -------
    return dict
    "breakpoints"

    '''
    x = np.arange(1, len(yt) + 1)

    if mode=="differential":
        num = len(extracted_by_mosum(yt, h)["breakpoints"])
        my_pwlf = PiecewiseLinFit(x, yt.values)
        res = my_pwlf.fit(num+1)
        ret= {"prediction":my_pwlf.predict(x),"breakpoints":res}

        return ret
    elif mode=="mosum":
        ret=extracted_by_mosum(yt,h)
        bp=ret["breakpoints"]
        boundary=[np.nanmin(x),np.nanmax(x)]
        if np.isnan(bp).all():
            bp=boundary
        else:
            bp=boundary+bp
        ret["prediction"]=pw_linear_predict(bp,x,yt)
        return ret
    elif mode == "None":
        boundary = [np.nanmin(x), np.nanmax(x)]
        ret = {"breakpoints":boundary}
        bp = boundary
        ret["prediction"] = pw_linear_predict(bp, x, yt)
    elif mode=="season":
        boundary = [0,len(x)-1]
        num=2    # test number
        my_pwlf = PiecewiseLinFit(x, yt.values)
        res=[int(i-1) for i in my_pwlf.fit(num+1)]
        x = season
        if num<1:
            ret = {"breakpoints": boundary}
            ret["prediction"] = pw_linear_predict(boundary, x, yt)
        else:
            ret = {"breakpoints": res}
            ret["prediction"]= pw_linear_predict(res, x, yt)
        ret["breakpoints"]=bp_val(res, 0.1, len(yt))

    return ret


def pw_linear_predict(bp,x,y):
    if x.ndim==1:
        my_pwlf = PiecewiseLinFit(x, y)
        my_pwlf.fit_with_breaks(bp)
        return my_pwlf.predict(x)
    if x.ndim>1:
        Y=np.zeros(len(y))
        for i in range(len(bp)-1):
            # i pieecewise of bp-1 segementations
            xi=x[bp[i]:bp[i+1]+1]
            yi=y[bp[i]:bp[i+1]+1]
            model = sm.OLS(yi, xi).fit()
            for j in range(len(xi)-1):
                # jst of len(xi) points in i piecewise
                Y[j+bp[i]]=model.predict(xi[j])
        return Y

def bp_val(bp,bp_resolution,n):
    br=bp_resolution
    br=n*br
    nbp=[]
    if len(bp)<=3:
        return bp
    else:
        i=1
        while i<len(bp)-2:
            if abs(bp[i]-bp[i+1])>=br:
                nbp.append(bp[i])
            else:
                nbp.append(int(np.mean((bp[i],bp[i+1]))))
                i=i+1
            i=i+1
    return [bp[0]]+nbp+[bp[-1]]



