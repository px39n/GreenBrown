import statsmodels.api as sm
import numpy as np
from greenbrown.utils import aic

def summary(rval,breaks=None):
    # return  bic list
    n=rval['n']
    k=rval["k"]
    rss_triang=rval["rss_triang"]
    if not breaks:
        breaks=int(rval["rss_table"].shape[1]/2)

    rss = np.array([rssf(1, n,rss_triang)] + [np.nan for i in range(breaks)])
    bic = [n * (np.log(rss[0]) + 1 - np.log(n) + np.log(2 * np.pi))
           + np.log(n) * (k + 1)] + [np.nan for i in range(breaks)]
    bp = breakpoints_rval(rval, breaks)
    bic[breaks]=aic(n,bp["breakpoints"],k,bp["rss"])
    if breaks>1:
        for m in np.arange(breaks-1,0,-1):
            bpm=breakpoints_rval(rval,m)
            bic[int(m)]=aic(n,bpm["breakpoints"],k,bpm["rss"])
        # bic=[x if x!=-np.inf else np.nan for x in bic]
        # print("ok")
    return bic


def breakpoints_rval(rval,breaks=None,mode="brute"):
    '''

    Parameters
    ----------
    rval
    breaks

    Returns
    -------
    breakpoints, RSS, nobs, nreg(k)
    '''

    rss_triang=rval["rss_triang"]
    rss_table=rval["rss_table"]
    n=rval["n"]
    h=rval["h"]
    if not breaks:
        sbp=summary(rval)
        min = np.nanmin(sbp)
        breaks = None if np.isnan(min) else sbp.index(min)
    if breaks<1:
        rval["breakpoints"]=None
        rval["rss"]=rssf(1,n,rss_triang)
    else:
        rss_tab=extend_rss_table(rss_table,breaks,rss_triang,n,h)

        rval["breakpoints"]=extract_breaks(rss_tab,breaks,n,rss_triang)
        bp=np.array(rval["breakpoints"])
        temp=np.stack((bp[:-1]+1,bp[1:]),axis=1)
        rss=np.nansum([rssf(int(x[1]),int(x[0]),rss_triang) for x in temp])
        rval["rss"]=rss
        return rval


def extracted_by_mosum(V,h=0.15,breaks=[]):

    y = V.values
    X = np.arange(1, len(V)+1)
    X = sm.add_constant(X)
    n, k = X.shape
    intercept_only = False
    if h <= 1:
        h = int(np.floor(n * h))
    if h <= k:
        raise ("minimum segment size must be greater than the number of regressors")
    if h >= np.floor(n / 2):
        raise ("minimum segment size must be smaller than half the number of observations")
    if not breaks:
        breaks = int(np.ceil(n / h) - 2)
    elif breaks < 1:
        breaks = 1
        print("warning: number of break must be at least 1")
    elif (breaks > np.ceil(n / h) - 2):
        breaks =np.ceil (n / h) - 2
        print("warning number of breaks too large")
    # ssri(5,n,X,y)
    rss_triang = rss_triangf(n, X, y, k,h)
    # rss_table=cbind(index,break_rss)
    index = np.arange(h, (n - h + 1))
    break_rss = break_rssf(index, rss_triang)
    rss_table = np.stack((index, break_rss), axis=1)
    rss_table1=extend_rss_table(rss_table,breaks,rss_triang,n,h)
    opt=extract_breaks(rss_table1,breaks,n,rss_triang)
    rval={"breakpoints":opt,"rss_table":rss_table1,"rss_triang":rss_triang,"n":n,"k":k,"h":h}
    breakpoints= breakpoints_rval(rval)
    return rval

def extract_breaks(rss_table, breaks, n, rss_triang):
    if breaks * 2 > rss_table.shape[1]:
        raise ("compute RSS_with enough break before")
    index = rss_table[:, 0]
    offset = int(rss_table[0, 0])
    break_rss = []
    for i in index:
        br = rss_table[int(i - offset), breaks * 2 - 1] + rssf(int(i + 1), n, rss_triang)
        br = br if not np.isnan(br) else 5000
        break_rss.append(br)
    # print(np.isnan(min(break_rss)))
    opt = [break_rss.index(min(break_rss))]

    if breaks > 1:
        for i in range( breaks * 2-1,2,-2):
            opt.append(rss_table[int(opt[-1]), i - 1])
    return opt




def extend_rss_table(rss_table,breaks,rss_triang,n,h):
    offset=int(rss_table[0,0])
    if ((breaks * 2) > rss_table.shape[1]):
        for m in range(int(rss_table.shape[1]/2+1),breaks+1):
            my_index=np.arange(m*h,n-h+1)
            my_rss_table=rss_table[:,(m-1)*2-2:(m-1)*2]
            my_rss_table=np.concatenate((my_rss_table,np.full((len(rss_table), 2), np.nan)),axis=1)
            #print(my_index)
            for i in my_index:
                pot_index=range((m-1)*h,i-h+1)
                break_rss=[my_rss_table[j-1-offset,1]+rssf(j,i-1,rss_triang) for j in pot_index]
                min=np.nanmin(break_rss)
                opt=0 if np.isnan(min) else break_rss.index(min)
                my_rss_table[i-offset,2:4]=[pot_index[opt],break_rss[opt]]

            rss_table=np.concatenate((rss_table,my_rss_table[:,2:4]),axis=1)
    return rss_table


def rssf(i,j,rss_triang):
    #accept  i, j with R index
    return rss_triang[i-1][j-i]

def break_rssf(index,rss_triang):
    ret=[]
    for i in index:
        ret.append(rssf(1,i,rss_triang))
    return ret

def rss_triangf(n,x,y,k,h):
    ret=[]
    for i in range(1,int(n-h+2)):
        ret.append(ssri(i,n=n,x=x,y=y,k=k))
    return ret

def ssri(i,n,x,y,k):
    # ret=recresid(x[i-1:],y[i-1:])
    mod = sm.RecursiveLS(x[i:], y[i:])
    ret = mod.fit().cusum_squares
    return np.concatenate((np.array([np.nan] *k),ret))
