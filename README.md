GreenBrown





### Introduction
This code is a python reimplementation of R package GreenBrown.
Detailed API/tutorial please see Example.ipynb

The package has been developed at the Max Planck Institute for Biogeochemistry, Jena, Germany in order to distribute the code of the following publications:
-   Forkel, M., Carvalhais, N., Verbesselt, J., Mahecha, M., Neigh, C., Reichstein, M., 2013.  [Trend Change Detection in NDVI Time Series: Effects of Inter-Annual Variability and Methodology.](http://www.mdpi.com/2072-4292/5/5/2113)  Remote Sensing 5, 2113–2144. doi:10.3390/rs5052113
-   Forkel, M., Migliavacca, M., Thonicke, K., Reichstein, M., Schaphoff, S., Weber, U., Carvalhais, N., 2015.  [Codominant water control on global interannual variability and trends in land surface phenology and greenness.](http://onlinelibrary.wiley.com/doi/10.1111/gcb.12950/abstract)  Glob Change Biol 21, 3414–3435. doi:10.1111/gcb.12950

Example:

### Spatial Statistics and Trend Analysis
#### 1.Load Example Data

    import numpy as np
    import matplotlib.pyplot as plt
    from greenbrown.utils import *
    import greenbrown as gb
    Yt = load_example()
    Tt = Yt.index

#### 2. Cell-based Trend Analysis 

    trd3=gb.AnomalyDetection(Yt,Tt)
    trd3.summary()
    trd3.plot()
    ATs,Am,An,Sts,St_bp,Rem=trd3.values

#### 3. Spatial Statistics 
	

    tif = load_tif_sample()[20:24,20:24,:]
    tif_index = pd.date_range('2000', periods=16, freq='Y')
    tif = interpolate_3d(tif)
    Ss = gb.Spatial_Statistcs(tif,tif_index)
    Ss.excute()
    Ss.plot()
    Ss.summary()




### Methods
#### 1. BFAST Season Decomposition by OLS-MoSUM

    bst=gb.bfast.BFAST(Yt,Tt,max_iter=1)
    bst.plot()
    bst.summary()
    yt,tt,st,rt=bst.composition

#### 2. Annual Trend and anomaly Detection
    trd=gb.TrendAAT(Yt,Tt,0.15,0.05)
    trd.summary()
    trd.plot()
    ATs,Am,An=trd.values


#### 3. Seasonal component extraction

    trd2=gb.TrendSST(Yt,Tt)
    trd2.summary()
    trd2.plot()
    STs,St_bp=trd2.values
