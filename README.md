
# GreenBrown  

  ### Introduction  
This code is a python reimplementation of R package GreenBrown.  Detailed API/tutorial please see **[Example.ipynb](https://github.com/px39n/GreenBrown/blob/main/example.ipynb)**  . It contains a range of R packages and methods reimplement like GeenBrown bfast, Strucchange.

For now, It could be used to
1. Detect Annual, Seasonal, and Short Structure change and Breakpoints based on OLS-SUM/differential method (or Differential or Recursive)
2. Execute trend test like MK. Sctest
3. Seasonal Decomposition: 1.Yt=Tt+An+Tm+St+rem 2. Yt=Tt+St+Res
4. Preprocess and Spatial Statistics according to Annual Data.

In the Future
1.  Finished some works in library
2. More functions, More Trend Tests
3. Design for Anomaly Classification
  
The package has been developed at the Max Planck Institute for Biogeochemistry, Jena, Germany in order to distribute the code of the following publications:  
- Verbesselt, J., Hyndman, R., Newnham, G., & Culvenor, D. (2010). Detecting trend and seasonal changes in satellite image time series. Remote Sensing of Environment, 114, 106-115. DOI: 10.1016/j.rse.2009.08.014. DownLoad Paper
- Verbesselt, J., Hyndman, R., Zeileis, A., & Culvenor, D. (2010). Phenological change detection while accounting for abrupt and gradual trends in satellite image time series. Remote Sensing of Environment, 114, 2970-2980. DOI: 10.1016/j.rse.2010.08.003. DownLoad Paper
- Verbesselt, J., Zeileis, A., & Herold, M. (2013). Near real-time disturbance detection using satellite image time series, Remote Sensing of Environment. DOI: 10.1016/j.rse.2012.02.022. DownLoad Paper
- Forkel, M., Carvalhais, N., Verbesselt, J., Mahecha, M., Neigh, C., Reichstein, M., 2013.  [Trend Change Detection in NDVI Time Series: Effects of Inter-Annual Variability and Methodology.](http://www.mdpi.com/2072-4292/5/5/2113)  Remote Sensing 5, 2113–2144. doi:10.3390/rs5052113  
- Forkel, M., Migliavacca, M., Thonicke, K., Reichstein, M., Schaphoff, S., Weber, U., Carvalhais, N., 2015.  [Codominant water control on global interannual variability and trends in land surface phenology and greenness.](http://onlinelibrary.wiley.com/doi/10.1111/gcb.12950/abstract)  Glob Change Biol 21, 3414–3435. doi:10.1111/gcb.12950  
   
  
### Spatial Statistics and Trend Analysis  
#### 1.Load Example Data  
  

     import numpy as np 
     import matplotlib.pyplot as plt 
     from greenbrown.utils import * 
     import greenbrown as gb 
     Yt = load_example() Tt = Yt.index  

#### 2. Cell-based Trend Analysis   
     trd3=gb.AnomalyDetection(Yt,Tt) 
     trd3.summary() 
     trd3.plot() 
     ATs,Am,An,Sts,St_bp,Rem=trd3.values  
      
  <img src="https://github.com/px39n/GreenBrown/blob/main/data/table.JPG?raw=true" width="400"/> 
   <img src="https://github.com/px39n/GreenBrown/blob/main/data/ano.JPG?raw=true" width="400"/>

 
#### 3. Spatial Statistics   
   

     tif = load_tif_sample()[20:24,20:24,:] 
     tif_index = pd.date_range('2000', periods=16, freq='Y') 
     tif = interpolate_3d(tif) 
     Ss = gb.Spatial_Statistcs(tif,tif_index) 
     Ss.excute() 
     Ss.plot() 
     # Ss.summary()  
      
      

  <img src="https://github.com/px39n/GreenBrown/blob/main/data/timespace.JPG?raw=true" width="400"/>
  
### Other Methods 
#### 1. BFAST Season Decomposition by OLS-MoSUM  
     bst=gb.bfast.BFAST(Yt,Tt,max_iter=1) 
     bst.plot() 
     bst.summary() 
     yt,tt,st,rt=bst.composition  


Output: See Example.juynb
#### 2. Annual Trend and anomaly Detection  
     trd=gb.TrendAAT(Yt,Tt,0.15,0.05) 
     trd.summary() 
     trd.plot() 
     ATs,Am,An=trd.values  
  
  Output: See Example.juynb
  
#### 3. Seasonal component extraction  
      
     trd2=gb.TrendSST(Yt,Tt) 
     trd2.summary() 
     trd2.plot() 
     STs,St_bp=trd2.values

Output: See Example.juynb