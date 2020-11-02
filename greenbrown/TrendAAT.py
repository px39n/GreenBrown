from prettytable import PrettyTable
from greenbrown.trendtest import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class TrendAAT(object):
    def __init__(self,Yt,Tt,h=0.15,alpha=0.05):

        self.ndvi=Yt
        self.Yt=Yt
        self.Tt=Tt
        self.mean=np.mean(self.Yt)
        self.Yt=self.Yt-self.mean
        self.h=h
        self.alpha=alpha
        self.flag=None
    def fit(self):
        self.flag=1
        ts_m = pd.Series(self.Yt, index=self.Tt)
        ts_a = ts_m.resample("Y").sum()
        self.Ats =ts_a  # used for sctest
        ts_y = ts_m.resample("M").asfreq().reindex(self.Tt)
        t_a = np.arange(1, len(ts_y) + 1)
        t_a = sm.add_constant(t_a)
        m_fit = sm.OLS(ts_y, t_a, missing='drop').fit()
        ts_y = m_fit.predict(t_a)
        ts_y=pd.Series(ts_y,index=self.Tt)
        anomaly = self.Ats  - ts_y.resample("Y").sum()
        self.anomaly = anomaly.resample("M").asfreq().reindex(self.Tt).fillna(method="backfill")
        self.ts_y=ts_y

        self.mk = self.mktest()
        self.sc = self.sctest()

    def plot(self):
        if not self.flag:
            self.fit()
        plt.figure(figsize=(6,6))
        plt.title("Annual Aggregated Time Series")
        plt.subplot(311)
        plt.plot(self.ts_y+self.mean,label='Y={} * Ti + {} (In Yrs with offset)'.format(str(int(self.mk[7])),str(int(self.mk[8]))))
        plt.plot(self.ndvi,label='NDVI')
        plt.axhline(self.mean,label="Mean of NDVI")
        plt.legend(loc="lower right")
        plt.subplot(312)
        plt.plot(self.ts_y,label='Y={} * Ti + {} (In Yrs with offset)'.format(str(int(self.mk[7])), str(int(self.mk[8]))))
        plt.legend(loc="lower right")
        plt.subplot(313)
        plt.plot(self.anomaly,label='Annual Anomalies')
        plt.legend(loc="lower right")
        plt.show()

    def mktest(self):
        return mk_test(self.Ats, self.alpha)


    def sctest(self):
        return sctest(self.Ats,self.h)

    @property
    def values(self):
        if not self.flag:
            self.fit()
        return self.ts_y,self.mean,self.anomaly

    def summary(self):
        if not self.flag:
            self.fit()
        print("--- Trend ---------------------------------------------------------\n"
              "Calculate Annual Aggravated trends and trend changes on time series\n"
              "-------------------------------------------------------------------\n")
        print("Table1. Time Series Information:")
        table = PrettyTable(['Name', 'Time'])
        table.add_row(['Time series start:',str(self.Tt[0])])
        table.add_row(['Time series end:',str(self.Tt[-1])])
        table.add_row(['Time series length:',str(len(self.Yt)) ])
        print(table)
        print("Table2. Test for Structural change\n")
        table = PrettyTable(['Name', 'Value'])
        table.add_row(['Method:', "OLS-based MOSUM test"])
        table.add_row(['Statistic:', self.sc["statistic"]])
        table.add_row(['P-value:',self.sc["p"]])
        table.add_row(['Breakpoints:',"Breakpoints were not detected" ])
        print(table)
        print("Table3. Trend in segments of the time series\n")
        table = PrettyTable(['Name', 'Value'])
        table.add_row(['Method:', "MK Test"])
        table.add_row(['Trend:', str(self.mk[0])])
        table.add_row(['P-value:', str(self.mk[2])])
        table.add_row(['Normalized test statistics:', str(self.mk[3])])
        table.add_row(['MK\'s score:', str(self.mk[4])])
        table.add_row(['Kendall Tau:', str(self.mk[5])])
        table.add_row(['Variance s:', str(self.mk[6])])
        table.add_row(['Slope:', str(self.mk[7])])
        table.add_row(['Intercept:', str(self.mk[8])])
        print(table)
