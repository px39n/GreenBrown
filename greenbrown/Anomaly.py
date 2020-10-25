import greenbrown as gb
import matplotlib.pyplot as plt
import numpy as np
class AnomalyDetection(object):
    def __init__(self,Yt,Tt,h=0.15,alpha=0.05,max_iter=3):
        self.Yt=Yt
        self.Tt=Tt
        self.flag = 0
    def fit(self):
        self.flag=1
        trd = gb.TrendAAT(self.Yt, self.Tt, 0.15, 0.05)
        self.ATs, self.Am, self.An = trd.values
        trd2=gb.TrendSST(self.Yt,self.Tt,max_iter=3)
        self.STs,self.St_bp=trd2.values
        self.Rem=self.Yt-self.ATs-self.Am-self.An-self.STs

    def plot(self):
        if not self.flag:
            self.fit()
        plt.figure(figsize=(7,12))
        plt.title("Seaonal Time Series with Annual and short term Detection")
        plt.subplot(511)
        plt.plot(self.Yt,label="NDVI")
        plt.legend(loc="lower right")
        plt.axhline(self.Am,label="NDVI mean")
        plt.subplot(512)
        plt.plot(self.ATs, label="Annual TS")
        plt.axhline(np.mean(self.ATs),alpha=0.5)
        plt.legend(loc="lower right")
        plt.subplot(513)
        plt.plot(self.An, label="Annual Anomaly\nStd={}".format(self.An.std()))
        plt.axhline(np.mean(self.An),alpha=0.5)
        plt.legend(loc="lower right")
        plt.subplot(514)
        plt.plot(self.STs, label="Season Ts")
        plt.axhline(np.mean(self.STs),alpha=0.5)
        plt.legend(loc="lower right")
        plt.subplot(515)
        plt.plot(self.Rem, label="Short Reminder\nStd={}".format(self.Rem.std()))
        plt.axhline(np.mean(self.Rem),alpha=0.5)
        plt.legend(loc="lower right")
        plt.show()

    @property
    def values(self):
        if not self.flag:
            self.fit()
        return [self.ATs,self.Am,self.An,self.STs,self.St_bp,self.Rem]

    def summary(self):
        if not self.flag:
            self.fit()
        print("--- Trend ---------------------------------------------------------\n"
              "This Part is Unfinished. Will be added in next Version\n"
              "----------------------------------------------a---------------------\n")

