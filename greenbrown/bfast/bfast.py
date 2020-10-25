import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from greenbrown import bfast as bf
import matplotlib.pyplot as plt
from  greenbrown.trendtest.sctest import sctest
from prettytable import PrettyTable
from greenbrown.trendtest import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BFAST(object):
    def __init__(self, Yt, Tt,max_iter=3):

        self.Yt = Yt
        self.Tt = Tt
        self.flag = None
        self.max_iter=max_iter

    def fit(self):
        self.flag = 1
        self.ret = bfast_detection(self.Yt, self.Tt,max_iter=self.max_iter)

    def plot(self):
        if not self.flag:
            self.fit()
        ret=self.ret
        Yt=self.Yt
        ti=self.Tt
        St, Tt, bp_St, bp_Tt, Rt = ret["St"], ret["Tt"], ret["bp_St"], ret["bp_Tt"], ret["Rt"]
        plt.figure(figsize=(8, 8))
        plt.subplot(411)
        plt.title('Yt')
        plt.plot(Yt)
        plt.axhline(np.mean(Yt), color="red")
        plt.subplot(412)
        for i in bp_Tt[1:-1]:
            plt.axvline(ti[i])
        plt.title('Tt')
        plt.plot(Tt)
        plt.subplot(413)
        plt.title('St')
        plt.plot(St)
        for i in bp_St[1:-1]:
            plt.axvline(ti[i])
        plt.subplot(414)
        plt.title('Rt')
        plt.plot(Rt)
        plt.show()


    @property
    def composition(self):
        ret=self.ret
        Yt,St, Tt, Rt = self.Yt, ret["St"], ret["Tt"], ret["Rt"]

        return  np.stack((Yt,Tt,St,Rt),axis=0)

    def summary(self):
        print("Warning: This Func is unfinished")
        print("--- Trend ---------------------------------------------------------\n"
              "CONFIDENCE INTERVALS FOR BREAKPOINTS OF OPTIMAL x SEGMENT PARTITION\n"
              "-------------------------------------------------------------------\n")
        table = PrettyTable(['2.5%', '~bp~',"97.5%"])
        table.add_row(['XX', "XX",  "XX"])
        table.add_row(['XX', "XX", "XX"])
        table.add_row(['XX', "XX", "XX"])
        table.add_row(['XX', "XX", "XX"])
        print(table)
        print("--- TimeSt ---------------------------------------------------------\n")
        table = PrettyTable(['2.5%', '~bp~', "97.5%"])
        table.add_row(['2010', "2011", "2012"])
        table.add_row(['2010', "2011", "2012"])
        table.add_row(['2010', "2011", "2012"])
        table.add_row(['2010', "2011", "2012"])
        table.add_row(['2010', "2011", "2012"])
        table.add_row(['2010', "2011", "2012"])
        print(table)


def bfast_detection(Yt, ti, h=0.15, season="harmonic", max_iter=1, breaks=None, level=0.05, reg="lm", frequency=12):
    if 1:
        level = [level] * 2
        output = []
        Tt = 0
        Wt = np.zeros(shape=(len(Yt)))
        if season == "harmonic":
            tl = np.arange(1, len(Yt) + 1)
            co = np.cos(2 * np.pi * tl / frequency)
            si = np.sin(2 * np.pi * tl / frequency)
            co2 = np.cos(2 * np.pi * tl / frequency * 2)
            si2 = np.sin(2 * np.pi * tl / frequency * 2)
            co3 = np.cos(2 * np.pi * tl / frequency * 3)
            si3 = np.sin(2 * np.pi * tl / frequency * 3)
            smod = np.stack((co, si, co2, si2, co3, si3), axis=1)
            St = seasonal_decompose(Yt).seasonal

        bp_Tt = 0
        bp_St = 0
        CheckTimeTt = 1
        CheckTimeSt = 1
        i = 0
        while (not np.nansum(CheckTimeTt == np.nansum(bp_Tt)) or not np.nansum(CheckTimeSt) == np.nansum(
                bp_St)) and i < max_iter:
            CheckTimeTt = bp_Tt
            CheckTimeSt = bp_St
            ### Change in trend component
            Vt = Yt - St  # Deasonalized Time series
            p_Vt = sctest(Vt, h)
            if (p_Vt["p"] <= level[0]):
                rav_Vt = bf.extract_bps(Vt, h, mode="differential")
            else:
                rav_Vt = bf.extract_bps(Vt, h, mode="None")
            Tt = rav_Vt["prediction"]

            St = Yt - Tt
            p_wt = sctest(St, h)
            if p_wt['p'] <= level[1]:
                rav_Wt = bf.extract_bps(St, h, mode="season", season=smod)
            else:
                nobp_Wt = True
                rav_Wt = None
                rav_Wt = bf.extract_bps(St, h, mode="season", season=smod)
            St = rav_Wt["prediction"]
            bp_Tt = rav_Vt["breakpoints"]
            bp_St = rav_Wt["breakpoints"]
            i = i + 1

    Rt = Yt - St - Tt
    St=pd.Series(St)
    Tt=pd.Series(Tt)
    St.index=Yt.index
    Tt.index=Yt.index

    bp_Tt = np.array([np.floor(i) for i in bp_Tt]).astype(int) - 1
    bp_St = np.array([np.floor(i) for i in bp_St]).astype(int) - 1
    return {"St": St, "Tt": Tt, "Rt": Rt, "bp_Tt": bp_Tt, "bp_St": bp_St, "Yt": Yt}

