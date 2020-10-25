from .bfast.bfast import BFAST
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from greenbrown.trendtest import *

class TrendSST(object):
    def __init__(self, Yt, Tt,max_iter=3):

        self.Yt = Yt
        self.Tt = Tt
        self.sflag = False

    def fit(self):
        self.sflag = 1
        bf = BFAST(self.Yt, self.Tt)
        bf.fit()
        self.ret=bf.ret
        self.mk = self.mktest()
        self.sc = self.sctest()

    def plot(self):
        if not self.sflag:
            self.fit()
        bp_St=self.ret["bp_St"]
        ti = self.Tt
        plt.title("Seasoned Time Series by BFAST")
        plt.plot(self.ret["St"],label='NDVI')
        for i in bp_St[1:-1]:
            plt.axvline(ti[i])
        plt.legend(loc="lower right")

        plt.show()

    def mktest(self):
        return mk_test(self.ret["St"], 0.05)


    def sctest(self):
        return sctest(self.ret["St"],0.15)

    @property
    def values(self):
        if not self.sflag:
            self.fit()
        return  [self.ret["St"],self.ret["bp_St"]]

    def summary(self):
        if not self.sflag:
            self.fit()
        print("--- Trend ---------------------------------------------------------\n"
              "Calculate Seasonal composition and trend changes on time series\n"
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
        table.add_row(['Breakpoints:',str(self.ret["bp_St"])])
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
