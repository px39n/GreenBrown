
import matplotlib.pyplot as plt

from greenbrown.utils import *

import greenbrown as gb

class Spatial_Statistcs(object):
    def __init__(self, tif,tif_index):

        self.tif = tif
        self.flag = 0
        self.tt=tif_index
        self.stats = np.zeros((tif.shape[0], tif.shape[1], 5))
        self.distr = np.zeros((5))
        self.simul = np.zeros((5))

    def excute(self):
        self.flag = 1
        for i in range(self.tif.shape[0]):
            for j in range(self.tif.shape[1]):
                Yt = self.tif[i,j,:]
                Tt = pd.date_range('2000', periods=16, freq='Y')
                Yt=pd.Series(Yt,index=Tt)
                trd3 = gb.AnomalyDetection(Yt, Tt, max_iter=1)
                ATs, Am, An, Sts, St_bp, Rem = trd3.values
                ret = [np.nanmean(ATs), np.nanmean(Am), np.nanmean(An), np.nanmean(Sts), np.nanmean(Rem)]
                self.stats[i,j ,:] = np.array((ret))
                print("The {},{}st iter of {},{}".format(i, j, self.tif.shape[0],self.tif.shape[1]))
    def summary(self):
        if self.flag:
            pass

    def plot(self):
        if self.flag:

            tlist = ["NDVI mean", "NDVI trend", "NDVI IAV", "NDVI Seaonal", "NDVI Rem"]
            plt.figure(figsize=(12, 16))
            plt.title("Spatial Statistics")
            for i in range(5):
                plt.subplot(5, 3, i * 3 + 1)
                plt.title(tlist[i])
                a = plt.imshow(self.stats[:, :, i])
                plt.colorbar(a)

            for i in range(5):
                plt.subplot(5, 3, i * 3 + 2)
                plt.title("Distribution of " + tlist[i])
                s = self.stats[:, :, i].flatten()
                s=s[s>0]
                p1 = np.percentile(s, 5)
                p2 = np.percentile(s, 50)
                p3 = np.percentile(s, 95)
                a = plt.hist(s, bins=20, histtype='step', color="black")
                plt.axvline(p1, label="low")
                plt.axvline(p2, Label="medium")
                plt.axvline(p3, Label="strong")
                plt.legend(loc="lower right")

            plt.subplot(5, 3, 3)
            plt.title("Simulated Data" + tlist[0])

            plt.axhline(np.nanmean(self.stats[:,:,0]),label="1%")


            plt.subplot(5, 3, 6)
            x = np.arange(1, len(self.tt) + 1)
            s = self.stats[:, :, 1].flatten()
            s = s[s > 0]
            p=[0,0,0]
            p[0] = np.percentile(s, 5)
            p[1] = np.percentile(s, 50)
            p[2] = np.percentile(s, 95)
            for j in range(3):
                abline_values = [p[j] * (i-int(len(self.tt))) for i in x]
                plt.plot(x, abline_values, '--',label="p{}".format(str(i)))

            allin=np.mean(self.tif,axis=(0,1))
            allin=pd.Series(allin,self.tt)
            trd3 = gb.AnomalyDetection(allin, self.tt, max_iter=1)
            ATs, Am, An, Sts, St_bp, Rem = trd3.values
            ret = [ATs, Am, An, Sts, Rem]

            for i in range(2, 5):
                plt.subplot(5, 3, i * 3 + 3)
                xxx=pd.Series(ret[i],index=self.tt)
                plt.title("Simulated Data" + tlist[i])
                plt.plot(xxx)
            plt.show()