import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class Spectrum_data:
    def __init__(self, ch: np.ndarray, cnt: np.ndarray = np.array([]), u_cnt: np.ndarray = np.array([])):
        if cnt.size==0:
            cnt = np.zeros(ch.shape)
        if ch.size != cnt.size:
            print("Error: cnt must have same size as ch.")
            raise ValueError
        self.__ch = ch
        self.__cnt = cnt
        if u_cnt.size>0:
            if ch.size != cnt.size:
                print("Error: u_cnt must have same size as ch.")
                raise ValueError
            self.__u_cnt = u_cnt
        else:
            self.__u_cnt = np.sqrt(cnt)

    # getter
    def ch(self):
        return self.__ch

    def cnt(self):
        return self.__cnt

    def u_cnt(self):
        return self.__u_cnt

    # operator overloading
    def __mul__(self, num: float):
        new_cnt = self.__cnt*num
        new_u_cnt = self.__u_cnt*num
        rslt = Spectrum_data(self.__ch, new_cnt, new_u_cnt)
        return rslt

    def __rmul__(self, num: float):
        return self*num

    def __truediv__(self, num: float):
        return self*(1./num)

    def __pos__(self):
        return self

    def __neg__(self):
        rslt = Spectrum_data(self.__ch, -self.__cnt, self.__u_cnt)
        return rslt

    def __add__(self, second):
        if not np.array_equal(self.__ch, second.__ch):
            print("Error: self.ch must be same as second.ch.")
            raise ValueError
        new_cnt = self.__cnt + second.__cnt
        new_u_cnt = np.sqrt(self.__u_cnt*self.__u_cnt +
                            second.__u_cnt*second.__u_cnt)
        rslt = Spectrum_data(self.__ch, new_cnt, new_u_cnt)
        return rslt

    def __sub__(self, second):
        return self + (-second)

    # calculation
    def sub_data(self, bound):
        idx = np.digitize(bound, self.__ch)
        new_ch = self.__ch[idx[0]:idx[1]]
        new_cnt = self.__cnt[idx[0]:idx[1]]
        new_u_cnt = self.__u_cnt[idx[0]:idx[1]]
        rslt = Spectrum_data(new_ch, new_cnt, new_u_cnt)
        return rslt

    def translation(self, nch: int):
        if nch>0:
            new_cnt = np.r_[self.__cnt[nch:], np.zeros((nch,))]
            new_u_cnt = np.r_[self.__u_cnt[nch:], np.zeros((nch,))]
        else:
            new_cnt = np.r_[np.zeros((-nch,)), self.__cnt[:nch]]
            new_u_cnt = np.r_[np.zeros((-nch,)), self.__u_cnt[:nch]]
        rslt = Spectrum_data(self.__ch, new_cnt, new_u_cnt)
        return rslt

    def derivative(self):
        h = self.__ch[1] - self.__ch[0]
        rslt = (self.translation(1) - self.translation(-1)) / (2*h)
        return rslt

    def movmean(self, nch: int):
        if nch<=0 or nch%2==0:
            print("Error: nch must be a pos odd int.")
            raise ValueError
        rslt = Spectrum_data(self.__ch)
        for i in range(-(nch//2), (nch//2) + 1):
            rslt += self.translation(i)
        rslt /= nch
        return rslt

    def smooth_savgol(self, nch: int, order: int):
        new_cnt = signal.savgol_filter(self.__cnt, nch, order)
        new_u_cnt = np.zeros(self.__u_cnt.shape)
        rslt = Spectrum_data(self.__ch, new_cnt, new_u_cnt)
        return rslt

    # print
    def show(self):
        plt.figure(1)
        plt.errorbar(self.__ch, self.__cnt, self.__u_cnt)
        plt.show()

    def errorbar(self):
        plt.errorbar(self.__ch, self.__cnt, self.__u_cnt)


def search_peak(data: Spectrum_data, significance: float):
    data_2ndder = data.derivative().derivative().smooth_savgol(51, 3).derivative()
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.errorbar(data.ch(), data.cnt(), data.u_cnt())
    ax2.errorbar(data_2ndder.ch(), data_2ndder.cnt(), data_2ndder.u_cnt(), color='r')
    
    # threshold = data_2ndder.u_cnt() * (-significance)
    ax2.plot(data.ch(), np.zeros(data.ch().size), color='g')

    plt.show()
    
    # cnt_1diff =


if __name__ == "__main__":
    # a = Spectrum_data(np.arange(0, 2), np.array([9, 16])) / 10.
    # b = Spectrum_data(np.arange(0, 100), np.random.randint(100, size=(100,)))
    # b_sub = b.sub_data(np.array([0, b.ch()[-1]]))
    # print(b_sub.ch())

    bkg = np.loadtxt("/home/yk/work/Other1_Gamma밀도계/rawdata/" + "20200908_Bkg_3600s.TKA")
    eu152 = np.loadtxt("/home/yk/work/Other1_Gamma밀도계/rawdata/" + "20200908_CalibEu152_600s.TKA")
    print(bkg)
    print(bkg.size)
    ch = np.arange(0, bkg.size)
    print(ch)
    bkgdata = Spectrum_data(ch, eu152) / 3600.
    eu152data = Spectrum_data(ch, eu152) / 600. - bkgdata
    # search_peak(bkgdata, 1.5)
    # eu152data.errorbar()
    # eu152data.movmean(11).errorbar()
    # eu152data.smooth_savgol(5, 2).errorbar()
    # plt.show()

    # bkgdata.derivative().errorbar()
    # plt.show()
    # plt.yscale('log')
    search_peak(eu152data, 1.5)
