import numpy as np
import scipy.optimize
import scipy.signal
from sklearn import linear_model
import matplotlib.pyplot as plt
# import read_rawdata


def geb(ene: np.ndarray, a: float, b: float, c: float):
    in_sqrt = ene + c * ene ** 2
    in_sqrt[np.where(in_sqrt < 0.)] = 0.
    fwhm = a + b * np.sqrt(in_sqrt)  # in MeV
    return fwhm


def sample_ene_geb(ene: np.ndarray, a: float, b: float, c: float):
    std = geb(ene, a, b, c) / (2 * np.sqrt(np.log(2)))
    std[np.where(std < 0.)] = 0.
    ene_new = np.random.normal(ene, std)
    return ene_new


def fit_geb(ene: np.ndarray, fwhm: np.ndarray):
    ene = np.append(ene, 0.)
    fwhm = np.append(fwhm, 0.)
    pars, cov = scipy.optimize.curve_fit(f=geb, xdata=ene, ydata=fwhm)
    return pars, cov


def calibrate_bycsco(x: np.ndarray, cnt: list):
    if type(cnt) is np.ndarray:
        cnt = [cnt]

    calib_pts = []
    enes = [0.662, 1.173, 1.332]
    k = 0

    for n in range(len(cnt)):
        y = cnt[n]
        plt.plot(x, y)
        plt.draw()

        pts = plt.ginput(n=2)
        x_idx = np.digitize(np.array([pts[0][0], pts[1][0]]), x)
        peaks, _ = scipy.signal.find_peaks(
            y[x_idx[0]:x_idx[1]], prominence=np.max(cnt)*0.1)
        peaks += x_idx[0]
        plt.plot(x[peaks], y[peaks], "xr")
        plt.draw()
        plt.show()

        for i in range(len(peaks)):
            # ene = float(input("Peak energy (MeV): ")); enes.append(ene)
            ene = enes[k]
            k += 1
            calib_pts.append((peaks[i], ene))

    print("-- Selected (channel, energy) points --")
    print(calib_pts)
    ch_pts = [[i[0]] for i in calib_pts]
    ene_pts = [i[1] for i in calib_pts]
    plt.plot(ch_pts, ene_pts, "xr")

    model = linear_model.LinearRegression()
    model.fit(ch_pts, ene_pts)
    chtmp = [[i] for i in np.arange(2048)]
    enetmp = model.predict(chtmp)
    plt.plot(chtmp, enetmp)
    plt.show()

    return model


def calibrate_byeu152(eu152data, fig=True):
    x = np.arange(2048)
    y = eu152data
    peaks, _ = scipy.signal.find_peaks(y, prominence=y*0.3, height=np.max(y)*0.01)
    peaks = peaks[1:] # ignore first peak (noise)
    if fig:
        plt.plot(x, y)
        plt.plot(x[peaks], y[peaks], "xr")
        plt.show()

    ch_pts = peaks.reshape(-1, 1)
    ene_pts = np.array([0.040, 0.122, 0.245, 0.344, 0.779, 0.964, 1.122, 1.408])
    model = linear_model.LinearRegression()
    model.fit(ch_pts, ene_pts)
    chtmp = x.reshape(-1, 1)
    enetmp = model.predict(chtmp)
    if fig:
        plt.plot(ch_pts, ene_pts, "xr")
        plt.plot(chtmp, enetmp)
        plt.show()

    rslt_half = scipy.signal.peak_widths(y, peaks, rel_height=0.5)
    ene_half_left = model.predict(rslt_half[2].reshape(-1, 1))
    ene_half_right = model.predict(rslt_half[3].reshape(-1, 1))
    fwhms = ene_half_right - ene_half_left
    geb_pars, _ = fit_geb(ene_pts, fwhms)

    return model, geb_pars


def find_broadpeak(x: np.ndarray, y: np.ndarray, npeaks=1):
    def g1peak_p1bkg(x, p0, p1, a, b, c):
        return a * np.exp(-(x-b)**2 / (2*(c**2))) + (p0 + p1*x)

    def g2peak_p1bkg(x, p0, p1, a1, b1, c1, a2, b2, c2):
        return a1 * np.exp(-(x-b1)**2 / (2*(c1**2))) + a2 * np.exp(-(x-b2)**2 / (2*(c2**2))) + (p0 + p1*x)

    plt.plot(x, y)
    plt.draw()

    pts = plt.ginput(n=2)

    x_idx = np.digitize(np.array([pts[0][0], pts[1][0]]), x)

    x_sel = x[x_idx[0]:x_idx[1]]
    y_sel = y[x_idx[0]:x_idx[1]]

    peaks, _ = scipy.signal.find_peaks(y_sel, prominence=np.max(y)*0.1)

    if npeaks == 1:
        pars, cov = scipy.optimize.curve_fit(f=g1peak_p1bkg, xdata=x_sel, ydata=y_sel,
                                             p0=[1., -1., y_sel[peaks[0]],
                                                 x_sel[peaks[0]], 0.01],
                                             bounds=([0., -np.inf, 0., x_sel[0], 0.],
                                                     [np.inf, 0., np.inf, x_sel[-1], x_sel[-1]-x_sel[0]]))
        a = np.sqrt(2 * np.pi) * 0.997
        A = pars[2]
        sA = np.sqrt(cov[2, 2])
        B = pars[4]
        sB = np.sqrt(cov[4, 4])
        sAB = cov[2, 4]
        area = a * A * B
        areaErr = np.sqrt(a * (A * B) ** 2 * ((sA / A) ** 2 +
                                              (sB / B) ** 2 + 2 * (sAB / (A * B))))
        fwhm = pars[4] * 2 * np.sqrt(2 * np.log(2))

        plt.plot(x_sel, g1peak_p1bkg(x_sel, *pars))
        plt.draw()

    elif npeaks == 2:
        pars, cov = scipy.optimize.curve_fit(f=g2peak_p1bkg, xdata=x_sel, ydata=y_sel,
                                             p0=[1., -1., y_sel[peaks[0]], x_sel[peaks[0]],
                                                 0.01, y_sel[peaks[1]], x_sel[peaks[1]], 0.01],
                                             bounds=([0., -np.inf, 0., x_sel[0], 0., 0., x_sel[int(x_sel.size/2)], 0.],
                                                     [np.inf, 0., np.inf, x_sel[int(x_sel.size/2)], (x_sel[-1]-x_sel[0])/2, np.inf, x_sel[-1], (x_sel[-1]-x_sel[0])/2]))
        a = np.sqrt(2 * np.pi) * 0.997
        A = pars[2]
        sA = np.sqrt(cov[2, 2])
        B = pars[4]
        sB = np.sqrt(cov[4, 4])
        sAB = cov[2, 4]
        area1 = a * A * B
        areaErr1 = np.sqrt(a * (A * B) ** 2 * ((sA / A) **
                                               2 + (sB / B) ** 2 + 2 * (sAB / (A * B))))
        fwhm1 = pars[4] * 2 * np.sqrt(2 * np.log(2))

        A = pars[5]
        sA = np.sqrt(cov[5, 5])
        B = pars[7]
        sB = np.sqrt(cov[7, 7])
        sAB = cov[5, 7]
        area2 = a * A * B
        areaErr2 = np.sqrt(a * (A * B) ** 2 * ((sA / A) **
                                               2 + (sB / B) ** 2 + 2 * (sAB / (A * B))))
        fwhm2 = pars[7] * 2 * np.sqrt(2 * np.log(2))

        area = [area1, area2]
        areaErr = [areaErr1, areaErr2]
        fwhm = [fwhm1, fwhm2]

        plt.plot(x_sel, g2peak_p1bkg(x_sel, *pars))
        plt.draw()

    plt.show()

    return area, areaErr, fwhm, pars


if __name__ == "__main__":
    ene_eu152 = np.array([0.040, 0.122, 0.245, 0.344, 0.779, 0.964, 1.122, 1.408])
    bkg = read_rawdata.load_tka("/home/evan/work/Other1_Gamma밀도계/rawdata/" + "20200908_Bkg_3600s.TKA")
    eu152 = read_rawdata.load_tka("/home/evan/work/Other1_Gamma밀도계/rawdata/" + "20200908_CalibEu152_600s.TKA")
    csco = read_rawdata.load_tka("/home/evan/work/Other1_Gamma밀도계/rawdata/" + "20200908_CalibCs137Co60_600s.TKA")
    # plt.plot(bkg)
    # plt.plot(eu152)
    # plt.plot(csco)
    # plt.show()

    x = np.arange(2048)
    y = eu152 - bkg
    peaks, _ = scipy.signal.find_peaks(y, prominence=y*0.3, height=np.max(y)*0.01)
    peaks = peaks[1:]
    plt.plot(x, y)
    plt.plot(x[peaks], y[peaks], "xr")
    plt.show()

    ch_pts = [[i] for i in peaks]
    ene_pts = ene_eu152
    model = linear_model.LinearRegression()
    model.fit(ch_pts, ene_pts)
    chtmp = [[i] for i in np.arange(2048)]
    enetmp = model.predict(chtmp)
    plt.plot(ch_pts, ene_pts, "xr")
    plt.plot(chtmp, enetmp)
    plt.show()