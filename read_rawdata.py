import re
import numpy as np
# import gamma_spectroscopy


def meastime(filename):
    meastime_regpattern = re.compile("_(\d+)([smhd])\w*?[\._]")
    meastime_match = meastime_regpattern.search(filename)
    if meastime_match.group(2) == "s":
        conversion2sec = 1.
    elif meastime_match.group(2) == "m":
        conversion2sec = 60.
    elif meastime_match.group(2) == "h":
        conversion2sec = 3600.
    elif meastime_match.group(2) == "d":
        conversion2sec = 86400.
    meastime_sec = float(meastime_match.group(1)) * conversion2sec

    return meastime_sec


def load_tka(filename):
    data = np.loadtxt(filename)
    data_cps = data / meastime(filename)

    return data_cps


def load_simul_data(filename):
    data = np.loadtxt(filename, skiprows=1)
    edep_data = data[:, 1]
    # edep_data_new = gamma_spectroscopy.sample_ene_geb(edep_data, 1.22369657e-05, 6.09542015e-02, -1.00390113e-01)
    edep_data_new = gamma_spectroscopy.sample_ene_geb(edep_data, -0.00148391,  0.06320691, -0.102377)
    hist, ene_axis = np.histogram(edep_data_new, bins=np.arange(0., 2., 0.01))
    data_cps = np.append(hist, 0.) / meastime(filename) / 0.01

    return data_cps, ene_axis


if __name__ == "__main__":
    filename1 = "20200908_CalibCs137Co60_600s_.TKA"
    filename2 = "20200819_Bkg_30min_1.TKA"
    data = load_tka(filename1)
