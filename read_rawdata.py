import re
import numpy as np


def load_tka(fileName):
    meastime_regpattern = re.compile("_(\d+)([smhd])\w*?[\._]")
    meastime_match = meastime_regpattern.search(fileName)
    if meastime_match.group(2) == "s":
        conversion2sec = 1.
    elif meastime_match.group(2) == "m":
        conversion2sec = 60.
    elif meastime_match.group(2) == "h":
        conversion2sec = 3600.
    elif meastime_match.group(2) == "d":
        conversion2sec = 86400.
    meastime_sec = float(meastime_match.group(1)) * conversion2sec

    data = np.loadtxt(fileName)
    data_cps = data / meastime_sec

    return data_cps


def load_simul_data(filename):
    
    return data_cps


if __name__ == "__main__":
    fileName1 = "20200908_CalibCs137Co60_600s_.TKA"
    fileName2 = "20200819_Bkg_30min_1.TKA"
    data = load_tka(filename1)
