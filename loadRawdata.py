import re
import numpy as np


def loadTKA(fileName):
    measTRegPattern = re.compile("_(\d+)([smhd])\w*?[\._]")
    measTMatch = measTRegPattern.search(fileName)
    if measTMatch.group(2) == "s":
        conversion2sec = 1.
    elif measTMatch.group(2) == "m":
        conversion2sec = 60.
    elif measTMatch.group(2) == "h":
        conversion2sec = 3600.
    elif measTMatch.group(2) == "d":
        conversion2sec = 86400.
    measTInSec = float(measTMatch.group(1)) * conversion2sec

    data = np.loadtxt(fileName)
    dataCPS = data / measTInSec

    return dataCPS


if __name__ == "__main__":
    fileName1 = "20200908_CalibCs137Co60_600s_.TKA"
    fileName2 = "20200819_Bkg_30min_1.TKA"
    p = re.compile("_(\d+)([smhd])\w*?[\._]")
    m = p.search(fileName1)
    print(m)
    if m.group(2) == "s":
        conversion2sec = 1.
    elif m.group(2) == "m":
        conversion2sec = 60.
    elif m.group(2) == "h":
        conversion2sec = 3600.
    elif m.group(2) == "d":
        conversion2sec = 86400.
    timeInSec = float(m.group(1)) * conversion2sec
