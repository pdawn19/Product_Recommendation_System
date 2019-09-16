import numpy as np


def calc_observed_rmse(r, rHat, mask):
    errA = np.where(mask != 0, r, 0)
    errB = np.where(mask != 0, rHat, 0)
    count = np.sum(mask)
    res = np.square(np.subtract(errA, errB)).mean() * np.size(mask) / count
    return res


def calc_unobserved_rmse(r, rHat, mask):
    errA = np.where(mask == 0, r, 0)
    errB = np.where(mask == 0, rHat, 0)
    count = np.size(mask) - np.sum(mask)
    res = np.square(np.subtract(errA, errB)).mean() * np.size(mask) / count
    return res
