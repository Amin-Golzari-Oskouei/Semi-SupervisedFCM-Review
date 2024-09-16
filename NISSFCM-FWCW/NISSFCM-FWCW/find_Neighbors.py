import numpy as np
from scipy.spatial.distance import pdist, squareform

def naneucdist(XI, XJ):
    sqdx = (XI - XJ) ** 2
    nstar = np.nansum(sqdx)
    D2 = 1 - np.exp(-nstar)
    return D2

def find_neighbors(NR, X):
    D = pdist(X, metric=naneucdist)
    D = squareform(D)
    sorted_indices = np.argsort(D, axis=1)
    result = sorted_indices[:, 1:NR+1]
    return result