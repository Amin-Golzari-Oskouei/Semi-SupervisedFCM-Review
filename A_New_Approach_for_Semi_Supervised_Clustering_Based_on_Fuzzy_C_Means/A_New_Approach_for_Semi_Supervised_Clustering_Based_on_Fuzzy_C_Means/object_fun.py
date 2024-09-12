def object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, Fnm_Dnm):
    import math
    import numpy as np

    distance = np.zeros([k, row, col])
    dNK = np.zeros([row, k])
    abs_diff = np.zeros([row, row, k])

    for j in range(k):
        distance = (data-np.tile(center_points[j, :], (row, 1))) ** 2
        dNK[:, j] = np.sqrt(np.sum(distance, 1))
        abs_diff[:, :, j] = (Cluster_elem[j, np.newaxis, :] - Cluster_elem[j, :, np.newaxis]) ** 2
        
    
    j_fun1 = np.sum(np.sum(dNK * np.transpose(Cluster_elem ** fuzzy_degree)))

    j_fun2 = np.sum(np.tile(Fnm_Dnm, (k,1,1)).T * abs_diff)

    return j_fun1 + j_fun2













