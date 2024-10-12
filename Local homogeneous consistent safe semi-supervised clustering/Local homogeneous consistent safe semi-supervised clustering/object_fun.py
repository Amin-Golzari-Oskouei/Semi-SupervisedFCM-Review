def object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, f, b, alpha, alpha2, pre_cluster_element, w_rn, u_nk, u_rk, labeled_mask, unlabeled_mask):
    import math
    import numpy as np
    
    distance = np.zeros([k, row, col])
    dNK = np.zeros([row, k])
    u_diff = np.zeros([row, row, k])

    for j in range(k):
        distance = (data-np.tile(center_points[j, :], (row, 1))) ** 2
        dNK[:, j] = np.sqrt(np.sum(distance, 1))
        u_diff[:, :, j] = (Cluster_elem[j, np.newaxis, :] - Cluster_elem[j, :, np.newaxis]) ** 2

    j_fun1 = np.sum(np.sum(dNK * np.transpose(Cluster_elem ** fuzzy_degree))) 
    j_fun2 = np.sum(np.sum((   ( u_nk - f[labeled_mask,:].T ) ** 2   ) * dNK[labeled_mask,:].T, axis=0))
    diff_squared = np.sum((u_nk[:, np.newaxis, :] - u_rk[:, :, np.newaxis])**2, axis=0)
    j_fun3 = np.sum(w_rn * diff_squared)
    
    
    return j_fun1 + (alpha * j_fun2) + (alpha2 * j_fun3)



