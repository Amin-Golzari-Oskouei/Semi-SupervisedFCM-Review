def object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, f, b, alpha, alpha2, pre_cluster_element):
    import math
    import numpy as np
    
    distance = np.zeros([k, row, col])
    dNK = np.zeros([row, k])
    for j in range(k):
        distance = (data-np.tile(center_points[j, :], (row, 1))) ** 2
        dNK[:, j] = np.sqrt(np.sum(distance, 1))
    j_fun1 = np.sum(np.sum(dNK * np.transpose(Cluster_elem ** fuzzy_degree)))
    j_fun2 = np.sum(np.sum(dNK * np.transpose((Cluster_elem -   np.transpose(np.transpose(np.tile(b, (k,1))) * f ) ) ** fuzzy_degree)))
    j_fun3 = np.sum(np.sum(dNK * np.transpose((Cluster_elem -   np.transpose(np.transpose(np.tile(b, (k,1))) * pre_cluster_element.T ) ) ** fuzzy_degree)))
    
    return j_fun1 + (alpha * j_fun2) + (alpha2 * j_fun3)
