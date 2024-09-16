def object_fun(row, col, k, Cluster_elem, landa, center_points, fuzzy_degree, w, z, q, p, data, f, alpha, b, alpha2, neighbors, Nr):
    import math
    import numpy as np
    distance = np.zeros([k, row, col])
    dNK = np.zeros([row, k])
    j_fun3 = 0
    for j in range(k):
        distance[j, :, :] = (1 - np.exp((-1 * np.tile(landa, (row, 1))) * ((data - np.tile(center_points[j, :], (row, 1))) ** 2)))
        WBETA = np.transpose(z[j, :] ** q)
        WBETA[np.where(np.isinf(WBETA))] = 0
        dNK[:, j] = np.squeeze(np.matmul(w[j] ** p * np.reshape(distance[j, :, :], (row, col)), np.expand_dims(WBETA, 1)))
        cc = (1 - Cluster_elem[j, :]) ** fuzzy_degree
        j_fun3 = j_fun3 + np.sum(np.transpose(Cluster_elem[j, :] ** fuzzy_degree) * np.sum(np.transpose(np.tile(cc, (Nr, 1))), 1))


    j_fun1 = np.sum(np.sum(dNK * (Cluster_elem ** fuzzy_degree)))
    mf2 = (Cluster_elem - (np.transpose(np.tile(b, (k,1))) * f )) ** fuzzy_degree
    j_fun2 = alpha * np.sum(np.sum(mf2 * dNK))
    
    return j_fun1 + (alpha2 * j_fun2) + (alpha / Nr) * j_fun3
