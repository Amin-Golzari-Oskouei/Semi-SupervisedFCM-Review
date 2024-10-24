def object_fun(row, col, k, Cluster_elem, center_points, data, f, b,u_bar, landa):
    import numpy as np

    distance = np.zeros([k, row, col])
    dNK = np.zeros([row, k])
    for j in range(k):
        distance = (data-np.tile(center_points[j, :], (row, 1))) ** 2
        dNK[:, j] = np.sqrt(np.sum(distance, 1))
        
    j_fun = np.sum(np.sum(dNK * np.transpose(Cluster_elem)))
    
    tmp = np.abs(Cluster_elem - u_bar)  * np.log(np.abs(Cluster_elem - u_bar)) 
    tmp[np.where(np.isnan(tmp))] = 0
    j_fun2 = np.sum(np.sum(tmp))
    
    
    return j_fun + (1/landa * j_fun2)
