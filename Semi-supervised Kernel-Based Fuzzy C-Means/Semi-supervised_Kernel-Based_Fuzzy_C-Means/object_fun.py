def object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, f, b, sigma):
    import math
    import numpy as np

    distance_u = np.zeros([k, np.shape(data[b==0])[0], col])
    dNK_u = np.zeros([np.shape(data[b==0])[0], k])
    
    distance_l = np.zeros([k, np.shape(data[b==1])[0], col])
    dNK_l = np.zeros([np.shape(data[b==1])[0], k])
       
    
    for j in range(k):
        
        #Unlabled
        distance_u = 1-np.exp((-1*((data[b==0]-np.tile(center_points[j, :], (np.shape(data[b==0])[0], 1))) ** 2)) / (2*(sigma**2)))
        dNK_u[:, j] = np.sum(distance_u, 1)
        
        #labeled
        distance_l = 1-np.exp((-1*((data[b==1]-np.tile(center_points[j, :], (np.shape(data[b==1])[0], 1))) ** 2)) / (2*(sigma**2)))
        dNK_l[:, j] = np.sum(distance_l, 1)
        
    j_fun1 = np.sum(np.sum(dNK_u * np.transpose(Cluster_elem ** fuzzy_degree)))
    j_fun2 = np.sum(np.sum(dNK_l * (f[b==1] ** fuzzy_degree)))
    
  
    return j_fun1 + j_fun2
