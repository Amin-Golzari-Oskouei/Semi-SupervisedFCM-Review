def main(data, center_points, k, t_max, row, fuzzy_degree, col, f, b):
    import numpy as np
    import math
    import sys
    from object_fun import object_fun

#-----------------------------------------------------------------------------------------
    # Other initializations.
    Iter = 1                  # Number of iterations.
    E_w_old = math.inf        # Previous iteration objective (used to check convergence).

    Cluster_elem = np.zeros([k, row])
    Cluster_elem[:, b==1] = np.transpose(f[b==1,:])
    distance = np.zeros([k, row, col])
    dNK = np.zeros([row, k])
    F_nm = np.zeros([row, row, k])
    abs_diff = np.zeros([k, k, row])
    Fnm_Dnm = np.zeros([row, row])
    
    
    D_nm = np.linalg.norm(data[:, np.newaxis] - data, axis=2, ord=2)
    
    f[b==0,:]= 0
    
    for j in range(k):
        F_nm[:, :, j] = f[:,j] * f[:,j].T
        Fnm_Dnm = Fnm_Dnm + (D_nm * F_nm[:, :, j])

    # --------------------------------------------------------------------------
    print('Start of Fuzzy C-Means iterations')
    print('----------------------------------')

    # the algorithm iteration procedure
    while 1:

        # Update the cluster assignments.
        for j in range(k):
            distance = (data-np.tile(center_points[j, :], (row, 1))) ** 2
            dNK[:, j] = np.sqrt(np.sum(distance, 1))
            abs_diff[j, :, :] = np.abs(Cluster_elem[j, :] - Cluster_elem[:, :])
        
        tmp3 = np.zeros([k, row, k])
        tmp4 = np.zeros([k, row])

        for j in range(k):
            #
            tmp3[j, :, :]  = (Fnm_Dnm @ abs_diff [: , j, :].T) / np.tile(dNK[:,j] + np.sum(Fnm_Dnm, 1), (k,1)).T
            tmp4[j, :] = np.sum(Fnm_Dnm, 0) + dNK[:, j]
        
        tmp3[np.where(np.isnan(tmp3))] = 0
        tmp4[np.where(np.isinf(tmp4))] = 0
        
        Cluster_elem = (1+np.sum(tmp3, 2)) / (tmp4 * (np.sum(1/tmp4, 0)))
        
        Cluster_elem[np.where(np.isnan(Cluster_elem))] = 1
        Cluster_elem[np.where(np.isinf(Cluster_elem))] = 1
        
        # Calculate the MinMax k-means objective.
        E_w = object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, Fnm_Dnm)

        if math.isnan(E_w) == False:
            print(f'The algorithm objective is E_w={E_w}')

        # Check for convergence. Never converge if in the current (or previous)
        # iteration empty or singleton clusters were detected.
        if (math.isnan(E_w) == False) and (math.isnan(E_w_old) == False) and (abs(1 - E_w / E_w_old) < 10**-6 or Iter >= t_max):

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Converging for after {Iter} iterations.')
            print(f'The proposed algorithm objective is E_w={E_w}.')

            return Cluster_elem

        E_w_old = E_w
        
        mf = Cluster_elem ** fuzzy_degree
        for j in range(k):
            center_points[j, :] = np.matmul(np.expand_dims(mf[j, :], 0), data) / np.matmul(np.expand_dims(mf[j, :], 0), np.ones_like(data))

        Iter = Iter + 1
