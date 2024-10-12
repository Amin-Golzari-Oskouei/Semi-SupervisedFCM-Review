def main(data, center_points, k, t_max, row, fuzzy_degree, col, f, b, alpha, alpha2, pre_cluster_element, w_rn, labeled_mask, unlabeled_mask):
    import numpy as np
    import math
    import sys
    from object_fun import object_fun

#-----------------------------------------------------------------------------------------
    # Other initializations.
    Iter = 1                  # Number of iterations.
    E_w_old = math.inf        # Previous iteration objective (used to check convergence).




    Cluster_elem = np.zeros([k, row])
    distance = np.zeros([k, row, col])
    dNK = np.zeros([row, k])
    u_nk = np.zeros([k, np.sum(labeled_mask)])
    u_rk = np.zeros([k, np.sum(unlabeled_mask)])
    

    # --------------------------------------------------------------------------
    print('Start of Fuzzy C-Means iterations')
    print('----------------------------------')

    # the algorithm iteration procedure
    while 1:

        # Update the cluster assignments.
        for j in range(k):
            distance = (data-np.tile(center_points[j, :], (row, 1))) ** 2
            dNK[:, j] = np.sqrt(np.sum(distance, 1))


        tmp1 = np.zeros([k, np.sum(labeled_mask)])
        p_nk = np.zeros([k, np.sum(labeled_mask)])
        q_nk = np.zeros([k, np.sum(labeled_mask)])
        s_rk = np.zeros([k, np.sum(unlabeled_mask)])
        t_rk = np.zeros([k, np.sum(unlabeled_mask)])
        
        for i in range(k): 
            tmp1[i, :] = (2 * alpha2 * (np.dot(w_rn.T, u_rk[i, :])))
            q_nk[i, :] = (2 * dNK[labeled_mask, i])    +    (2 * alpha * dNK[labeled_mask, i])    +    2 * alpha2 * (np.sum(w_rn, axis=0))
            s_rk[i, :] = 2 * alpha2 * np.dot(w_rn, u_nk[i, :])
            t_rk[i, :] = (2 * dNK[unlabeled_mask, i])    +     2 * alpha2 * (np.sum(w_rn, axis=1))

        p_nk = 2 * alpha * (f[labeled_mask,:]*dNK[labeled_mask,:]).T  +   tmp1

        q_nk[np.where(np.isnan(q_nk))] = 0
        t_rk[np.where(np.isinf(t_rk))] = 0
        
        tmp2 = np.zeros([k, np.sum(labeled_mask)])
        tmp3 = np.zeros([k, np.sum(unlabeled_mask)])
        
        for i in range(k):
            tmp2[i, :] = ( 1 - (np.sum(p_nk / q_nk , axis=0)) )         /     np.sum(1   /  q_nk , axis=0 )
            tmp3[i, :] = ( 1 - (np.sum(s_rk / t_rk, axis=0)) )         /      np.sum(1   /  t_rk , axis=0 )

            
        u_nk = ( p_nk + tmp2 ) / q_nk
        u_rk = ( s_rk + tmp3 ) / t_rk
        
        Cluster_elem[:, labeled_mask] = u_nk
        Cluster_elem[: ,unlabeled_mask] = u_rk

        Cluster_elem[np.where(np.isnan(Cluster_elem))] = 1
        Cluster_elem[np.where(np.isinf(Cluster_elem))] = 1
        
        for j in np.where(dNK == 0)[0]:
            Cluster_elem[np.where(dNK[j, :] == 0)[0], j] = 1 / len(np.where(dNK[j, :] == 0)[0])
            Cluster_elem[np.where(dNK[j, :] != 0)[0], j] = 0

        # Calculate the MinMax k-means objective.
        E_w = object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, f, b, alpha, alpha2, pre_cluster_element, w_rn, u_nk, u_rk,labeled_mask , unlabeled_mask)

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

        mf1 = Cluster_elem ** fuzzy_degree
        mf2 = alpha * (u_nk - f[labeled_mask,:].T) ** fuzzy_degree
        for j in range(k):
            center_points[j, :] =   (   np.matmul(np.expand_dims(mf1[j, :], 0),data) + (alpha * (np.matmul(np.expand_dims(mf2[j, :], 0),data[labeled_mask,:])))   ) / (   np.matmul(np.expand_dims(mf1[j, :], 0),np.ones_like(data)) + (alpha * (np.matmul(np.expand_dims(mf2[j, :], 0),np.ones_like(data)[labeled_mask,:])))) 
                                    

        Iter = Iter + 1