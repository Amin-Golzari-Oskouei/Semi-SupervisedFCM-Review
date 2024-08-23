def main(data, center_points, k, t_max, row, fuzzy_degree, col, f, b, alpha, alpha2, pre_cluster_element):
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



    # --------------------------------------------------------------------------
    print('Start of Fuzzy C-Means iterations')
    print('----------------------------------')

    # the algorithm iteration procedure
    while 1:

        # Update the cluster assignments.
        for j in range(k):
            distance = (data-np.tile(center_points[j, :], (row, 1))) ** 2
            dNK[:, j] = np.sqrt(np.sum(distance, 1))

        tmp1 = np.zeros([row, k])

        for j in range(k):
            tmp2 = (dNK / np.tile(dNK[:, j].reshape(-1, 1), k)) ** (1 / (fuzzy_degree - 1))
            tmp2[np.isinf(tmp2)] = 0
            tmp2[np.isnan(tmp2)] = 0
            tmp1 += tmp2
            
        Cluster_elem = ((1 / (1 + alpha + alpha2)) *
                        (((1 + alpha + alpha2 - np.sum(alpha * (np.transpose(np.tile(b, (k, 1))) * f) +
                        alpha2 * (np.transpose(np.tile(b, (k, 1))) * pre_cluster_element.T), axis=1).reshape(-1,1)) / tmp1) +
                         (alpha * (np.transpose(np.tile(b, (k, 1))) * f) + alpha2 * (np.transpose(np.tile(b, (k, 1))) * pre_cluster_element.T)))).T

        Cluster_elem[np.isnan(Cluster_elem)] = 1
        Cluster_elem[np.isinf(Cluster_elem)] = 1
        
        for j in np.where(dNK == 0)[0]:
            Cluster_elem[np.where(dNK[j, :] == 0)[0], j] = 1 / len(np.where(dNK[j, :] == 0)[0])
            Cluster_elem[np.where(dNK[j, :] != 0)[0], j] = 0

        # Calculate the MinMax k-means objective.
        E_w = object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, f, b, alpha, alpha2, pre_cluster_element)

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
        mf2 = alpha * (Cluster_elem -   np.transpose(np.transpose(np.tile(b, (k,1))) * f ) ) ** fuzzy_degree
        mf3 = alpha2 * (Cluster_elem -   np.transpose(np.transpose(np.tile(b, (k,1))) * pre_cluster_element.T ) ) ** fuzzy_degree
        mf = mf1 + mf2 + mf3
        for j in range(k):
            center_points[j, :] = np.matmul(np.expand_dims(mf[j, :], 0), data) / np.matmul(np.expand_dims(mf[j, :], 0), np.ones_like(data))

        Iter = Iter + 1
