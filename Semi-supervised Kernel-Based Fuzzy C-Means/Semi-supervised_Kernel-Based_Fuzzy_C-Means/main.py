def main(data, center_points, k, t_max, row, fuzzy_degree, col, f, b, sigma):
    import numpy as np
    import math
    import sys
    from object_fun import object_fun

#-----------------------------------------------------------------------------------------
    # Other initializations.
    Iter = 1                  # Number of iterations.
    E_w_old = math.inf        # Previous iteration objective (used to check convergence).

    Cluster_elem = np.zeros([k, np.shape(data[b==0])[0]])
    distance = np.zeros([k, np.shape(data[b==0])[0], col])
    dNK = np.zeros([np.shape(data[b==0])[0], k])

    # --------------------------------------------------------------------------
    print('Start of Fuzzy C-Means iterations')
    print('----------------------------------')

    # the algorithm iteration procedure
    while 1:

        # Update the cluster assignments.
        for j in range(k):
            distance = 1-np.exp((-1*((data[b==0]-np.tile(center_points[j, :], (np.shape(data[b==0])[0], 1))) ** 2)) / (2*(sigma**2)))
            dNK[:, j] = np.sum(distance, 1)


        tmp1 = np.zeros([np.shape(data[b==0])[0], k])
        for j in range(k):
            tmp2 = (dNK / np.transpose(np.tile(dNK[:, j], (k, 1)))) ** (1 / (fuzzy_degree - 1))
            tmp2[np.where(np.isnan(tmp2))] = 0
            tmp2[np.where(np.isinf(tmp2))] = 0
            tmp1 = tmp1 + tmp2

        Cluster_elem = np.transpose(1 / tmp1)
        Cluster_elem[np.where(np.isnan(Cluster_elem))] = 1
        Cluster_elem[np.where(np.isinf(Cluster_elem))] = 1

        for j in np.where(dNK == 0)[0]:
            Cluster_elem[np.where(dNK[j, :] == 0)[0], j] = 1 / len(np.where(dNK[j, :] == 0)[0])
            Cluster_elem[np.where(dNK[j, :] != 0)[0], j] = 0

        # Calculate the MinMax k-means objective.
        E_w = object_fun(row, col, k, Cluster_elem, center_points, fuzzy_degree, data, f, b, sigma)

        if math.isnan(E_w) == False:
            print(f'The algorithm objective is E_w={E_w}')

        # Check for convergence. Never converge if in the current (or previous)
        # iteration empty or singleton clusters were detected.
        if (math.isnan(E_w) == False) and (math.isnan(E_w_old) == False) and (abs(1 - E_w / E_w_old) < 10**-6 or Iter >= t_max):

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Converging for after {Iter} iterations.')
            print(f'The proposed algorithm objective is E_w={E_w}.')
            temp = np.zeros([k, row])
            temp[:,b==0] = Cluster_elem
            temp[:,b==1] = np.transpose(f[b==1])

            return temp

        E_w_old = E_w

        # Update the cluster centers.
        mf_u = Cluster_elem ** fuzzy_degree
        mf_l = np.transpose((f[b==1] ** fuzzy_degree))
        
        for j in range(k):
            
            distance_u = np.exp((-1*((data[b==0]-np.tile(center_points[j, :], (np.shape(data[b==0])[0], 1))) ** 2)) / (2*(sigma**2)))
            distance_l = np.exp((-1*((data[b==1]-np.tile(center_points[j, :], (np.shape(data[b==1])[0], 1))) ** 2)) / (2*(sigma**2)))
            
            numerator  = np.matmul(np.expand_dims(mf_u[j, :], 0),(data[b==0] * distance_u) ) + np.matmul(np.expand_dims(mf_l[j, :], 0),(data[b==1] * distance_l) )
            denominator = np.matmul(np.expand_dims(mf_u[j, :], 0), distance_u) + np.matmul(np.expand_dims(mf_l[j, :], 0), distance_l)
            
            center_points[j, :] = numerator / denominator

        Iter = Iter + 1
