# This demo shows how to call the extention of fuzzy C-means algorithm 
# For the demonstration, the Iris dataset of the above paper is used.
# Courtesy of A.Golzari Oskouei

import numpy as np
from main import main
from calculateMetrics import calculateMetrics
import scipy.io
from parameters import parameters
import time


# Load the dataset.
dataset='iris'
mat = scipy.io.loadmat(f'{dataset}'+'.mat')
data = mat[f'{dataset}']

lable_true = data[:, -1]
data = data[:, 0:-1]
[row, col] = data.shape

data = (data-data.min())/(data.max()-data.min())  # normalized data

# Algorithm parameters.
k, t_max, Restarts, fuzzy_degree, f, labeled_rate = parameters(dataset, lable_true)

# initializations.
np.random.seed(1373)
ACC_repeat = np.empty([Restarts])
NMI_repeat = np.empty([Restarts])
F_repeat = np.empty([Restarts])
DBS_repeat = np.empty([Restarts])
SI_repeat = np.empty([Restarts])
R_index_repeat = np.empty([Restarts])
FMI_repeat = np.empty([Restarts])
JI_repeat = np.empty([Restarts])
RunTime_repeat = np.empty([Restarts])

# --------------------------------------------------------
# Clustering the samples using the proposed procedure.
# --------------------------------------------------------

for repeat in range(Restarts):
    print(f'========================================================')
    print(f'Proposed Algorithm: Restart {repeat+1}.')
    
    b = np.zeros(row)
    tmp1 = np.random.choice(row, int(row*labeled_rate/100), replace=False)
    b[tmp1] = 1
       
    if labeled_rate==0:
        # Randomly initialize the cluster centers.
        center_points = data[np.random.choice(data.shape[0], k, replace=False), :]
    else:
        # initialize the cluster centers by labels.
        center_points = np.transpose(np.transpose(data) @ (np.transpose(np.tile(b, (k,1))) * f)) / np.transpose(np.tile(b@f,(col,1)))
        
        if np.isnan(np.sum(center_points)):
            tmp = data[np.random.choice(data.shape[0], k, replace=False), :]
            center_points[np.isnan(center_points)] = tmp[np.isnan(center_points)]
    
    u_bar = np.copy(f)
    u_bar[b == 0] = 0
    u_bar = u_bar.T
    # Execute proposed algorithm.
    # Get the cluster assignments and other parameters.
    start_time = time.time()
    Cluster_elem = main(data, center_points, k, t_max, row, fuzzy_degree, col, f, b, u_bar)
    end_time = time.time()

    RunTime_repeat[repeat] = end_time - start_time

    lable_pre = Cluster_elem.argmax(axis=0)
    if min(set(lable_true))==0:
        lable_pre[tmp1] = lable_true[tmp1]
    else:
        lable_pre[tmp1] = lable_true[tmp1] - 1 
        

    if Cluster_elem is not None and len(np.unique(lable_pre))==k:
        ans = calculateMetrics(lable_pre, lable_true, row, data)
        ACC_repeat[repeat] = ans[0]
        NMI_repeat[repeat] = ans[1]
        F_repeat[repeat] = ans[4]
        DBS_repeat[repeat] = ans[5]
        SI_repeat[repeat] = ans[6]
        R_index_repeat[repeat] = ans[7]
        FMI_repeat[repeat] = ans[9]
        JI_repeat[repeat] = ans[10]

        print(f'The accurcy score in {repeat+1}th  Restart is {ans[0]}.')
        print(f'The NMI score in {repeat+1}th  Restart is {ans[1]}.')
        print(f'The F1 score in {repeat+1}th  Restart is {ans[4]}.')
        print(f'The davies bouldin score in {repeat+1}th  Restart is {ans[5]}.')
        print(f'The silhouette score in {repeat+1}th  Restart is {ans[6]}.')
        print(f'The R_index score in {repeat+1}th  Restart is {ans[7]}.')
        print(f'The FMI score in {repeat+1}th is Restart {ans[9]}.')
        print(f'The JI score in {repeat+1}th is Restart {ans[10]}.') 
        print(f'The runtime in {repeat+1}th  Restart is {RunTime_repeat[repeat]}.')
        print(f'End of Restart {repeat + 1}')

        print('========================================================')
    else:
        ACC_repeat[repeat] = np.nan
        NMI_repeat[repeat] = np.nan
        F_repeat[repeat] = np.nan
        DBS_repeat[repeat] = np.nan
        SI_repeat[repeat] = np.nan
        R_index_repeat[repeat] = np.nan
        FMI_repeat[repeat] = np.nan
        JI_repeat[repeat] = np.nan

        print(f'The accurcy score in {repeat+1}th  Restart is NaN.')
        print(f'The NMI score in {repeat+1}th  Restart is NaN.')
        print(f'The F1 score in {repeat+1}th  Restart is NaN.')
        print(f'The davies bouldin score in {repeat+1}th  Restart is NaN.')
        print(f'The silhouette score in {repeat+1}th  Restart is NaN.')
        print(f'The R_index score in {repeat+1}th  Restart is NaN.')
        print(f'The FMI score in {repeat+1}th is Restart NaN.')
        print(f'The JI score in {repeat+1}th is Restart NaN.') 
        print(f'The runtime in {repeat+1}th  Restart is {RunTime_repeat[repeat]}.')

        print(f'End of Restart {repeat + 1}')
        print('========================================================')

print(f'Average accurcy score over {Restarts} restarts: {np.nanmean(ACC_repeat)}')
print(f'Average NMI score over {Restarts} restarts: {np.nanmean(NMI_repeat)}')
print(f'Average F1 score over {Restarts} restarts: {np.nanmean(F_repeat)}')
print(f'Average davies bouldin score over {Restarts} restarts: {np.nanmean(DBS_repeat)}')
print(f'Average silhouette score over {Restarts} restarts: {np.nanmean(SI_repeat)}')
print(f'Average R_index score over {Restarts} restarts: {np.nanmean(R_index_repeat)}')
print(f'Average FMI score over {Restarts} restarts: {np.nanmean(FMI_repeat)}')
print(f'Average JI score score over {Restarts} restarts: {np.nanmean(JI_repeat)}')
print(f'Average runtime over {Restarts} restarts: {np.nanmean(RunTime_repeat)}')
