from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import pdist
    

def compute_wrn(data, pre_cluster_element, b, f, p, row, labeled_mask, unlabeled_mask, sigma):
    labeled_indices = np.where(b == 1)[0]  
    unlabeled_indices = np.where(b == 0)[0]  
    distances = euclidean_distances(data)
    sorted_indices = np.argsort(distances, axis=1)
    w_rn = np.zeros((row, row))
    predicted_labels_labeled = np.argmax(f, axis=1)  
    predicted_labels_unlabeled = np.argmax(pre_cluster_element, axis=0)  

    for row in labeled_indices:
        neighbors = sorted_indices[row, 1:p+1]
        unlabeled_neighbors = neighbors[np.isin(neighbors, unlabeled_indices)]
        
        matching_labels = predicted_labels_unlabeled[unlabeled_neighbors] == predicted_labels_labeled[row]
        w_rn[row, unlabeled_neighbors[matching_labels]] = 1

     

    
    def dfun(u, v, sigma):
        sqdx = np.sqrt(np.sum((u-v)**2))
        D2 = np.exp(-1*sqdx/ (sigma**2))
        return D2 
    
       
    dist = DistanceMetric.get_metric(dfun, sigma=sigma)
    dm = dist.pairwise(data, data)

    
    out = w_rn[labeled_mask, :][:, unlabeled_mask] * dm[labeled_mask, :][:, unlabeled_mask]
    
    return out.T