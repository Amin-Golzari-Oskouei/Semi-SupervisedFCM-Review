from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def compute_wrn(data, pre_cluster_element, b, f, p, row, labeled_mask, unlabeled_mask):
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
    
    out = w_rn[labeled_mask, :][:, unlabeled_mask] 
    return out.T