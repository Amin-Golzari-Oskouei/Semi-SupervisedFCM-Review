from sklearn.metrics import confusion_matrix
import numpy as np
    

def compute_sn(pre_cluster_element, lable_true, labeled_mask, unlabeled_mask):
    
    predicted_labels_unlabeled = np.argmax(pre_cluster_element, axis=0)  
    
    labels = np.array(lable_true[labeled_mask]-1, dtype=np.uint8)
    y_pred = np.array(predicted_labels_unlabeled[labeled_mask], dtype=np.uint8)
    
    
    MC = confusion_matrix(labels, y_pred, normalize='pred')
    
    # For matches:
    match_weights = MC[labels, y_pred] * pre_cluster_element[y_pred, labeled_mask]
    
    # For non-matches: 
    non_match_weights = MC[labels, y_pred] * (1 -  pre_cluster_element[y_pred, labeled_mask])
    
    # Combine the results using the mask
    match_mask = labels == y_pred  # Boolean array of shape (N,)
    sn = np.where(match_mask, match_weights, non_match_weights)
    
    return sn