def parameters(dataset, lable_true):
    import numpy as np
    # General Parameters
    k = len(set(lable_true))     # number of clusters.
    t_max = 100                  # maximum number of iterations.
    Restarts = 10                # number of FCM restarts.
    fuzzy_degree = 2             # fuzzy membership degree
    labeled_rate = 20            # rate of labeled data (0-100)

    
    f = np.zeros([len(lable_true),k])
    if min(set(lable_true))==0:
        f[np.arange(lable_true.size), lable_true.astype(int)] = 1
    else:
        f[np.arange(lable_true.size), lable_true.astype(int) - 1] = 1


    # specific parameters
    if dataset == 'iris':
        alpha = 0.5
        alpha2 = 10 
        p=3
    elif dataset == 'balance':
        alpha = 0.5
        alpha2 = 100
        p = 1  
    elif dataset == 'new_breast':
        alpha = 0.05
        alpha2 = 100
        p=3
    elif dataset == 'bupa':
        alpha = 0.05
        alpha2 = 100
        p=1  
    elif dataset == 'cancer':
        alpha = 0.001
        alpha2 = 10
        p=3   
    elif dataset == 'australian':
        alpha = 10
        alpha2 = 100
        p=5   
    elif dataset == 'blood':
        alpha = 10
        alpha2 = 10
        p = 5 
    elif dataset == 'diabet':
        alpha = 10
        alpha2 = 100
        p=5 
    elif dataset == 'heberman':
        alpha = 0.5
        alpha2 = 100
        p=7 
    elif dataset == 'seed':
        alpha = 0.1
        alpha2 = 1
        p=3 
    elif dataset == 'spectfheart':
        alpha = 100
        alpha2 = 100
        p=7 
    elif dataset == 'vowel':
        alpha = 0.01
        alpha2 = 10
        p=1 
    elif dataset == 'wine':
        alpha = 100
        alpha2 = 0.1
        p=3   
    elif dataset == 'thyroid':
        alpha = 10
        alpha2 = 1
        p=7 
    elif dataset == 'waveform':
        alpha = 0.05
        alpha2 = 100
        p=1
        
    return k, t_max, Restarts, fuzzy_degree, alpha, alpha2, f, labeled_rate,p
