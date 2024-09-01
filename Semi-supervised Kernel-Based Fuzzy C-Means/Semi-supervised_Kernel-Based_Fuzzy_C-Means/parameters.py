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
        sigma = 0.1 
    elif dataset == 'balance':
        sigma = 1000000  
    elif dataset == 'breast':
        sigma = 10
    elif dataset == 'bupa':
        sigma = 10  
    elif dataset == 'cancer':
        sigma = 0.01  
    elif dataset == 'australian':
        sigma = 0.1  
    elif dataset == 'blood':
        sigma = 10
    elif dataset == 'diabet':
        sigma = 0.01
    elif dataset == 'heberman':
        sigma = 0.001
    elif dataset == 'seed':
        sigma = 0.1
    elif dataset == 'spectfheart':
        sigma = 1
    elif dataset == 'vowel':
        sigma = 0.1
    elif dataset == 'wine':
        asigma = 1  
    elif dataset == 'thyroid':
        sigma = 0.01
    elif dataset == 'waveform':
        sigma = 1
        
    return k, t_max, Restarts, fuzzy_degree, sigma, f, labeled_rate
