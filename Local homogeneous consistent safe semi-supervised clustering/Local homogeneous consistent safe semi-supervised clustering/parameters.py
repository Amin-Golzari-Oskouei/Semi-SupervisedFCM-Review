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
        alpha = 2
        alpha2 = 2.1 
        p=5
    elif dataset == 'balance':
        alpha = 9.6
        alpha2 = 2.1  
    elif dataset == 'breast':
        alpha = 1.6
        alpha2 = 9.6
    elif dataset == 'bupa':
        alpha = 1.1
        alpha2 = 6.1  
    elif dataset == 'cancer':
        alpha = 2.6
        alpha2 = 3.6  
    elif dataset == 'australian':
        alpha = 6.1
        alpha2 = 9.1  
    elif dataset == 'blood':
        alpha = 1.1
        alpha2 = 1.6
    elif dataset == 'diabet':
        alpha = 1.1
        alpha2 = 7.1
    elif dataset == 'heberman':
        alpha = 3.1
        alpha2 = 1.1
    elif dataset == 'seed':
        alpha = 1.1
        alpha2 = 2.1
    elif dataset == 'spectfheart':
        alpha = 4.1
        alpha2 = 2.6
    elif dataset == 'vowel':
        alpha = 9.6
        alpha2 = 8.1
    elif dataset == 'wine':
        alpha = 5.1
        alpha2 = 9.1  
    elif dataset == 'thyroid':
        alpha = 1.6
        alpha2 = 1.1
    elif dataset == 'waveform':
        alpha = 3.6
        alpha2 = 1.1
        
    return k, t_max, Restarts, fuzzy_degree, alpha, alpha2, f, labeled_rate,p
