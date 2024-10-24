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
        pass
    elif dataset == 'balance':
        pass
    elif dataset == 'breast':
        pass
    elif dataset == 'bupa':
        pass
    elif dataset == 'cancer':
        pass
    elif dataset == 'australian':
        pass
    elif dataset == 'blood':
        pass
    elif dataset == 'diabet':
        pass
    elif dataset == 'heberman':
        pass
    elif dataset == 'seed':
        pass
    elif dataset == 'spectfheart':
        pass
    elif dataset == 'vowel':
        pass
    elif dataset == 'wine':
        pass
    elif dataset == 'thyroid':
        pass
    elif dataset == 'waveform':
        pass

    return k, t_max, Restarts, fuzzy_degree, f, labeled_rate
