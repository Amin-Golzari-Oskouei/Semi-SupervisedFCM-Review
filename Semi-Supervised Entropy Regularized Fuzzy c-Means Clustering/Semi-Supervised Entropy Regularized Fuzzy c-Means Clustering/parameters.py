def parameters(dataset, lable_true):
    import numpy as np
    # General Parameters
    k = len(set(lable_true))     # number of clusters.
    t_max = 100                  # maximum number of iterations.
    Restarts = 10                # number of FCM restarts.
    labeled_rate = 20            # rate of labeled data (0-100)
    
    f = np.zeros([len(lable_true),k])
    if min(set(lable_true))==0:
        f[np.arange(lable_true.size), lable_true.astype(int)] = 1
    else:
        f[np.arange(lable_true.size), lable_true.astype(int) - 1] = 1


    # specific parameters
    if dataset == 'iris':
        landa = 10
    elif dataset == 'balance':
        landa = 1
    elif dataset == 'new_breast':
        landa = 1
    elif dataset == 'bupa':
        landa = 6
    elif dataset == 'cancer':
        landa = 7
    elif dataset == 'australian':
        landa = 2
    elif dataset == 'blood':
        landa = 1
    elif dataset == 'diabet':
        landa = 1
    elif dataset == 'heberman':
        landa = 1
    elif dataset == 'seed':
        landa = 10
    elif dataset == 'spectfheart':
        landa = 4
    elif dataset == 'vowel':
        landa = 1
    elif dataset == 'wine':
        landa = 10
    elif dataset == 'thyroid':
        landa = 1
    elif dataset == 'waveform':
        landa = 1

    return k, t_max, Restarts, f, labeled_rate, landa
