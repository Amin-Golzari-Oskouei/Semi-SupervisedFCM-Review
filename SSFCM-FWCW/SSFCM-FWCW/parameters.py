def parameters(dataset, lable_true, data, col):

    import numpy as np

    # General Parameters
    k = len(set(lable_true))     # number of clusters.
    t_max = 100                  # maximum number of iterations.
    Restarts = 10                # number of FCM restarts.
    fuzzy_degree = 2             # fuzzy membership degree
    p_init = 0                   # initial p.
    p_max = 0.5                  # maximum p.
    p_step = 0.01                # p step.
    labeled_rate = 20            # rate of labeled data (0-100)
    
    f = np.zeros([len(lable_true),k])
    if min(set(lable_true))==0:
        f[np.arange(lable_true.size), lable_true.astype(int)] = 1
    else:
        f[np.arange(lable_true.size), lable_true.astype(int) - 1] = 1
    
# specific parameters
    if dataset == 'iris':
        q = 2
        beta_memory = 0
        l = 1
        alpha = 2
    elif dataset == 'balance':
        q = 10                        # the value for the feature weight updates.
        beta_memory = 0.1            # amount of memory for the cluster weight updates.
        l = 0.1
        alpha = 4    
    elif dataset == 'new_breast':
        q = -2                       # the value for the feature weight updates.
        beta_memory = 0            # amount of memory for the cluster weight updates.
        l = 0.0001
        alpha = 1
    elif dataset == 'bupa':
        q = -4                       # the value for the feature weight updates.
        beta_memory = 0            # amount of memory for the cluster weight updates.
        l = 1
        alpha = 1  
    elif dataset == 'cancer':
        q = -2                       # the value for the feature weight updates.
        beta_memory = 0.3             # amount of memory for the cluster weight updates.
        l = 0.1   
        alpha = 2
    elif dataset == 'australian':
        q = -2                       # the value for the feature weight updates.
        beta_memory = 0.3             # amount of memory for the cluster weight updates.
        l = 0.1 
        alpha = 1    
    elif dataset == 'blood':
        q = 4                       # the value for the feature weight updates.
        beta_memory = 0.1             # amount of memory for the cluster weight updates.
        l = 0.0001
        alpha = 5   
    elif dataset == 'diabet':
        q = -2                        # the value for the feature weight updates.
        beta_memory = 0.3            # amount of memory for the cluster weight updates.
        l = 0.01
        alpha = 2 
    elif dataset == 'heberman':
        q = 2                        # the value for the feature weight updates.
        beta_memory = 0.3            # amount of memory for the cluster weight updates.
        l = 0.0001
        alpha = 10
    elif dataset == 'seed':
        q = -4                        # the value for the feature weight updates.
        beta_memory = 0            # amount of memory for the cluster weight updates.
        l = 0.0001
        alpha = 2
    elif dataset == 'spectfheart':
        q = 2                        # the value for the feature weight updates.
        beta_memory = 0           # amount of memory for the cluster weight updates.
        l = 1
        alpha = 10
    elif dataset == 'vowel':
        q = 8                        # the value for the feature weight updates.
        beta_memory = 0.1            # amount of memory for the cluster weight updates.
        l = 0.0001
        alpha = 1
    elif dataset == 'wine':
        q = -2                        # the value for the feature weight updates.
        beta_memory = 0            # amount of memory for the cluster weight updates.
        l = 1
        alpha = 1  
    elif dataset == 'thyroid':
        q = 8                        # the value for the feature weight updates.
        beta_memory = 0.1            # amount of memory for the cluster weight updates.
        l = 0.0001
        alpha = 2
    elif dataset == 'waveform':
        q = 2                        # the value for the feature weight updates.
        beta_memory = 0.3            # amount of memory for the cluster weight updates.
        l = 0.0001
        alpha = 1
    
    
    
    
    landa = np.zeros(col)
    landa = l/np.var(data, 0)
    landa[np.where(np.isinf(landa))] = 1
        
    return k, t_max, Restarts, fuzzy_degree, q, p_init, p_max, p_step, beta_memory, landa, labeled_rate, alpha, f
