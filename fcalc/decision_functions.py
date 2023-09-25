import numpy as np

def standard_method(pos, neg, C = 0):
    sup_pos = np.sum(pos[0][pos[1] <= len(neg[0]) * C]) / len(pos[0])**2
    sup_neg = np.sum(neg[0][neg[1] <= len(pos[0]) * C]) / len(neg[0])**2
    
    if sup_pos > sup_neg:
        return 1
    elif sup_pos == sup_neg:
        return -1
    else:
        return 0
