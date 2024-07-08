import numpy as np


def estimate_mean(states,labels,weights):
    kinds = np.unique(labels)
    valid_kinds = kinds[kinds!=-1]
    if len(valid_kinds)!=0:
        kinds = valid_kinds
    cluster_ps = [states[labels == kind] for kind in kinds]
    cluster_ws = [weights[labels == kind] for kind in kinds]
    weights_sum = np.array([np.sum(w) for w in cluster_ws])
    means = np.array([np.sum(s.T * w, axis=1) / p for s, w, p in zip(cluster_ps, cluster_ws, weights_sum)])
    probs = weights_sum / np.sum(weights_sum)
    return kinds, probs, means


def estimate_sequence_mean(states_seq,labels_seq,weights):
    means_seq = [[]]
    for i,(states,labels) in enumerate(zip(states_seq,labels_seq)):
        kinds,probs,means = estimate_mean(states,labels,weights)
        if i == 0:
            prev_kinds = kinds
        if len(kinds) != len(prev_kinds):
            means_seq.append([])
        means_seq[-1].append(np.column_stack([kinds,probs,means]))
        prev_kinds = kinds
    means_seq = [np.stack(means,axis=1) for means in means_seq]
    return means_seq
