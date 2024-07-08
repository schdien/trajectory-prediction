import numpy as np
from sklearn.cluster import DBSCAN
from dtw import dtw
from scipy import stats


# ------------------------------------------------------------------------- #
class SequenceCluster:
    def __init__(self,eps,min_samples=2) -> None:
        def dtw_distance(x,y):
            x = x[x!=-1].reshape((-1,2))
            y = y[y!=-1].reshape((-1,2))
            return dtw(x,y).normalizedDistance
        self.cluster = DBSCAN(eps=eps,min_samples=min_samples,metric=dtw_distance)
        self.labels = None

    def fit(self,seqs):
        #填充trajs,让它们的长度相同
        padded_seqs = -1*np.ones((len(seqs),len(max(seqs, key=len)),seqs[0].shape[-1]))
        for i,seq in enumerate(seqs):
            padded_seqs[i,:len(seq)] = seq
        padded_seqs = padded_seqs[:,:,:2]
        self.labels = self.cluster.fit_predict(padded_seqs.reshape((padded_seqs.shape[0],-1)))

    def fit_predict(self,seqs):
        self.fit(seqs)
        return self.labels


# ------------------------------------------------------------------------- #
def mode_filter(x,r):
    l = len(x)
    filtered_x = np.zeros(l)
    for i in range(l):
        start = max(0,i-r)
        end = min(l,i+r+1)
        filtered_x[i] = stats.mode(x[start:end],keepdims=True)[0]
    return filtered_x


class PartitionalSequenceCluster:
    def __init__(self,normalizer,eps=0.003,min_samples=2) -> None:
        def distance(x,y):
            p_dist = np.linalg.norm(x[:2]-y[:2])
            v_dist = np.linalg.norm(x[2:]-y[2:])
            return max(p_dist,0.1*v_dist)
        self.cluster = DBSCAN(eps=eps,min_samples=min_samples,metric=distance)
        self.normalizer = normalizer
        self.labels = None
        self.sequences = None
        self.sample_indices = None

    def fit(self,seqs):
        ref_traj = seqs[0][:,:2]
        #轨迹同步采样
        self.sample_indices = np.array([dtw(ref_traj,traj[:,:2],step_pattern='asymmetric',open_begin=True,open_end=True).index2 for traj in seqs])
        self.sequences = np.array([traj[i] for traj,i in zip(seqs,self.sample_indices)])
        states = self.sequences.swapaxes(0,1)
        #聚类
        n_states = np.array([self.normalizer.transform(state) for state in states])
        labels = np.array([self.cluster.fit_predict(state) for state in n_states])
        #统计滤波器去噪
        self.labels = np.array([mode_filter(label,20) for label in labels.T])

    def fit_predict(self,seqs,time_first:bool):
        self.fit(seqs)
        if time_first:
            return self.sequences.swapaxes(0,1), self.labels.T
        else:
            return self.sequences, self.labels