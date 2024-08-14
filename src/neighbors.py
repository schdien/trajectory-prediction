import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate


def synchronous_sampling(state,traj):
    pos = state[:,:2]
    poss = traj[:,:2]
    a = pos[:2] - poss[:3]
    dist = np.linalg.norm(a, axis=1)
    close_ind = np.argsort(dist)[:2]
    start_ind = min(close_ind)
    a = a[start_ind]
    b = poss[max(close_ind)]-poss[start_ind]
    start_ind = start_ind + abs(np.dot(a,b)/np.linalg.norm(b)**2)
    end_ind = len(traj)
    sync_inds = np.arange(start_ind,end_ind-1)
    inds = np.arange(0,end_ind)
    f = interpolate.interp1d(inds, traj.T, kind='quadratic',assume_sorted=True)
    sync_traj = f(sync_inds).T
    return sync_traj

class SequenceQuerier:
    def __init__(self,trajs) -> None:
        self.trajs = np.vstack(trajs)
        lengths = np.array([len(traj) for traj in trajs])
        self.ends = np.repeat(np.cumsum(lengths)-1,lengths)
        self.normalizer = MinMaxScaler()
        normalized_trajs = self.normalizer.fit_transform(self.trajs)
        self.tree = KDTree(normalized_trajs)

    def query(self, x, step, mode, k=100):
        n_x = self.normalizer.transform(x)
        dist,inds = self.tree.query(n_x,k=k)
        ends = self.ends[inds]

        if mode == 'state':
            y = self.trajs[np.min(np.stack((inds+step,ends)),axis=0)]

        elif mode == 'trajectory':
            if len(x) != 1:
                raise ValueError('the length of x must be 1.')
            #去除重复轨迹的索引
            ends,uniq = np.unique(ends,return_index=True)
            inds = inds.take(uniq)
            dist = dist.take(uniq)
            #根据索引取多个相似轨迹
            y = []
            if step == -1:
                for ind,end in zip(inds,ends):
                    traj = self.trajs[ind-1:end]
                    y.append(synchronous_sampling(x,traj))
                #y = [self.trajs[ind:end] for ind,end in zip(inds,ends)]
            else:
                for ind,end in zip(inds,ends):
                    traj = self.trajs[ind-1:min(ind+step+1,end)]
                    y.append(synchronous_sampling(x,traj))
                #y = [self.trajs[ind:min(ind+step,end)] for ind,end in zip(inds,ends)]

        weight = np.exp(-(dist/0.003)**2/2)
        weight = weight / np.max(weight)
        return y, weight
    
