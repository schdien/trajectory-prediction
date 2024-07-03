import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler


class SequenceQuerier:
    def __init__(self,trajs) -> None:
        self.trajs = np.vstack(trajs)
        lengths = np.array([len(traj) for traj in trajs])
        self.ends = np.repeat(np.cumsum(lengths)-1,lengths)
        self.normalizer = MinMaxScaler()
        normalized_trajs = self.normalizer.fit_transform(self.trajs)
        self.tree = BallTree(normalized_trajs)

    def query(self, x, step, mode, k=100):
        x = self.normalizer.transform(x)
        dist,inds = self.tree.query(x,k=k)
        ends = self.ends[inds]

        if mode == 'state':
            y = self.trajs[min(inds+step,ends)]

        elif mode == 'trajectory':
            if len(x) != 1:
                raise ValueError('the length of x must be 1.')
            #去除重复轨迹的索引
            ends,uniq = np.unique(ends,return_index=True)
            inds = inds.take(uniq)
            dist = dist.take(uniq)
            #根据索引取多个相似轨迹
            if step == -1:
                y = [self.trajs[ind:end] for ind,end in zip(inds,ends)]
            else:
                y = [self.trajs[ind:min(ind+step,end)] for ind,end in zip(inds,ends)]

        weight = np.exp(-(dist/0.003)**2/2)
        weight = weight / np.max(weight)
        return y, weight
