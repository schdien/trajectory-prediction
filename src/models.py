import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from dtw import dtw
import matplotlib.pyplot as plt


class TrajPredictor:
    def __init__(self,traj_list):
        self.trajs = np.vstack(traj_list)
        self.identifiers = np.concatenate([i * np.ones(len(traj),dtype=int) for i,traj in enumerate(traj_list)])
        self.edges = np.cumsum(np.array([len(traj) for traj in traj_list]))
        self.normalizer = MinMaxScaler()
        normalized_trajs = self.normalizer.fit_transform(self.trajs)
        self.tree = BallTree(normalized_trajs)

    def _get_edge_index(self,ind):
        identifier = self.identifiers[ind]
        return self.edges[identifier]

    def _get_pred_index(self,ind,step,policy):
        pred_ind = ind + step
        curr_id = self.identifiers[ind]
        pred_id = self.identifiers[pred_ind]
        if policy == 'remove':
            return pred_ind[np.where(curr_id==pred_id)]
        elif policy == 'edge':
            mismatch = curr_id!=pred_id
            mismatch_id = curr_id[mismatch]
            pred_ind[mismatch] = self.edges[mismatch_id]
            return pred_ind
        else:
            raise ValueError('policy must be \'remove\' or \'edge\'')
    
    def predict(self,x,step):
        points,weights = self._query_point(x,step,100)
        mean = []
        for i,(p,w) in enumerate(zip(points[:,:2],weights)):
            mean_p,label = self._cluster_point(p, w,3,5)
            ind = i*np.ones(mean_p.shape[0])
            mean_p = np.column_stack((ind,mean_p))
            mean.append(mean_p)
        return np.row_stack(mean)

    def _query_trajectory(self,x,step=-1,k=200):
        if len(x) != 1:
            raise ValueError('the length of x must be 1.')
        x = self.normalizer.transform(x)
        dist,ind = self.tree.query(x,k=k)
        #去除重复轨迹的索引
        traj_ind = self.identifiers[ind]
        _, unique_ind = np.unique(traj_ind, return_index=True)
        dist = dist[:,unique_ind]
        ind = ind[:,unique_ind]
        #根据索引取多个相似轨迹
        if step == -1: #预测到结束点
            end_ind = self._get_edge_index(ind)
        else: 
            end_ind = self._get_pred_index(ind,step,'edge')
        slice_ind = np.hstack([np.arange(s, e) for s, e in zip((ind + 1).T, end_ind.T)])
        pred_trajs = self.trajs.take(slice_ind,axis=0)
        pred_traj_ids = self.identifiers.take(slice_ind)
        return pred_traj_ids, pred_trajs, dist

    
    def _cluster_trajectory(self,trajs):
        trajs= self.normalizer.transform(trajs)
        cluster = DBSCAN(eps=0.02,min_samples=10)
        cluster.fit(trajs)
        return cluster.labels_

    def predict_trajectory(self,x,step=-1):
        pred_traj_ids,pred_trajs, d = self._query_trajectory(x,step)
        labels = self._cluster_trajectory(pred_trajs)
        return pred_trajs,labels


if __name__=='__main__':
    import data
    import trajplot
    raw_trajs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=3000)
    traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
    test_data = np.array([traj_list[0][100]])
    #test_data = traj_list[0][98:100]
    predictor = TrajPredictor(traj_list)
    trajs,labels = predictor.predict_trajectory(test_data,step=-1)
    colors = trajplot.label2color(labels)
    trajplot.scatter2d(trajs,colors)
