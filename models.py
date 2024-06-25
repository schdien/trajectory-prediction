import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from dtw import dtw
import matplotlib.pyplot as plt


class TrajPredictor:
    def __init__(self,traj_list):
        self.trajs = np.row_stack(traj_list)
        self.identifiers = np.concatenate([i * np.ones(len(traj)) for i,traj in enumerate(traj_list)])
        self.normalizer = MinMaxScaler()
        normalized_trajs = self.normalizer.fit_transform(self.trajs)
        self.tree = BallTree(normalized_trajs)

    def _get_pred_index(self,ind,step,policy):
        pred_ind = ind + step
        curr_id = self.identifiers[ind]
        pred_id = self.identifiers[pred_ind]
        if policy == 'remove':
            return pred_ind[np.where(curr_id==pred_id)]
        elif policy == 'end':
            mismatch = curr_id!=pred_id
            mismatch_id = curr_id[mismatch]
            end_ind = np.array([np.where(self.identifiers==m)[0][-1] for m in mismatch_id])
            pred_ind[mismatch] = end_ind
            return pred_ind
        else:
            raise ValueError('policy must be \'remove\' or \'end\'')

    def _query_point(self,x,step,k=100):
        x = self.normalizer.transform(x)
        dist,ind = self.tree.query(x,k=k)
        weights = np.exp(-(dist/0.003)**2/2)
        weights = weights / np.max(weights)
        pred_ind = self._get_pred_index(ind,step)
        pred_points = self.trajs[pred_ind]
        return pred_points, weights
    
    def _cluster_point(self,sample,weight,eps,min_smaples):
        cluster = DBSCAN(eps=eps, min_samples=min_smaples)
        cluster.fit(sample,sample_weight=weight)
        label = cluster.labels_
        cs = np.unique(label)
        valid_cs = cs[cs!=-1]
        if len(valid_cs)!=0:
            cs = valid_cs
        sample = [sample[np.where(label == c)] for c in cs]
        weight = [weight[np.where(label == c)] for c in cs]
        weights_sum = np.array([np.sum(w) for w in weight])
        mean = np.array([np.sum(s.T * w, axis=1) / p for s, w, p in zip(sample, weight, weights_sum)])
        prob = weights_sum / np.sum(weights_sum)
        return np.column_stack((prob, mean)),label
    
    def predict_step(self, x):
        points,weights = self._query_point(x,1)
        mean = []
        for i,(v,w) in enumerate(zip(points[:,2:],weights)):
            mean_v,label = self._cluster_point(v, w,1.5,5)
            ind = i*np.ones(mean_v.shape[0])
            mean_v = np.column_stack((ind,mean_v))
            mean.append(mean_v)
        return np.row_stack(mean)
        #return Y.squeeze(),l
    
    def predict(self,x,step):
        points,weights = self._query_point(x,step,100)
        mean = []
        for i,(p,w) in enumerate(zip(points[:,:2],weights)):
            mean_p,label = self._cluster_point(p, w,3,5)
            ind = i*np.ones(mean_p.shape[0])
            mean_p = np.column_stack((ind,mean_p))
            mean.append(mean_p)
        return np.row_stack(mean)

    def _query_trajectory(self,x,step,k=100):
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
        end_ind = self._get_pred_index(ind,step,'end')
        slice_ind = np.array([np.arange(s, e) for s, e in zip((ind + 1).T, end_ind.T)])
        pred_trajs = self.trajs[slice_ind]
        return pred_trajs, dist

    def _cluster_trajectory(self,sample):
        sample = sample[:,:2]
        cluster = DBSCAN(eps=3,min_samples=3,metric=lambda x,y: dtw(x.reshape((-1,2)),y.reshape((-1,2))).distance)
        cluster.fit(sample.reshape((sample.shape[0],-1)))
        return cluster.labels_

    def predict_trajectory(self,x,step=-1):
        Y, W = self._query_trajectory(x,step)
        return Y,self._cluster(Y,W)



if __name__=='__main__':
    import data
    import trajplot
    raw_trajs = data.load_files("H:/TrajectoryPrediction/adsb/PEK-SHA", usecols=[7, 8, 5],num=3000)
    traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
    test_data = np.array([traj_list[0][100]])
    #test_data = traj_list[0][98:100]
    predictor = TrajPredictor(traj_list)
    trajs,labels = predictor.predict_trajectory(test_data,step=600)
    subplot = plt.subplot()
    colors = trajplot.label2color(labels)
    trajplot.plot2d(trajs,colors)
