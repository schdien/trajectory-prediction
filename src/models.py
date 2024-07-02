import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN,HDBSCAN,OPTICS
from sklearn.preprocessing import MinMaxScaler
from dtw import dtw
from modefilter import mode_filter
import matplotlib.pyplot as plt


class TrajPredictor:
    def __init__(self,trajs):
        self.trajs = np.vstack(trajs)
        lengths = np.array([len(traj) for traj in trajs])
        self.ends = np.repeat(np.cumsum(lengths)-1,lengths)
        self.normalizer = MinMaxScaler()
        normalized_trajs = self.normalizer.fit_transform(self.trajs)
        self.tree = BallTree(normalized_trajs)

    def _query_point(self,x,step,k=100):
        x = self.normalizer.transform(x)
        dist,ind = self.tree.query(x,k=k)
        weights = np.exp(-(dist/0.003)**2/2)
        weights = weights / np.max(weights)
        pred_ind = self._get_pred_index(ind,step)
        pred_points = self.trajs[pred_ind]
        return pred_points, weights
    
    def _cluster_point(self,points,weight,eps,min_smaples):
        cluster = DBSCAN(eps=eps, min_samples=min_smaples)
        cluster.fit(points,sample_weight=weight)
        label = cluster.labels_
        cs = np.unique(label)
        valid_cs = cs[cs!=-1]
        if len(valid_cs)!=0:
            cs = valid_cs
        points = [points[np.where(label == c)] for c in cs]
        weight = [weight[np.where(label == c)] for c in cs]
        weights_sum = np.array([np.sum(w) for w in weight])
        mean = np.array([np.sum(s.T * w, axis=1) / p for s, w, p in zip(points, weight, weights_sum)])
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

    def _query_trajectory(self,x,step=-1,k=70):
        if len(x) != 1:
            raise ValueError('the length of x must be 1.')
        x = self.normalizer.transform(x)
        dist,inds = self.tree.query(x,k=k)
        ends = self.ends[inds]
        #去除重复轨迹的索引
        ends,uniq = np.unique(ends,return_index=True)
        inds = inds.take(uniq)
        #根据索引取多个相似轨迹
        if step == -1:
            pred_trajs = [self.trajs[ind:end] for ind,end in zip(inds,ends)]
        else:
            pred_trajs = [self.trajs[ind:min(ind+step,end)] for ind,end in zip(inds,ends)]
        return pred_trajs, dist

    def _cluster_trajectory(self,trajs):
        def dtw_distance(x,y):
            x = x[x!=-1].reshape((-1,2))
            y = y[y!=-1].reshape((-1,2))
            return dtw(x,y).distance
        trajs = trajs[:,:,:2]
        cluster = DBSCAN(eps=10,min_samples=2,metric=dtw_distance)
        cluster.fit(trajs.reshape((trajs.shape[0],-1)))
        return cluster.labels_
    
    def _cluster_trajectory2(self,trajs):
        #trajs = np.array(trajs)
        ref_traj = trajs[0][:,:2]
        #这一步计算不考虑速度
        time_inds = np.array([dtw(ref_traj,traj[:,:2],step_pattern='asymmetric',open_begin=True,open_end=True).index2 for traj in trajs])
        samples = np.array([traj[time_ind] for traj,time_ind in zip(trajs,time_inds)])
        samples = [self.normalizer.transform(sample) for sample in samples.swapaxes(0,1)]
        def metric(x,y):
            p_dist = np.linalg.norm(x[:2]-y[:2])
            v_dist = np.linalg.norm(x[2:]-y[2:])
            return max(p_dist,0.1*v_dist)
        cluster = DBSCAN(eps=0.003,min_samples=2,metric=metric)
        labels = np.array([cluster.fit(sample).labels_ for sample in samples])
        #统计滤波器去噪
        labels = np.array([mode_filter(label,20) for label in labels.T]).T
        return samples,labels

    def predict_trajectory(self,x,step=-1):
        pred_trajs, W = self._query_trajectory(x,step)
        samples,labels = self._cluster_trajectory2(pred_trajs)
        return samples,labels




if __name__=='__main__':
    import data
    import trajplot
    raw_trajs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=3000)
    traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
    test_data = np.array([traj_list[0][100]])
    #test_data = traj_list[0][98:100]
    predictor = TrajPredictor(traj_list)
    truth = traj_list[0][50:200]
    trajs,labels = predictor.predict_trajectory(test_data,step=250)
    colors = [trajplot.label2color(label) for label in labels]
    #trajplot.plot2d(trajs+[truth],colors+['y'])
    #trajplot.scatter2d2(trajs[-2],colors[-2])
    trajplot.scatter2d(trajs,colors)
