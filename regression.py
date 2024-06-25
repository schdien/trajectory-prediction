import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from dtw import dtw
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self):
        self.X = None
        self.Y = None
        self.tree = None
        self.cluster = DBSCAN(eps=1.5,min_samples=5)
        self.normalizer = MinMaxScaler()

    def fit(self,X,Y):
        self.X = X
        X = self.normalizer.fit_transform(X)
        self.Y = Y
        self.tree = BallTree(X)

    def _query(self,x,k=50):
        x = self.normalizer.transform(x)
        dist,ind = self.tree.query(x,k=k)
        Y = self.Y[ind]
        W = np.exp(-(dist/0.003)**2/2)
        W = W / np.max(W)
        return Y,W

    # 混合分布估计
    def _estimate_cluster_mean(self,sample,weight):
        self.cluster.fit(sample,sample_weight=weight)
        label = self.cluster.labels_
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

    def predict_step(self, x, return_cluster=False):
        Y,W = self._query(x,100)
        mean = np.empty((0, 4))
        for i,(Yi,Wi) in enumerate(zip(Y,W)):
            m,l = self._estimate_cluster_mean(Yi, Wi)
            ind = i*np.ones(m.shape[0])
            m = np.column_stack((ind,m))
            mean = np.row_stack((mean, m))
        return mean


class Regressor3:
    def __init__(self,traj_list):
        self.trajs = np.row_stack(traj_list)
        self.index = np.concatenate([i * np.ones(len(traj)) for i,traj in enumerate(traj_list)])
        self.normalizer = MinMaxScaler()
        normalized_trajs = self.normalizer.fit_transform(self.trajs)
        self.tree = BallTree(normalized_trajs)

    def _query(self,x,k=100,step=1,return_traj=False):
        if return_traj and len(x) != 1:
            raise ValueError('the length of x must be 1 when return trajectory.')
        x = self.normalizer.transform(x)
        dist,ind = self.tree.query(x,k=k)
        weights = np.exp(-(dist/0.003)**2/2)
        weights = weights / np.max(weights)
        if return_traj:
            #去除重复轨迹的索引
            traj_ind = self.index[ind]
            _, unique_ind = np.unique(traj_ind, return_index=True)
            weights = weights[:,unique_ind]
            ind = ind[:,unique_ind]
            #根据索引取多个相似轨迹
            slice_ind = np.array([np.arange(s, e) for s, e in zip((ind + 1).T, (ind + step).T)])
            pred_samples = self.trajs[slice_ind,:2]
        else:
            pred_samples = self.trajs[ind+step]

        return pred_samples, weights

    def _cluster(self,sample,weight):
        cluster = DBSCAN(eps=3,min_samples=3,metric=lambda x,y: dtw(x.reshape((-1,2)),y.reshape((-1,2))).distance)
        cluster.fit(sample.reshape((sample.shape[0],-1)))
        return cluster.labels_

    # 混合分布估计
    def _estimate_cluster(self,sample,weight):
        cluster = DBSCAN(eps=1.5, min_samples=5)
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

    def predict_step(self, x, return_cluster=False):
        Y,W = self._query(x,100)
        mean = []
        for i,(Yi,Wi) in enumerate(zip(Y,W)):
            m,l = self._estimate_cluster(Yi, Wi)
            ind = i*np.ones(m.shape[0])
            m = np.column_stack((ind,m))
            mean.append(m)
        #return np.row_stack(mean)
        return Y.squeeze(),l

    def predict(self,x):
        Y, W = self._query(x,step=100,return_traj=True)
        return Y,self._cluster(Y,W)



if __name__=='__main__':
    import data
    import trajplot
    raw_trajs = data.load_files("H:/TrajectoryPrediction/adsb/PEK-SHA", usecols=[7, 8, 5],num=3000)
    traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
    test_data = np.array([traj_list[0][100]])
    #test_data = traj_list[0][98:100]
    predictor = Regressor3(traj_list)
    trajs,labels = predictor.predict(test_data)
    subplot = plt.subplot()
    colors = trajplot.label2color(labels)
    trajplot.plot2d(trajs,colors)
    print(1)

    '''
    dataset = data.policy_dataset(trajs)
    X = dataset[:, 2:6]
    Y = dataset[:, 8:10]

    yawing_regressor = Regressor()
    yawing_regressor.fit(X, Y)
    obs = np.array([117,37.29,51.83,-215.712])
    obs = np.expand_dims(X[308],0)
    #obs = X[14:16]
    #(ind, _), theta, w = yawing_regressor.predict(obs)
    Y,labels = yawing_regressor.predict_step(obs)
    subplot = plt.subplot()
    color_map = {-1:'k',0:'r', 1:'g', 2:'b',3:'y',4:'c',5:'lime',6:'gold',7:'yellow',8:'m',9:'pink'}
    #待完成：约束协方差
    color = [color_map[label] for label in labels]
    subplot.scatter(Y[:, 0], Y[:, 1], c=color)
    plt.show()
    print(1)
    '''