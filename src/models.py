import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from dtw import dtw
import matplotlib.pyplot as plt
from math import pi

def velocity_model(p, v, dt):
    [lon, lat] = p.T
    d = v * dt
    c = 180 / (pi * 6371393)
    _lon = lon + c * d[:,0] / np.cos(pi * lat / 180)
    _lat = lat + c * d[:,1]
    return np.column_stack([_lon, _lat])

class Node:
    def __init__(self,data=None):
        self.data = data
        self.children = None

    def plot(self):
        subplot = plt.subplot()
        nodes = [self]
        for node in nodes:
            subplot.plot(node.data[:, 0], node.data[:, 1], marker='')
            if node.children is not None:
                nodes += node.children
        plt.show()

class TrajPredictor:
    def __init__(self,traj_list):
        self.trajs = np.vstack(traj_list)
        self.identifiers = np.concatenate([i * np.ones(len(traj),dtype=int) for i,traj in enumerate(traj_list)])
        self.edges = np.cumsum(np.array([len(traj) for traj in traj_list]))
        self.normalizer = MinMaxScaler()
        normalized_trajs = self.normalizer.fit_transform(self.trajs)
        self.tree = BallTree(normalized_trajs)


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

    def _query_point(self,x,step,k=100):
        x = self.normalizer.transform(x)
        dist,ind = self.tree.query(x,k=k)
        weights = np.exp(-(dist/0.003)**2/2)
        weights = weights / np.max(weights)
        pred_ind = self._get_pred_index(ind,step,'edge')
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
        return prob, mean, label
    
    def predict_step(self, x, dt=10):
        ps = x[:,:2]
        points,weights = self._query_point(x,1)
        preds = []
        for i,(p,v,w) in enumerate(zip(ps,points[:,:,2:],weights)):
            prob,pred_v,label = self._cluster_point(v, w,1.5,5)
            _,repeats = np.unique(pred_v[:,0],return_counts=True)
            p = np.repeat(p,repeats,axis=0)
            pred_p = velocity_model(p,pred_v,dt)
            ind = i*np.ones(pred_v.shape[0])
            preds.append(np.column_stack((ind,prob,pred_p,pred_v)))
        return np.vstack(preds)
        #return Y.squeeze(),l
    
    def predict(self,obs,n_steps,dt=10):
        head = Node(obs)
        nodes = [head]
        for i in range(n_steps):
            for node in nodes[:]:
                obs = node.data[None,-1,:]
                obs = self.predict_step(obs)[:,2:]
                if len(obs) == 1:
                    node.data = np.append(node.data,obs,axis=0)
                else:
                    node.children = [Node(obsi[None, :]) for obsi in obs]
                    nodes.remove(node)
                    nodes += node.children
        return head


if __name__=='__main__':
    import data
    import trajplot
    raw_trajs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=3000)
    traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]

    predictor = TrajPredictor(traj_list)

    #单步预测
    test_data = traj_list[0][50:150]
    pred_result = predictor.predict_step(test_data)[:,2:4]
    trajplot.plot2d([test_data,pred_result])

    #多步预测
    test_data = np.array([traj_list[0][50]])
    pred_result = predictor.predict(test_data,100)
    pred_result.plot()

