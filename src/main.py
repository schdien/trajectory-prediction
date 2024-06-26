from models import Regressor
import matplotlib.pyplot as plt
import trajplot
import kinematics
import numpy as np
import data

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
    def __init__(self):
        self.velocity_predictor = Regressor()
        self.physical_model = kinematics.velocity_model

    def train(self,OBS,V):
        self.velocity_predictor.fit(OBS,V)

    def _culculate_position(self,p,v,dt):
        _,repeats = np.unique(v[:,0],return_counts=True)
        p = np.repeat(p,repeats,axis=0)
        p = kinematics.velocity_model(p,v[:,2:],dt)
        return np.column_stack((v[:,:2],p))

    def predict(self,obs,dt=10):
        p = obs[:,:2]
        pred_v = self.velocity_predictor.predict_step(obs)
        pred_p = self._culculate_position(p, pred_v, dt)
        return pred_p

    def multi_step_predict(self,obs,n_steps,dt=10):
        head = Node(obs)
        nodes = [head]
        for i in range(n_steps):
            for node in nodes[:]:
                obs = node.data[None,-1,:]
                p = obs[:,:2]
                v = self.velocity_predictor.predict_step(obs)
                p = self._culculate_position(p,v,dt)
                obs = np.column_stack((p[:,2:], v[:,2:]))
                if len(obs) == 1:
                    node.data = np.append(node.data,obs,axis=0)
                else:
                    node.children = [Node(obsi[None, :]) for obsi in obs]
                    nodes.remove(node)
                    nodes += node.children
        return head




raw_trajs = data.load_files("H:/TrajectoryPrediction/adsb/PEK-SHA", usecols=[7, 8, 5], num=3000)
trajs = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
dataset = data.velocity_policy_dataset(trajs)
OBS = dataset[:, :4]
V = dataset[:, 4:]

predictor = TrajPredictor()
predictor.train(OBS,V)
obs = np.array([OBS[300]])
p = OBS[:500]
multi_pred_p = predictor.multi_step_predict(obs,200)
multi_pred_p.plot()
