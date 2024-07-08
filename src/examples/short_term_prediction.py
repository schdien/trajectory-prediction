import numpy as np
from src import data
from src.neighbors import SequenceQuerier
from sklearn.cluster import DBSCAN
from src.estimate import estimate_mean
import matplotlib.pyplot as plt
from src import accuracy
from math import pi


#-------------------------------------------数据读取和预处理---------------------------------------------#
raw_trajs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=3000)
trajs = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
test_traj = trajs[0][:-1]            #测试轨迹
truth = trajs[0][1:,:2]


#---------------------------------------------短期预测(10秒)---------------------------------------------#
querier = SequenceQuerier(trajs[1:])  #从训练数据中剔除测试轨迹
cluster = DBSCAN(0.03)
neighbor_states_seq, weights_seq = querier.query(test_traj,1,'state')
pred_traj = []
for neighbor_states,weights in zip(neighbor_states_seq, weights_seq):
    n_neighbor_states = querier.normalizer.transform(neighbor_states)
    labels = cluster.fit_predict(n_neighbor_states,sample_weight=weights)
    pred_state = estimate_mean(neighbor_states,labels,weights,return_max_prob=True)
    pred_traj.append(pred_state)
pred_traj = np.array(pred_traj)
pred_v = pred_traj[:,2:] 
def velocity_model(p, v, dt):
    [lon, lat] = p.T
    d = v * dt
    c = 180 / (pi * 6371393)
    _lon = lon + c * d[:,0] / np.cos(pi * lat / 180)
    _lat = lat + c * d[:,1]
    return np.column_stack([_lon, _lat])
pred_pos = velocity_model(test_traj[:,:2],pred_v,10) #使用预测的速度计算下一时刻位置，比直接取下一时刻位置的方法精度更高。


#------------------------------------------------精度----------------------------------------------------#
euclidean_error = accuracy.euclidean_distance(truth,pred_pos).mean()
cross_track_error = accuracy.cross_track_distance(truth,pred_pos).mean()
print('mean euclidean error:',euclidean_error,'m.\nmean cross track error:',cross_track_error,'m.')


#------------------------------------------------画图----------------------------------------------------#
subplot = plt.subplot()
subplot.plot(truth[:,0],truth[:,1],marker='.',label='ground truth')
subplot.plot(pred_pos[:,0],pred_pos[:,1],marker='.',label='prediction')
plt.legend()
plt.show()