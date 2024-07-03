import numpy as np
import data
from neighbors import SequenceQuerier
from cluster import PartitionalSequenceCluster, SequenceCluster, estimate_mean
import matplotlib.pyplot as plt


#读取和预处理数据
raw_seqs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=3000)
traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_seqs]
truth = traj_list[0][:350]
start_state = np.array([truth[100]])


#轨迹聚类
querier = SequenceQuerier(traj_list)
cluster = SequenceCluster(0.02)
neighbor_trajs, weights = querier.query(start_state,250,'trajectory')
labels = cluster.fit_predict(neighbor_trajs)

#画图
color_map = {-1:'k',0:'#1f77b4',1: '#ff7f0e',2: '#2ca02c',3: '#d62728',4: '#9467bd',
            5: '#8c564b',6: '#e377c2',7: '#7f7f7f', 8:'#bcbd22',9: '#17becf'}
colors = [color_map[label] for label in labels]
subplot = plt.subplot()
for traj,color in zip(neighbor_trajs,colors):
    subplot.plot(traj[:, 0], traj[:, 1],marker='.',c=color)
plt.show()


#轨迹分割聚类+预测
querier = SequenceQuerier(traj_list)
cluster = PartitionalSequenceCluster(querier.normalizer)
neighbor_trajs, weights = querier.query(start_state,250,'trajectory')
states, labels = cluster.fit_predict(neighbor_trajs,True)
pred_traj = estimate_mean(states,labels,weights)

#画图
subplot = plt.subplot()
subplot.plot(truth[:,0],truth[:,1],marker='.')
for stage in pred_traj:
    for traj in stage:
        subplot.plot(traj[:, 2], traj[:, 3],marker='.')
plt.show()