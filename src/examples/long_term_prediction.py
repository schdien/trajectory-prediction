import numpy as np
from src import data
from src.neighbors import SequenceQuerier
from src.cluster import PartitionalSequenceCluster, SequenceCluster
from src.estimate import estimate_sequence_mean
import matplotlib.pyplot as plt
from src import accuracy


#-------------------------------------------数据读取和预处理---------------------------------------------#
raw_seqs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=3000)
traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_seqs]
test_traj = traj_list[0][:350]              #测试轨迹
start_state = np.array([test_traj[100]])    #开始预测位置点
truth = test_traj[100:,:2]


#----------------------------------------分段聚类和长期预测(2500秒)---------------------------------------#
querier = SequenceQuerier(traj_list[1:])  #从训练数据中剔除测试轨迹
cluster = PartitionalSequenceCluster(querier.normalizer)
neighbor_trajs, weights = querier.query(start_state,250,'trajectory')
states, labels = cluster.fit_predict(neighbor_trajs,True)
pred_traj = estimate_sequence_mean(states,labels,np.ones_like(weights))


#------------------------------------------------精度----------------------------------------------------#
euclidean_error = accuracy.probabilistic_accuracy(truth,pred_traj,'euclidean').mean()
cross_track_error = accuracy.probabilistic_accuracy(truth,pred_traj,'cross track').mean()
dtw_error = accuracy.probabilistic_accuracy(truth,pred_traj,'dtw')
print('mean euclidean error:',euclidean_error,'m\nmean cross track error:',cross_track_error,'m\ndtw error:',dtw_error,)


#------------------------------------------------画图----------------------------------------------------#
#将标签转换为颜色
def label2color(labels):
    color_map = {-1:'k',0:'#1f7156',1: '#ff7f0e',2: '#2ca02c',3: '#d62728',4: '#9467bd',
                5: '#8c564b',6: '#e377c2',7: '#7f7f7f', 8:'#bcbd22',9: '#17becf'}
    return [color_map[label] for label in labels]
colors = [label2color(label) for label in labels]

#轨迹聚类结果
subplot = plt.subplot(1,2,1)
subplot.plot(test_traj[:,0],test_traj[:,1],marker='.',label='ground truth')
for state,color in zip(states,colors):
    subplot.scatter(state[:,0],state[:,1],c=color)
plt.legend()

#预测轨迹
subplot = plt.subplot(1,2,2)
subplot.plot(test_traj[:,0],test_traj[:,1],marker='.',label='ground truth')
for stage in pred_traj:
    for traj in stage:
        subplot.plot(traj[:, 2], traj[:, 3],marker='.')
plt.legend()
plt.show()


'''
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
'''