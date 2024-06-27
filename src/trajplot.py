import matplotlib.pyplot as plt
import numpy as np


def label2color(labels):
    color_map = {-1:'k',0:'#1f77b4',1: '#ff7f0e',2: '#2ca02c',3: '#d62728',4: '#9467bd',
                5: '#8c564b',6: '#e377c2',7: '#7f7f7f', 8:'#bcbd22',9: '#17becf',
                10: '#172965',11: '#12b5cf',12:'k',13:'#1f77b4',14: '#ff7f0e',15: '#2ca02c',16: '#d62728',17: '#9467bd',
                18: '#8c564b',19: '#e377c2',20: '#7f7f7f', 21:'#bcbd22',22: '#17becf',
                23: '#172965',24: '#12b5cf'}
    return [color_map[label] for label in labels]

def scatter2d(points,colors=None):
    subplot = plt.subplot()
    subplot.scatter(points[:, 0], points[:, 1],c=colors)
    plt.show()

def plot2d(trajs,colors=None):
    subplot = plt.subplot()
    if colors is None:
        for traj in trajs:
            subplot.plot(traj[:, 0], traj[:, 1],marker='.')
    else:
        for traj,color in zip(trajs,colors):
            subplot.plot(traj[:, 0], traj[:, 1],marker='.',c=color)
    plt.show()

def plot3d(trajs):
    subplot = plt.subplot(projection='3d')
    for traj in trajs:
        subplot.plot(traj[:, 0], traj[:, 1], traj[:, 2],marker='.',ms=1,linestyle='')
    plt.show()



if __name__ == '__main__':
    from data import load_files,preprocess
    traj_list = load_files(r"assets/PEK-SHA/MU5122", usecols=[7, 8, 5], num=200)
    traj_list = [preprocess(traj) for traj in traj_list]
    plot2d(traj_list)



