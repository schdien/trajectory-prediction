import numpy as np
from sklearn.cluster import DBSCAN
from dtw import dtw
from scipy import stats
from collections import deque
from scipy import ndimage


def func(trajs):
    outlier = [np.arange(len(traj)) for traj in trajs]
    for i in range(len(trajs)):
        for j in range(i+1,len(trajs)):
            dist = np.sum((trajs[i][outlier[i], np.newaxis, :] - trajs[j][np.newaxis, outlier[j], :]) ** 2, axis=-1)
            nearest_ind = np.argmin(dist, axis=1)
            outlier[j] = np.setdiff1d(outlier[j],np.unique(nearest_ind))

# ------------------------------------------------------------------------- #
class SequenceCluster:
    def __init__(self,eps,min_samples=2) -> None:
        def dtw_distance(x,y):
            x = x[x!=-1].reshape((-1,2))
            y = y[y!=-1].reshape((-1,2))
            return dtw(x,y).normalizedDistance
        self.cluster = DBSCAN(eps=eps,min_samples=min_samples,metric=dtw_distance)
        self.labels = None

    def fit(self,seqs):
        #填充trajs,让它们的长度相同
        padded_seqs = -1*np.ones((len(seqs),len(max(seqs, key=len)),seqs[0].shape[-1]))
        for i,seq in enumerate(seqs):
            padded_seqs[i,:len(seq)] = seq
        padded_seqs = padded_seqs[:,:,:2]
        self.labels = self.cluster.fit_predict(padded_seqs.reshape((padded_seqs.shape[0],-1)))

    def fit_predict(self,seqs):
        self.fit(seqs)
        return self.labels


# ------------------------------------------------------------------------- #
def mode_filter(x,r):
    l = len(x)
    filtered_x = np.zeros(l)
    for i in range(l):
        start = max(0,i-r)
        end = min(l,i+r+1)
        filtered_x[i] = stats.mode(x[start:end],keepdims=True)[0]
    return filtered_x


class PartitionalSequenceCluster:
    def __init__(self,normalizer,eps=0.003,min_samples=2) -> None:
        def distance(x,y):
            p_dist = np.linalg.norm(x[:2]-y[:2])
            v_dist = np.linalg.norm(x[2:]-y[2:])
            return max(p_dist,0.1*v_dist)
        self.cluster = DBSCAN(eps=eps,min_samples=min_samples,metric=distance)
        self.normalizer = normalizer
        self.labels = None
        self.sequences = None
        self.sample_indices = None

    def fit(self,seqs):
        ref_traj = seqs[0][:,:2]
        #轨迹同步采样
        self.sample_indices = np.array([dtw(ref_traj,traj[:,:2],step_pattern='asymmetric',open_begin=True,open_end=True).index2 for traj in seqs])
        self.sequences = np.array([traj[i] for traj,i in zip(seqs,self.sample_indices)])
        states = self.sequences.swapaxes(0,1)
        #聚类
        n_states = np.array([self.normalizer.transform(state) for state in states])
        labels = np.array([self.cluster.fit_predict(state) for state in n_states])
        #统计滤波器去噪
        self.labels = np.array([mode_filter(label,20) for label in labels.T])

    def fit_predict(self,seqs,time_first:bool):
        self.fit(seqs)
        if time_first:
            return self.sequences.swapaxes(0,1), self.labels.T
        else:
            return self.sequences, self.labels

class DBTPSCAN4:
    def __init__(self,trajs,eps,min_samples) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.trajs = trajs
        self.labels = [-np.ones(len(traj)) for traj in trajs]
        self.times = [np.arange(len(traj)) for traj in trajs]
        self.grouped =[np.empty(0) for _ in range(len(trajs))]
        self.curr_label = 0

    def get_ungrouped(self):
        for i,label in enumerate(self.labels):
            ungrouped_ts = np.where(label == -1)[0]
            if len(ungrouped_ts) != 0:
                return i,ungrouped_ts
            elif i==self.total_num-1:
                return None

    def modify_label(self,i,ts,c):
        self.labels[i][ts] = c
    
    def modify_time(self,i1,ts1,i2,ts2):
        self.times[i1][ts1] = self.times[i2][ts2]



    def noise_filter(self,ts):
        ts = np.unique(ts)
        a = ts[:-1] == ts[1:] - 1
        b = ts[1:] == ts[:-1] + 1
        a = np.append(a,False)
        b = np.append(False,b)
        return ts[a|b]

    def query(self,i,ts):
        neighbors_i = []
        neighbors_ts = []
        ref_traj = self.trajs[i][ts]
        for j,traj in enumerate(self.trajs):
            if i ==j:
                continue
            pos_dists = np.sqrt(np.sum((ref_traj[:, np.newaxis, :2] - traj[np.newaxis, :, :2]) ** 2, axis=-1))
            dir_dists = np.sqrt(np.sum(((ref_traj[:, 2:]/np.linalg.norm(ref_traj[:, 2:],axis=1,keepdims=True))[:,np.newaxis,:] - (traj[:, 2:]/np.linalg.norm(traj[:, 2:],axis=1,keepdims=True))[np.newaxis,:,:]) ** 2,axis=-1))
            dists = np.maximum(pos_dists*0.9,dir_dists)
            min_dist = np.min(dists, axis=1)
            neighbor_ts = np.argmin(dists, axis=1)
            neighbor_ts[min_dist>self.eps] = -1
            min_dist2 = np.min(dists, axis=0)
            neighbor_ts2 = np.argmin(dists, axis=0)
            neighbor_ts2[min_dist2>self.eps] = -1
            neighbors_i.append(j)
            neighbors_ts.append(neighbor_ts)
        return np.array(neighbors_i), np.array(neighbors_ts)

    def query2(self,i,mask):
        neighbors = []
        ref_traj = self.trajs[i][mask]
        neighbors_num = np.zeros(len(ref_traj))
        for j,traj in enumerate(self.trajs):
            if i ==j:
                continue
            pos_dists = np.sqrt(np.sum((ref_traj[:, np.newaxis, :2] - traj[np.newaxis, :, :2]) ** 2, axis=-1))
            dir_dists = np.sqrt(np.sum(((ref_traj[:, 2:]/np.linalg.norm(ref_traj[:, 2:],axis=1,keepdims=True))[:,np.newaxis,:] - (traj[:, 2:]/np.linalg.norm(traj[:, 2:],axis=1,keepdims=True))[np.newaxis,:,:]) ** 2,axis=-1))
            dists = np.maximum(pos_dists*0.9,dir_dists)
            neighbor_mask = np.any(dists<self.eps,axis=0)
            pair = np.argmin(dists, axis=0)
            neighbors.append((j,neighbor_mask,pair))
            neighbor_mask2 = np.any(dists<self.eps,axis=1)
            neighbors_num = neighbors_num + neighbor_mask2
        return neighbors, neighbors_num

    def build(self,i,mask):
        q = deque()
        q.append((i,mask))
        while q:
            i,mask = q.popleft()
            #先加入组？
            #查询当前轨迹与其余轨迹的相邻情况，-1表示该时间索引处与这条轨迹没有相邻点，！=-1的值表示相邻点的时间索引
            neighbors,neighbors_num = self.query2(i,mask)

            #分组
            for n_i,neighbor_mask,pair in neighbors:
                ungroup_mask = self.labels[n_i] == -1
                ungroup_neighbor_mask = neighbor_mask & ungroup_mask
                #未分组&邻居加入组
                if np.sum(ungroup_neighbor_mask) < 5:
                    continue
                self.labels[n_i][ungroup_neighbor_mask] = self.curr_label #修改标签

                #核心点邻居&未分组加入扩散队列
                core_mask = neighbors_num[pair] >= self.min_samples
                core_ungroup_neighbor_mask = ungroup_neighbor_mask & core_mask
                q.append((n_i,core_ungroup_neighbor_mask))



    def cluster(self):
        for i,label in enumerate(self.labels):
            if i == 4:
                break
            ungroup_mask = label == -1
            #ts = self.noise_filter(ts) #加上以后噪声点可能会被分配到其他轨迹组中
            if np.sum(ungroup_mask) != 0:
                #_,neighbors_ts = self.query(i,ts)
                #neighbor_num = np.count_nonzero(neighbors_ts!=-1,axis=0)
                #core_mask = neighbor_num>=self.min_samples
                #ts = ts[core_mask]
                self.labels[i][ungroup_mask] = self.curr_label #这个组的初始核心点，从这里开始扩散
                self.build(i,ungroup_mask)
                self.curr_label += 1


class DBTPSCAN2:
    def __init__(self,trajs,eps,min_samples) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.trajs = trajs
        self.labels = [-np.ones(len(traj)) for traj in trajs]
        self.times = [np.arange(len(traj)) for traj in trajs]
        #self.grouped =[np.empty(0) for _ in range(len(trajs))]
        self.curr_label = 0

    def modify_label(self,i,ts,c):
        self.labels[i][ts] = c

    def is_grouped(self,i,ts):
        grouped_ts = np.where(self.labels[i] != -1)[0]
        return np.isin(ts,grouped_ts)
    
    def modify_time(self,i1,ts1,i2,ts2):
        self.times[i1][ts1] = self.times[i2][ts2]

    def noise_filter(self,ts):
        ts = np.unique(ts)
        a = ts[:-1] == ts[1:] - 1
        b = ts[1:] == ts[:-1] + 1
        a = np.append(a,False)
        b = np.append(False,b)
        return ts[a|b]

    def query(self,i,ts):
        neighbors_i = []
        neighbors_ts = []
        ref_traj = self.trajs[i][ts]
        for j,traj in enumerate(self.trajs):
            if i ==j:
                continue
            pos_dists = np.sqrt(np.sum((ref_traj[:, np.newaxis, :2] - traj[np.newaxis, :, :2]) ** 2, axis=-1))
            dir_dists = np.sqrt(np.sum(((ref_traj[:, 2:]/np.linalg.norm(ref_traj[:, 2:],axis=1,keepdims=True))[:,np.newaxis,:] - (traj[:, 2:]/np.linalg.norm(traj[:, 2:],axis=1,keepdims=True))[np.newaxis,:,:]) ** 2,axis=-1))
            dists = np.maximum(pos_dists*0.9,dir_dists)
            min_dist = np.min(dists, axis=1)
            neighbor_ts = np.argmin(dists, axis=1)
            neighbor_ts[min_dist>self.eps] = -1
            neighbors_i.append(j)
            neighbors_ts.append(neighbor_ts)
        return np.array(neighbors_i), np.array(neighbors_ts)

    def cluster(self):
        for i,label in enumerate(self.labels):
            if i == 4:
                break
            ts = np.where(label == -1)[0]
            if len(ts) != 0:
                self.build(i,ts,i,ts)
                self.curr_label += 1

    def build(self,i,ts,u_i,u_ts):
        #查找未分组
        ungroup_masked = ~self.is_grouped(i,ts)
        ts = ts[ungroup_masked]
        if len(ts) == 0:
            return
        u_ts = u_ts[ungroup_masked]
        #查找未分组的点的邻居
        neighbors_i,neighbors_ts = self.query(i,ts)
        #判断是否为核心点，取出核心点邻居用于扩散。核心点：其他轨迹的邻居数>n，并且自身轨迹两侧都有m个索引相邻的点
        neighbor_num = np.count_nonzero(neighbors_ts!=-1,axis=0)
        core_mask = neighbor_num>5
        #加入上级组
        self.modify_label(i,ts,self.curr_label)
        self.modify_time(i,ts,u_i,u_ts)
        #核心点邻居扩散
        for n_i,n_ts in zip(neighbors_i,neighbors_ts):
            is_neighbor = n_ts!=-1
            valid = is_neighbor #& core_mask
            n_ts = n_ts[valid]
            match_ts = ts[valid]
            if len(n_ts)>5:
                self.build(n_i,n_ts,i,match_ts)


class DBTPSCAN3:
    def __init__(self,trajs,eps,min_samples) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.trajs = trajs
        self.labels = [-np.ones(len(traj)) for traj in trajs]
        self.curr_label = 0

    def modify_label(self,i,ts,c):
        self.labels[i][ts] = c

    def is_grouped(self,i,ts):
        grouped_ts = np.where(self.labels[i] != -1)[0]
        return np.isin(ts,grouped_ts)

    def query(self,i,ts):
        neighbors_i = []
        neighbors_ts = []
        ref_traj = self.trajs[i][ts]
        for j,traj in enumerate(self.trajs):
            if i ==j:
                continue
            pos_dists = np.sqrt(np.sum((ref_traj[:, np.newaxis, :2] - traj[np.newaxis, :, :2]) ** 2, axis=-1))
            dir_dists = np.sqrt(np.sum(((ref_traj[:, 2:]/np.linalg.norm(ref_traj[:, 2:],axis=1,keepdims=True))[:,np.newaxis,:] - (traj[:, 2:]/np.linalg.norm(traj[:, 2:],axis=1,keepdims=True))[np.newaxis,:,:]) ** 2,axis=-1))
            dists = np.maximum(pos_dists*0.9,dir_dists)
            min_dist = np.min(dists, axis=1)
            neighbor_ts = np.argmin(dists, axis=1)
            neighbor_ts[min_dist>self.eps] = -1
            neighbors_i.append(j)
            neighbors_ts.append(neighbor_ts)
        return np.array(neighbors_i), np.array(neighbors_ts)

    def cluster(self):
        for i,label in enumerate(self.labels):
            ts = np.where(label == -1)[0]
            if len(ts) != 0:
                self.build(i,ts)
                self.curr_label += 1

    def build(self,i,ts):
        #查找未分组
        ungroup_masked = ~self.is_grouped(i,ts)
        ts = ts[ungroup_masked]
        if len(ts) == 0:
            return
        #查找未分组的点的邻居
        neighbors_i,neighbors_ts = self.query(i,ts)
        #判断是否为核心点，取出核心点邻居用于扩散。核心点：其他轨迹的邻居数>n，并且自身轨迹两侧都有m个索引相邻的点
        neighbor_num = np.count_nonzero(neighbors_ts!=-1,axis=0)
        core_mask = neighbor_num>5
        #加入上级组
        self.modify_label(i,ts,self.curr_label)
        #核心点邻居扩散
        for n_i,n_ts in zip(neighbors_i,neighbors_ts):
            is_neighbor = n_ts!=-1
            valid = is_neighbor #& core_mask
            n_ts = n_ts[valid]
            if len(n_ts)>5:
                self.build(n_i,n_ts)

from sklearn.neighbors import KDTree
from threading import Event,Thread
import time


class DBTPSCAN:
    def __init__(self,trajs,eps,min_samples,min_len) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.min_len = min_len
        self.trajs_num = len(trajs)
        self.trajs = np.vstack([np.hstack([traj[:,:2],traj[:, 2:]/np.linalg.norm(traj[:, 2:],axis=1,keepdims=True)*1.1]) for traj in trajs])
        self.clustered_num = 0
        self.total_num = len(self.trajs)
        self.tree = KDTree(self.trajs)
        self.identifiers = np.hstack([np.ones(len(traj),dtype=int)*i for i,traj in enumerate(trajs)])
        self.ts = np.hstack([np.arange(len(traj),dtype=int) for traj in trajs])
        self.labels = -np.ones(self.total_num,dtype=int)
        self.core_mask = None
        self.ungroup_mask = np.ones(self.total_num,dtype=bool)
        self.neighbors = None
        self.sequence = {}
        self.labels2 = None

    def culculate_neighbors(self):
        forward_neighbors,_ = self.tree.query_radius(self.trajs,self.eps,return_distance=True,sort_results=True)
        for i,(neighbor,curr_id) in enumerate(zip(forward_neighbors,self.identifiers)):
            neighbor_ids = self.identifiers[neighbor] #查找邻居所在的轨迹id
            neighbor_ids,ind = np.unique(neighbor_ids,return_index=True)    #同一轨迹中的点只保留最近的
            ind = ind[neighbor_ids != curr_id]  #去除自身所在轨迹的点
            forward_neighbors[i] = neighbor[ind]
        
        backward_neighbors = [[] for _ in range(self.total_num)]
        for i,neighbor in enumerate(forward_neighbors):
            for n in neighbor:
                backward_neighbors[n].append(i)

        self.neighbors = np.array([np.array(neighbor) for neighbor in backward_neighbors],dtype=object)
        self.core_mask = np.array([len(neighbor) >= self.min_samples for neighbor in self.neighbors])
    
    def filter_split(self,identifier,mask):
        id_mask = self.identifiers == identifier
        mask = mask & id_mask
        if np.all(~mask):
            return []
        mask = ndimage.binary_closing(mask,mask=id_mask).astype(bool)
        idxs = np.where(mask)[0]
        diff_idxs = np.diff(idxs)
        split_i = np.nonzero(diff_idxs > 1)[0] + 1
        idxs = np.split(idxs,split_i)
        return list(filter(lambda x: len(x)>=self.min_len,idxs))
    
    def filter_split2(self,mask):
        if np.all(~mask):
            return []
        mask = ndimage.binary_closing(mask).astype(bool)
        idxs = np.where(mask)[0]
        diff_idxs = np.diff(idxs)
        split_i = np.nonzero(diff_idxs > 1)[0] + 1
        idxs = np.split(idxs,split_i)
        idxs = tuple(filter(lambda x: len(x)>=2,idxs))
        if len(idxs) == 0:
            return []
        return np.hstack(idxs)

    def build_cluster(self,init_idx,curr_label):
        #标记初始序列
        self.labels[init_idx] = curr_label 
        self.ungroup_mask[init_idx] = False 
        q = deque()
        q.append(init_idx)
        while q:
            idx = q.popleft()
            self.clustered_num += len(idx)
            idx = idx[self.core_mask[idx]] #取出核心点
            if len(idx) == 0:
                continue
            neighbor = np.hstack(self.neighbors[idx])
            neighbor = np.unique(neighbor)
            neighbor_mask = np.zeros(self.total_num, dtype=bool)
            neighbor_mask[neighbor] = True
            neighbor = self.filter_split2(neighbor_mask & self.ungroup_mask)
            if len(neighbor) == 0:
                continue
            #neighbor = neighbor[self.ungroup_mask[neighbor]]
            self.labels[neighbor] = curr_label #加入组
            self.ungroup_mask[neighbor] = False #标记为已分组
            q.append(neighbor)
            '''
            neighbor_mask = np.zeros(self.total_num, dtype=bool)
            neighbor_mask[neighbor] = True
            for id in range(self.trajs_num):
                neighbor_idxs = self.filter_split(id, neighbor_mask & self.ungroup_mask)
                if len(neighbor_idxs) == 0:
                    continue
                neighbor_idxs = np.hstack(neighbor_idxs)
                self.labels[neighbor_idxs] = curr_label #加入组
                self.ungroup_mask[neighbor_idxs] = False
                q.append(neighbor_idxs) #扩散
            '''

    def build_sequence(self,idx,label):
        self.sequence[label] = ([],[])
        s_idx = idx[0]
        previous_labels = self.labels[max(0,s_idx-self.min_len):s_idx]
        previous_labels = previous_labels[previous_labels!=-1]
        if len(previous_labels) != 0:
            previous_label = previous_labels[-1]
            self.sequence[previous_label][1].append(label)

        e_idx = idx[-1]
        next_labels = self.labels[e_idx+1:min(self.total_num,e_idx+self.min_len+1)]
        next_labels = next_labels[next_labels!=-1]
        if len(next_labels) != 0:
            next_label = next_labels[0]
            self.sequence[next_label][0].append(label)



    def cluster(self,print_progress=False):
        if print_progress:
            print_thread = Thread(target=self.print_progress)
            print_thread.start()

        self.culculate_neighbors()
        curr_label = 0
        for id in range(self.trajs_num):
            init_idxs = self.filter_split(id, self.core_mask & self.ungroup_mask)
            if init_idxs == []:
                continue
            for init_idx in init_idxs:
                self.build_sequence(init_idx,curr_label)
                self.build_cluster(init_idx,curr_label)
                curr_label += 1

        self.labels2 = [self.labels[self.identifiers==id] for id in range(self.trajs_num)]

    def print_progress(self):
        print("culculating neighbors...",end='\r')
        while self.neighbors is not None:
            pass
        print("culculating neighbors: done.")
        while self.labels2 is None:
            process = self.clustered_num*100/self.total_num
            print("clustering: %d%%" %process,end='\r')
            time.sleep(1)
        print("clustering: done.")


#plt.switch_backend('TkAgg')
#plt.ion()
if __name__ == '__main__':
    from src import data
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    raw_trajs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=100)
    traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
    subplot = plt.subplot()
    cluster = DBTPSCAN(traj_list,0.028,5,10)
    cluster.cluster(True)

    colors = ['white', 'red', 'green', 'blue', 'indigo', 'orange', 'm']
    cmap_name = 'cyclic_cmap'
    n = len(colors)
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n)

    for traj,label in zip(traj_list,cluster.labels2):
        color = [cmap(l+1) for l in label]
        subplot.scatter(traj[:,0],traj[:,1],color = color,s=4)
    plt.show()