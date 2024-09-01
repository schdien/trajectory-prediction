import numpy as np
from sklearn.cluster import DBSCAN
from dtw import dtw
from scipy import stats


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
        

class DBTPSCAN:
    def __init__(self,trajs,eps,min_samples) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.trajs = trajs
        self.labels = [np.column_stack([np.ones(len(traj))*i,np.arange(len(traj))]) for i,traj in enumerate(trajs)]
        self.grouped =[np.empty(0) for _ in range(len(trajs))]

    def get_ungrouped(self):
        for i,(g,traj) in enumerate(zip(self.grouped,self.trajs)):
            full = np.arange(len(traj))
            ungrouped = full[~np.isin(full,g)]
            if len(ungrouped) != 0:
                return i,ungrouped
            elif i==len(self.trajs)-1:
                return None

    def noise_filter(self,ts):
        ts = np.unique(ts)
        a = ts[:-1] == ts[1:] - 1
        b = ts[1:] == ts[:-1] + 1
        a = np.append(a,False)
        b = np.append(False,b)
        return ts[a|b]

    def append_to_grouped(self,i,ts):
        self.grouped[i] = np.concatenate([self.grouped[i],ts])

    def is_grouped(self,i,ts):
        return np.isin(ts,self.grouped[i])

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

    def build(self,i,ts):
        #查询当前轨迹与其余轨迹的相邻情况，-1表示该时间索引处与这条轨迹没有相邻点，！=-1的值表示相邻点的时间索引
        neighbors_i,neighbors_ts = self.query(i,ts)
        is_ungrouped_neighbor = []
        #分组
        for n_i,n_ts in zip(neighbors_i,neighbors_ts):
            is_grouped = self.is_grouped(n_i,n_ts) #已分组？
            is_neighbor = n_ts!=-1  #相邻？
            mask = (~is_grouped) & is_neighbor #有效时间索引：未分组且相邻
            self.append_to_grouped(n_i,n_ts[mask]) #标记为已分组
            self.labels[n_i][n_ts[mask]] = self.labels[i][ts[mask]]  #加入组中
            is_ungrouped_neighbor.append(mask)  #储存有效时间索引为下一步核心点扩散使用

        #查找核心点核心点：其他轨迹的邻居数>n，并且自身轨迹两侧都有m个索引相邻的点
        neighbor_num = np.count_nonzero(neighbors_ts!=-1,axis=0)
        is_core = neighbor_num>=self.min_samples
        #核心点的未分组邻居扩散
        for n_i,n_ts,mask in zip(neighbors_i,neighbors_ts,is_ungrouped_neighbor):
            mask = mask & is_core
            n_ts = n_ts[mask]
            if len(n_ts)>5:
                n_ts = self.noise_filter(n_ts)
                self.build(n_i,n_ts)

        #要拆分循环，使第一次的邻居全部分组完成


    def cluster(self):
        while True:
            ungrouped = self.get_ungrouped()
            if ungrouped is None:
                break
            self.append_to_grouped(*ungrouped)
            self.build(*ungrouped)
'''
    def get_ungrouped2(self):
        for i,label in enumerate(self.labels):
            ungrouped = np.where(label[:,0] == -1)[0]
            if len(ungrouped) == 0 and i==len(self.trajs)-1:
                return i,None
            if len(ungrouped) == 0:
                continue
            return i,ungrouped
    def cluster2(self):
        while True:
            i,idxs = self.get_ungrouped()
            if idxs is None:
                break
            self.build2(i,idxs,i,idxs)

    def build2(self,i,idxs,u_i,u_idxs):
        #查找未分组
        ungrouped_idxs = np.where(self.labels[i][:,0] == -1)[0]
        if len(ungrouped_idxs) == 0:
            return
        is_ungrouped = np.isin(idxs,ungrouped_idxs)
        idxs = idxs[is_ungrouped]
        u_idxs = u_idxs[is_ungrouped]
        #加入上级组
        self.mark_grouped(i,idxs)
        self.labels[i][idxs] = self.labels[u_i][u_idxs]
        #查找未分组的点的邻居
        neighbors_i,neighbors_idxs = self.query(i,idxs,0.03)
        #判断是否为核心点，取出核心点邻居用于扩散。核心点：其他轨迹的邻居数>n，并且自身轨迹两侧都有m个索引相邻的点
        neighbor_num = np.count_nonzero(neighbors_idxs!=-1,axis=0)
        is_core = neighbor_num>5
        #核心点邻居扩散
        for n_i,n_idxs in zip(neighbors_i,neighbors_idxs):
            is_neighbor = n_idxs!=-1
            valid = is_neighbor & is_core
            n_idxs = n_idxs[valid]
            match_idxs = idxs[valid]
            if len(n_idxs)>5:
                self.build2(n_i,n_idxs,i,match_idxs)



    def build(self,i,idxs):
        #查询邻居
        neighbors_i,neighbors_idxs = self.query(i,idxs,0.03)
        #判断是否为核心点，取出核心点的邻居。核心点：其他轨迹的邻居数>n，并且自身轨迹两侧都有m个索引相邻的点
        neighbor_num = np.count_nonzero(neighbors_idxs!=-1,axis=0)
        core_neighbors_idxs = neighbors_idxs[:,np.where(neighbor_num>5)[0]] #核心点的邻居
        core_neighbors_idxs = [core_neighbor_idxs[core_neighbor_idxs!=-1] for core_neighbor_idxs in core_neighbors_idxs]
        ungrouped_idxs = [np.where(self.labels[neighbor_i][:,0] == -1)[0] for neighbor_i in neighbors_i]

        ungrouped_neighbors_idxs = [] #这一部分加入组中
        match_idxs = []
        for n_idxs,ug_idxs in zip(neighbors_idxs,ungrouped_idxs):
            ug_n_idxs,pos,_ = np.intersect1d(n_idxs,ug_idxs,return_indices=True) #bug/
            ungrouped_neighbors_idxs.append(ug_n_idxs)
            match_idxs.append(idxs[pos])

        ungrouped_core_neighbors_idxs = [np.intersect1d(x,y) for x,y in zip(core_neighbors_idxs,ungrouped_neighbors_idxs)] #这一部分在加入组后扩散
        
        #分组
        for n_i,n_idxs,m_idxs in zip(neighbors_i,ungrouped_neighbors_idxs,match_idxs):
            if len(n_idxs!=0):
                self.labels[n_i][n_idxs] = self.labels[i][m_idxs]
                #subplot.scatter(self.trajs[n_i][n_idxs,0],self.trajs[n_i][n_idxs,1])
                #plt.draw()

        #扩散
        for n_i,ug_c_n_idxs in zip(neighbors_i,ungrouped_core_neighbors_idxs):
            l = self.labels[n_i][ug_c_n_idxs]
            if len(ug_c_n_idxs)>5:
                self.build(n_i,ug_c_n_idxs)

def f1(trajs,eps):
    #初始化标签每个点的标签组成,第一列为类别标签，第二列为顺序索引
    #-1：离群值
    labels = [np.column_stack([np.ones(len(traj))*i,np.arange(len(traj))]) for i,traj in enumerate(trajs)]
    l = len(trajs)
    for i in range(l):
        for j in range(i,l-1):
            classified_ind = np.where(labels[j][:,0]==i)[0]
            classified = trajs[j][classified_ind]
            if len(classified) == 0:
                continue
            for k in range(i+1,l):
                if j==k:
                    continue
                unclassified_ind = np.where(labels[k][:,0]>i)[0]
                unclassified = trajs[k][unclassified_ind]
                if len(unclassified) == 0:
                    continue
                pos_dists = np.sqrt(np.sum((classified[:, np.newaxis, :2] - unclassified[np.newaxis, :, :2]) ** 2, axis=-1))
                dir_dists = np.sqrt(np.sum(((classified[:, 2:]/np.linalg.norm(classified[:, 2:],axis=1,keepdims=True))[:,np.newaxis,:] - (unclassified[:, 2:]/np.linalg.norm(unclassified[:, 2:],axis=1,keepdims=True))[np.newaxis,:,:]) ** 2,axis=-1))
                dists = pos_dists*0.9 + dir_dists
                #dists = np.maximum(pos_dists*0.9,dir_dists)
                min_dist = np.min(dists, axis=0)
                match_inds1 = np.where(min_dist<=eps)[0]
                match_inds2 = np.argmin(dists, axis=0)[match_inds1]
                match_inds1 = unclassified_ind[match_inds1]
                match_inds2 = classified_ind[match_inds2]
                labels[k][match_inds1] = labels[j][match_inds2]
    return labels
'''
#plt.switch_backend('TkAgg')
#plt.ion()
if __name__ == '__main__':
    from src import data
    import matplotlib.pyplot as plt
    raw_trajs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=60)
    traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
    subplot = plt.subplot()
    cluster = DBTPSCAN(traj_list,0.06,5)
    cluster.cluster()

    color_tab = {0: 'r',1: 'g',2: 'b',3: 'y',4: 'c',5: 'tan',6: 'm',7: 'pink',8: 'peru',9: 'gray',10: '#8A2BE2',11: '#A52A2A',12: '#DEB887',13: '#5F9EA0',14: '#7FFF00',15: '#D2691E'}
    for traj,label in zip(traj_list,cluster.labels[:15]):
        color = [color_tab[l] for l in label[:,0]]
        subplot.scatter(traj[:,0],traj[:,1],color = color,s=4)
    plt.show()