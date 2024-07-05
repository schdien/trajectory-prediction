from torch.utils.data.dataset import Dataset
import torch
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self,trajs_list):
        self.data = sum(trajs_list,[])
        self.labels = torch.eye(len(trajs_list))
        nums = np.array([len(trajs) for trajs in trajs_list])
        self.edges = np.cumsum(nums)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        traj = torch.tensor(self.data[item],dtype=torch.float32)
        ind = np.searchsorted(self.edges,item)
        label = self.labels[ind]
        label = label.unsqueeze(0).repeat(len(traj),1)
        return traj,label
