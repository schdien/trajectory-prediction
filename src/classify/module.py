import torch
from torch import nn
from classify import gridding

class GridEncoder(nn.Module):#不同特征同时提取
    def __init__(self, grid_info, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.grid_len, grid_capacity, self.min_obs, max_obs = grid_info

        if len(embedding_dim) != len(grid_capacity):
            raise ValueError("Expect feature number and traj_embedding dimension number at the same length.")

        self.embedding = []
        for grid_capacity,embedding_dim in zip(grid_capacity, embedding_dim):
            embedding = nn.Embedding(grid_capacity,embedding_dim)
            self.embedding.append(embedding)

    def forward(self, obs):
        # 将连续的输入网格化
        grid_obss = gridding.culculate_grid_point_position(obs, self.min_obs, self.grid_len)

        # 将网格化的输入转化为embendding向量
        output = torch.tensor(())
        for grid_obs,embedding in zip(grid_obss.T, self.embedding):
            output = torch.concatenate([output,embedding(grid_obs)],1)
        return output


class SeqentialClassifier(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super().__init__()
        self.is_full_output = True
        self.latent_size = out_size
        self.rnn = nn.GRU(in_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax(1)

    def forward(self,x):
        hs, _ = self.rnn(x)
        if self.is_full_output:
            x = hs
        else:
            x = hs[-1]
        x = self.fc(x)
        x = self.softmax(x)
        return x