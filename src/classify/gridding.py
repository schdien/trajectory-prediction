import torch
import numpy as np


def get_grid_info_by_len(stacked_traj, grid_len):
    grid_len = np.array(grid_len)
    min_obs = np.min(stacked_traj, axis=0)-0.5
    max_obs = np.max(stacked_traj, axis=0)+0.5
    # 计算格点数
    capacities = np.ceil((max_obs - min_obs) / grid_len).astype(int)

    return grid_len, capacities, min_obs, max_obs

def get_grid_info_by_capacity(stacked_traj, capacity, is_squre):
    min_obs = np.min(stacked_traj, axis=0)
    max_obs = np.max(stacked_traj, axis=0)
    grid_len = (max_obs - min_obs)/capacity
    if is_squre:
        pass
        #grid_len[:] = min(grid_len)
    return grid_len,capacity, min_obs, max_obs

def culculate_grid_point_position(obs,min_obs,grid_len):
    grid_pos = ((obs - min_obs) // grid_len)
    if type(obs) == torch.Tensor:
        grid_pos = grid_pos.int()
    else:
        grid_pos = grid_pos.astype(int)
    return grid_pos
