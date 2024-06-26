import os
import numpy as np
from scipy import interpolate
from math import pi

#读取文件夹
def load_files(parent_dir, usecols=[7,8,5], num=None):
    '''
    :param parent_dir:           csv文件的上级路径
    :param num:                 读取csv文件的个数，例如10表示只读取前10个csv文件
    :return:                    list[ndarray]
    '''

    # 获得文件夹下的csv路径列表
    paths = []
    for abs_dirs, _, file_names in os.walk(parent_dir):
        paths += list(map(lambda x: abs_dirs + "/" + x, file_names))
    if num is not None:
        paths = paths[:num]

    # 读取每一个csv文件
    trajs = [load_file(path,usecols) for path in paths]
    return trajs

def load_file(path, usecols):
    return np.loadtxt(path, skiprows=1, delimiter=",", usecols=usecols)

#预处理
def preprocess(traj):
    traj = unique_by_columns(traj, (0, 1))
    traj = cut_by_column_low_value(traj,2)
    pos = traj[:,:2]
    v = traj[:, 2] / 3.6
    t = extract_time(pos, v)
    pos = resample(pos, t, 10)
    vtheta = extract_velocity_yawing(pos, 10)
    traj = np.column_stack([pos[:-1], vtheta])
    return traj

def preprocess2(traj):
    traj = unique_by_columns(traj, (0, 1))
    traj = cut_by_column_low_value(traj,2)
    pos = traj[:,:2]
    v = traj[:, 2] / 3.6
    t = extract_time(pos, v)
    pos = resample(pos, t, 10)
    v = extract_velocity(pos, 10)
    traj = np.column_stack([pos[:-1], v])
    return traj

#数据集
def acceleration_policy_dataset(trajs,is_stack=True):
    dataset = []
    for traj in trajs:
        v = traj[:,3:5]
        a = extract_acceleration(v, 10)
        data = np.column_stack([traj[:-1], a])
        dataset.append(data)
    if is_stack:
        dataset = np.row_stack(dataset)
    return dataset

def velocity_policy_dataset(trajs,is_stack=True):
    dataset = []
    for traj in trajs:
        pos = traj[:,:2]
        v = traj[:, 2:4]
        next_v = v[1:]
        curr_v = v[:-1]
        policy = np.column_stack([pos[:-1], curr_v, next_v])
        dataset.append(policy)
    if is_stack:
        dataset = np.row_stack(dataset)
    return dataset

def policy_dataset(trajs,is_stack=True):
    dataset = []
    for i,traj in enumerate(trajs):
        len = len(traj)
        traj_ind = i*np.ones(len)
        step_ind = np.arange(0,len)
        next_traj = traj[1:]
        curr_traj = traj[:-1]
        policy = np.column_stack([traj_ind,step_ind,curr_traj, next_traj])
        dataset.append(policy)
    if is_stack:
        dataset = np.row_stack(dataset)
    return dataset


#去除重复值
def unique_by_columns(traj,cols=None):
    if cols is None:
        a = traj
    else:
        a = traj[:, cols]
    a, index = np.unique(a, axis=0, return_index=True)
    traj = traj.take(sorted(index),axis=0)
    return traj


#在速度很小时视为到达，截去后续轨迹点
def cut_by_column_low_value(traj,col,thold=18):
    l = len(traj)
    hl = 200
    arr = traj[:,col]
    low_pos = np.where(arr < thold)[0]
    start_pos = low_pos[low_pos < hl]
    end_pos = low_pos[low_pos > hl]

    if len(start_pos) == 0:
        start_pos = 0
    else:
        start_pos = start_pos[-1]+1
    if len(end_pos) == 0:
        end_pos = l+1
    else:
        end_pos = end_pos[0]

    return traj[start_pos:end_pos]


#重采样
def resample(pos, t, new_dt, kind='quadratic'):
    f = interpolate.interp1d(t, pos.T, kind=kind)
    t = np.arange(t[0], t[-1], step=new_dt)
    pos = f(t).T
    return pos


#两点的水平距离
def horizontal_displacement(pos):
    lat = pos[:-1,1]
    d_pos = np.diff(pos,axis=0)
    d_lon = d_pos[:, 0]
    d_lat = d_pos[:, 1]
    c = 180 / (pi * 6371393)
    dx = d_lon*np.cos(pi*lat/180)/c
    dy = d_lat/c
    return np.column_stack([dx, dy])


#重建时间
def extract_time(pos,vmag):
    d = horizontal_displacement(pos)
    d = np.sqrt(np.square(d[:, 0]) + np.square(d[:, 1]))
    dt = d / vmag[:-1]
    t = np.cumsum(dt)
    return np.concatenate([np.array([0]),t])

#提取速度矢量
def extract_velocity(pos,dt):
    d = horizontal_displacement(pos)
    return d/dt

#提取水平速度大小和水平航向角
def extract_velocity_yawing(pos, dt):
    d = horizontal_displacement(pos)
    #计算轴向速度（标量）
    mag = np.sqrt(np.square(d[:, 0]) + np.square(d[:, 1])) / dt
    #计算偏航角
    dir = np.mod(np.arctan2(*d.T),2*pi)
    return np.column_stack([mag, dir])

#提取水平加速度和水平转弯率
def extract_acceleration(v,dt):
    vmag = v[:, 0]
    vdir = v[:, 1]

    # 处理超过2pi后突变的角度,避免转弯率突变
    d_theta = np.diff(vdir)
    #d_theta = np.where(d_theta > pi,d_theta,d_theta-2*pi)
    #d_theta = np.where(d_theta < -pi, d_theta, d_theta + 2 * pi)
    overflows = np.where(abs(d_theta) > pi)[0]
    for overflow in overflows:
        if d_theta[overflow] >= 0:
            vdir[overflow + 1:] -= 2 * pi
        else:
            vdir[overflow + 1:] += 2 * pi

    amag = np.diff(vmag, axis=0) / dt
    adir = np.diff(vdir, axis=0) / dt
    return np.column_stack([amag, adir])


