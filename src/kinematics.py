import numpy as np
from math import pi


def haversine(lon1, lat1, lon2, lat2):
    # 将角度转化为弧度
    lon1, lat1, lon2, lat2 = map(lambda x: pi*x/180, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371393 # 地球平均半径，单位为米
    return c * r

def velocity_model(p, v, dt):
    [lon, lat] = p.T
    d = v * dt
    c = 180 / (pi * 6371393)
    _lon = lon + c * d[:,0] / np.cos(pi * lat / 180)
    _lat = lat + c * d[:,1]
    return np.column_stack([_lon, _lat])

def velocity_yaw_model(pos, vtheta, dt):
    [v,theta] = vtheta
    d = v*dt
    dy = d * np.cos(theta)
    dx = d * np.sin(theta)

    [lon,lat] = pos
    c = 180 / (pi * 6371393)
    _lon = lon + c * dx / np.cos(pi * lat / 180)
    _lat = lat + c * dy

    return np.column_stack([_lon, _lat]).squeeze()

def accelaration_turnrate_model(h_pos, v, a, dt):
    _v = v + a * dt
    vmag = _v[:, 0]
    vdir = _v[:, 1]
    d = vmag * dt
    dy = d * np.cos(vdir)
    dx = d * np.sin(vdir)
    lon = h_pos[:, 0]
    lat = h_pos[:, 1]
    c = 180 / (pi * 6371393)

    _lon = lon + c * dx / np.cos(pi * lat / 180)
    _lat = lat + c * dy
    return np.column_stack([_lon, _lat]), _v