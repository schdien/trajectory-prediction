import numpy as np
from math import pi
from dtw import dtw


def haversine(lon1, lat1, lon2, lat2):
    # 将角度转化为弧度
    lon1, lat1, lon2, lat2 = map(lambda x: pi*x/180, [lon1, lat1, lon2, lat2])
    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371393 # 地球平均半径，单位米
    return c * r


def euclidean_distance(truth, pred):
    return haversine(truth[:,0],truth[:,1],pred[:,0],pred[:,1])


def dtw_distance(truth, pred):
    return dtw(truth,pred).normalizedDistance


def triangle_height(p1,p2,p3):
    d12 = haversine(p1[:,0],p1[:,1],p2[:,0],p2[:,1])
    d13 = haversine(p1[:,0],p1[:,1],p3[:,0],p3[:,1])
    d23 = haversine(p2[:,0],p2[:,1],p3[:,0],p3[:,1])
    p = (d12+d13+d23)/2
    return np.sqrt(p*(p-d12)*(p-d13)*(p-d23))/d23

def cross_track_distance(truth, pred):
    tiled_pred = np.expand_dims(pred,1).repeat(len(truth),axis=1)
    dist = np.linalg.norm(tiled_pred - truth, axis=2)
    close_ind = np.argsort(dist)[:,:2]
    close_truth = truth[close_ind]
    return triangle_height(pred,close_truth[:,0],close_truth[:,1])