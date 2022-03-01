import math
import time

import numpy as np
from geopy.distance import geodesic
from src.common.load_feature import *


def sigmoid(x):
    # return 1 / (1 + math.exp(-x))
    return 1 + math.log(x)


def cal_appear_weight(appear_times):
    return [sigmoid(x) for x in appear_times]


# def cal_dis(node1, node2, node_dict):
#     t = time.time()
#     dis = geodesic(node_dict[node1], node_dict[node2]).m
#     t = time.time() - t
#     return dis, t

def cal_dis(node1, node2, edge_length):
    t = time.time()
    minval = min(node1, node2)
    maxval = max(node1, node2)
    if (minval, maxval) in edge_length.keys():
        dis = edge_length[(minval, maxval)][0]
    else:
        dis = 999999999
    t = time.time() - t
    return dis, t


def get_cos_similar(v1, v2):
    if v1.shape != v2.shape:
        raise RuntimeError("array {} shape not match {}".format(v1.shape, v2.shape))
    if v1.ndim == 1:
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
    elif v1.ndim == 2:
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(v1.ndim))
    similarity = np.dot(v1, v2.T) / (v1_norm * v2_norm)
    return similarity


# 计算两点之间的视觉联系（平均向量之间的联系）
def cal_vision(idx1, idx2, folder):
    t = time.time()
    filename = folder + '/avg_features.wyr'
    feature1 = load_avg_feature(filename, idx1)
    feature2 = load_avg_feature(filename, idx2)
    vision = get_cos_similar(feature1, feature2)
    # vision = get_euclid_distance(feature1, feature2)
    t = time.time() - t
    return vision, t


def cal_time_probability(node1, node2, travel_time, distribution):
    t = time.time()
    if (node1, node2) not in distribution.keys() and (node2, node1) not in distribution.keys():
        score = -1
    else:
        if (node1, node2) in distribution.keys():
            [low, q1, q3, up] = distribution[(node1, node2)]
        else:
            [low, q1, q3, up] = distribution[(node2, node1)]
        if travel_time <= low or travel_time >= up:
            score = 0
        elif low < travel_time < q1:
            score = (travel_time - low) / (q1 - low)
        elif q1 <= travel_time <= q3:
            score = 1
        else:
            score = (up - travel_time) / (up - q3)
    t = time.time() - t
    return score, t


def cal_score(val, max_val, min_val):
    if min_val == max_val or val <= min_val:
        return 1
    if val >= max_val:
        return 0
    score = (max_val - val) / (max_val - min_val)
    score = max(min(score, 1), 0)
    return score


def get_cos_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1.0 - similarity
    return dist


def get_euclid_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    dist = np.sqrt(np.linalg.norm(np.expand_dims(a, axis=1)-np.expand_dims(b, axis=1), ord=2, axis=0))
    # print(dist)
    return dist


def cal_v_score(x):
    if abs(x) < 1e-5:
        return 0.001
    return math.sqrt(x)


def comp_cos_similar(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1.0 - similarity
    return dist


def get_percentile(data, low_bound, up_bound):
    data.sort()
    all_num = len(data)
    low_bound_index = max(round(all_num * low_bound), 0)
    up_bound_index = min(round(all_num * up_bound), all_num - 1)
    return data[low_bound_index], data[up_bound_index]


def get_interval_range(data):
    data.sort()
    min_val = data[0]
    sec_val = data[bisect.bisect_right(data, min_val)]
    max_val = data[round(len(data) * 0.5)]
    min_range = sec_val - min_val
    max_range = max_val - min_val
    return min_range, max_range
