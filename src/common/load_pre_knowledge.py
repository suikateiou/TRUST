import pickle

import pandas as pd
import re
from src.setting import setting


def load_dis_file(folder_name):
    filename = setting.DATASET_PATH + "trajectory/%s/ground_truth/all_pairs_distance.csv" % folder_name
    data = pd.read_csv(filename, header=0).values.tolist()
    edge_length = {}
    for idx in range(len(data)):
        edge = data[idx]
        node1 = int(edge[0])
        node2 = int(edge[1])
        dis = float(edge[2])
        if dis == -1:
            dis = 999999999
        t = float(edge[3])
        edge_length[(node1, node2)] = (dis, t)
    return edge_length


# def load_basemap_neighbours(folder_name):
#     all_neighbours = {}
#     filename = setting.DATASET_PATH + "trajectory/%s/ground_truth/neighbors.csv" % folder_name
#     data = pd.read_csv(filename, header=0)
#     for idx, row in data.iterrows():
#         temp = set()
#         cnt = 0
#         [node, neighbours, rank] = row
#         if neighbours != "set()":
#             neighbours = re.split(r", ", neighbours[1:-1])
#             for possible_node in neighbours:
#                 if possible_node != '' and cnt <= setting.NEIGHBOUR_NUM:
#                     temp.add(int(possible_node))
#                     cnt += 1
#         all_neighbours[node] = temp
#     return all_neighbours

def load_basemap_neighbours(folder_name):
    filename = setting.DATASET_PATH + "trajectory/%s/ground_truth/neighbors.pkl" % folder_name
    all_neighbours = pickle.load(open(filename, "rb"))
    return all_neighbours


def load_video_frames(folder_name, node_num):
    node_frame_index = {}
    for node in range(node_num):
        frame_index = {}
        filename = setting.NODE_FEATURES + '%s/%d/frame_index_%d.csv' % (folder_name, node, node)
        data = pd.read_csv(filename).values.tolist()
        for record in data:
            ans_list = []
            [frame, row] = record
            index_list = re.split(r"[, ]", row[1:-1])
            for index in index_list:
                if index != '':
                    ans_list.append(int(index))
            frame_index[frame] = ans_list
        node_frame_index[node] = frame_index
    return node_frame_index


def load_pairs(folder_name):
    filename = setting.DATASET_PATH + "trajectory/%s/ground_truth/all_pairs_distance.csv" % folder_name
    all_pairs = pd.read_csv(filename, header=0)
    pair_info = {}
    for idx, row in all_pairs.iterrows():
        node1, node2, dis, t = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        pair_info[(node1, node2)] = (dis, t)
        pair_info[(node2, node1)] = (dis, t)
    return pair_info


def get_possible_nodes(all_neighbours, node1, node2):
    n1 = all_neighbours[node1]
    n2 = all_neighbours[node2]
    return n1 & n2

