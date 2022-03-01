from abc import ABCMeta, abstractmethod
import logging
import csv
from glob import glob
import pandas as pd
from src.common.calculate_function import *

INF = 9999999


class ProximityGraph(object):
    __metaclass__ = ABCMeta

    def __init__(self, dirs, folder_name, gt_file, traj_len, query_id):
        super(ProximityGraph, self).__init__()
        self.query_id = query_id
        self.dirs = dirs
        self.folder_name = folder_name
        self.top_k = []
        self.raw_top_k_set = {}
        self.top_k_set = {}
        self.original_scores = []
        self.original_scores_dic = {}

        self.gt_file = gt_file
        self.traj_len = traj_len
        self.ground_truth, self.gt_num, self.gt_edge = self.load_gt()
        self.top_k_recall = 0

        self.dis_list = []
        self.tim_list = []
        self.vis_list = []
        self.vel_list = []

        self.max_dis = -1
        self.max_time = -1
        self.max_vision = -1
        self.min_dis = INF
        self.min_time = INF
        self.min_vision = INF

        self.min_vel_range = -1
        self.max_vel_range = -1
        self.min_vis_range = -1
        self.max_vis_range = -1

        self.candidate_paths = []
        self.scores = []
        self.coherence_scores = []

        self.score_pool = {}
        self.all_edges = {}
        self.vis_dic = {}

    def load_gt(self):
        ground_truth = set()
        ground_truth_edges = set()
        with open(self.gt_file, 'r') as f:
            gt = [line[:-1].split(',') for line in f.readlines()][:self.traj_len]
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            for t in range(st, et + 1):
                ground_truth.add((node, t))
        for idx_a in range(len(gt)-1):
            for idx_b in range(idx_a + 1, min(len(gt), idx_a + 4)):
                node1, st1, et1 = int(gt[idx_a][0]), int(gt[idx_a][1]), int(gt[idx_a][2])
                node2, st2, et2 = int(gt[idx_b][0]), int(gt[idx_b][1]), int(gt[idx_b][2])
                for t1 in range(st1, et1 + 1):
                    for t2 in range(st2, et2 + 1):
                        ground_truth_edges.add((node1, t1, node2, t2))
        return ground_truth, len(gt), ground_truth_edges

    def cal_closeness(self, node_dict):
        # logging.info("Calculating 3-dimension closeness")
        # 按照时间（帧号）对top-k重新排序，方便处理
        raw_top_k = pd.read_csv(self.dirs + '/top_k.csv', header=0).values.tolist()

        hit_num = 0
        for item in raw_top_k:
            if (item[0], item[1]) in self.ground_truth:
                hit_num += 1
        self.top_k_recall = hit_num / self.gt_num

        raw_top_k_set = {}
        for idx, i in enumerate(raw_top_k):
            raw_top_k_set[(i[0], i[1])] = idx
        # self.raw_top_k_set = raw_top_k_set

        top_k = list(filter(lambda x: x[2] > 1, raw_top_k))

        # 删掉一些时间上重叠且只出现一次的
        duplicates = list(filter(lambda x: x[2] == 1, raw_top_k))
        duplicates.sort(key=lambda x: x[1])
        candidate_stack = []
        flag_pop = False
        while len(duplicates) > 0:
            if len(candidate_stack) == 0:
                candidate_stack.insert(0, duplicates.pop(0))
            else:
                curr_can = duplicates.pop(0)
                # 时间重叠了
                if abs(curr_can[1] - candidate_stack[0][1]) < setting.DOWN_SAMPLE_FPS:
                    flag_pop = True
                    continue
                else:
                    if flag_pop:
                        candidate_stack.pop(0)
                        flag_pop = False
                    candidate_stack.insert(0, curr_can)
        top_k = top_k + candidate_stack
        top_k.sort(key=lambda x: x[1])
        self.top_k = top_k

        top_k_set = {}
        cnt = 0
        for i in top_k:
            top_k_set[(i[0], i[1])] = cnt
            cnt += 1
        # self.top_k_set = top_k_set
        self.raw_top_k_set = raw_top_k_set

        f0 = open(self.dirs + '/sorted_top_k.csv', 'w')
        csv_writer = csv.writer(f0)
        csv_writer.writerow(
            ["camid", "avg timestamp", "appear times", "node weight", "timestamps", "feature index", "index in frame"])
        for line in top_k:
            csv_writer.writerow(line)

        # 开一个文件记录三个维度的信息
        f = open(self.dirs + '/original_score.csv', 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["start node", "start time", "end node", "end time", "index1", "index2", "dis", "time", "velocity", "vision"])

        # 计算所有可能的对
        time_cal_score_when_build_graph = time.time()

        # for i in range(0, len(top_k) - 10):
        #     for j in range(i + 1, i + 10):
        if len(raw_top_k) < 25:
            max_idx = len(top_k)-1
        else:
            max_idx = len(top_k) - 10
        for i in range(0, max_idx):
            for j in range(i + 1, min(i + 10, len(top_k))):
                # 同一个点不能自己连自己（即使时间不一样）
                if top_k[i][0] != top_k[j][0]:
                    # 计算每个维度的得分
                    dis, t_dis = cal_dis(int(top_k[i][0]), int(top_k[j][0]), node_dict)

                    if dis == float("inf"):
                        continue
                    t = int(top_k[j][1]) - int(top_k[i][1])
                    if t != 0:
                        velocity = dis / t * setting.DOWN_SAMPLE_FPS  # 5fps
                    else:
                        velocity = 0
                    # if velocity > 40:
                    #     continue
                    idx1 = raw_top_k_set[(top_k[i][0], top_k[i][1])]
                    idx2 = raw_top_k_set[(top_k[j][0], top_k[j][1])]
                    vision, t_vision = cal_vision(idx1, idx2, self.dirs)

                    # 记录信息
                    if float(dis) > 0 and int(t) > 0:
                        # if int(t) > 0:
                        temp = [top_k[i][0], top_k[i][1], top_k[j][0], top_k[j][1], i, j, dis, t, velocity, vision]
                        self.original_scores.append(temp)
                        nodes = (top_k[i][0], top_k[i][1], top_k[j][0], top_k[j][1])
                        info = [i, j, dis, t, velocity, vision]
                        self.original_scores_dic[nodes] = info
                        csv_writer.writerow(temp)

                        self.dis_list.append(dis)
                        self.tim_list.append(t)
                        self.vis_list.append(vision)
                        self.vel_list.append(velocity)

                        # self.max_dis = max(self.max_dis, dis)
                        # self.max_time = max(self.max_time, t)
                        # self.max_vision = max(self.max_vision, vision)
                        # self.min_dis = min(self.min_dis, dis)
                        # self.min_time = min(self.min_time, t)
                        # self.min_vision = min(self.min_vision, vision)
        self.min_dis, self.max_dis = get_percentile(self.dis_list, 0.15, 0.5)
        self.min_time, self.max_time = get_percentile(self.tim_list, 0.25, 0.75)
        # self.min_dis, self.max_dis = get_percentile(self.dis_list, 0, 1)
        # self.min_time, self.max_time = get_percentile(self.tim_list, 0, 1)
        self.min_vision, self.max_vision = get_percentile(self.vis_list, 0, 1)

        self.min_vel_range, self.max_vel_range = get_interval_range(self.vel_list)
        # logging.info("velocity range: %.4f ~ %.4f" % (self.min_vel_range, self.max_vel_range))
        self.min_vis_range, self.max_vis_range = get_interval_range(self.vis_list)
        # logging.info("vision range: %.4f ~ %.4f" % (self.min_vis_range, self.max_vis_range))

        time_cal_score_when_build_graph = time.time() - time_cal_score_when_build_graph
        # self.get_max_min_distance(node_dict)
        return time_cal_score_when_build_graph

    def get_max_min_distance(self, edge_length):
        max_dis, max_tim = -1, -1
        min_dis, min_tim = 99999, 99999
        nums = glob("/home/wyr/TRG_cityflow/src/data/datasets/video_gt/" + self.folder_name + "/*")
        for file in nums:
            # file = open ("/home/zju/czh/TRG_Iteration/src/data/datasets/video_gt/" + self.folder_name + "/" + str (
            #     idx + 1) + ".txt")
            reader = open(file)
            node_list_raw, node_list = [], []
            for line in reader:
                node = line.rstrip("\n").split(',')
                node_list.append(node)
            for node_i in range(len(node_list) - 1):
                dis, t_dis = cal_dis(int(node_list[node_i][0]), int(node_list[node_i + 1][0]), edge_length)
                tim = abs(int(node_list[node_i + 1][1]) - int(node_list[node_i][1]))
                if node_i != 0:
                    if dis > 99999:
                        dis = 0.0
                        tim = 0.0
                        max_dis = max(max_dis, dis)
                        max_tim = max(max_tim, tim)
                    else:
                        max_dis = max(max_dis, dis)
                        min_dis = min(min_dis, dis)
                        max_tim = max(max_tim, tim)
                        min_tim = min(min_tim, tim)
        # print (max_dis, min_dis, max_tim, min_tim)
        self.max_dis = max_dis
        self.min_dis = min_dis
        self.max_time = max_tim
        self.min_time = min_tim

    @abstractmethod
    def cal_3d_score(self, delta, query_feature, top_k_p, top_k_r):
        pass
