from abc import ABCMeta, abstractmethod
import logging
import csv
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

    def cal_closeness(self, node_dict):
        logging.info("Building proximity graph")
        raw_top_k = pd.read_csv(self.dirs + '/top_k.csv', header=0).values.tolist()

        raw_top_k_set = {}
        for idx, i in enumerate(raw_top_k):
            raw_top_k_set[(i[0], i[1])] = idx

        top_k = list(filter(lambda x: x[2] > 1, raw_top_k))

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
        self.raw_top_k_set = raw_top_k_set

        f0 = open(self.dirs + '/sorted_top_k.csv', 'w')
        csv_writer = csv.writer(f0)
        csv_writer.writerow(
            ["camid", "avg timestamp", "appear times", "node weight", "timestamps", "feature index", "index in frame"])
        for line in top_k:
            csv_writer.writerow(line)

        f = open(self.dirs + '/original_score.csv', 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["start node", "start time", "end node", "end time", "index1", "index2", "dis", "time", "velocity", "vision"])

        time_cal_score_when_build_graph = time.time()

        for i in range(0, len(top_k) - 1):
            for j in range(i + 1, len(top_k)):
                if top_k[i][0] != top_k[j][0]:
                    dis, t_dis = cal_dis(int(top_k[i][0]), int(top_k[j][0]), node_dict)

                    if dis == float("inf"):
                        continue
                    t = int(top_k[j][1]) - int(top_k[i][1])
                    if t != 0:
                        velocity = dis / t * setting.DOWN_SAMPLE_FPS
                    else:
                        velocity = 0
                    idx1 = raw_top_k_set[(top_k[i][0], top_k[i][1])]
                    idx2 = raw_top_k_set[(top_k[j][0], top_k[j][1])]
                    vision, t_vision = cal_vision(idx1, idx2, self.dirs)

                    if float(dis) > 0 and int(t) > 0:
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

        self.min_dis, self.max_dis = get_percentile(self.dis_list, 0, 1)
        self.min_time, self.max_time = get_percentile(self.tim_list, 0, 1)
        self.min_vision, self.max_vision = get_percentile(self.vis_list, 0, 1)

        self.min_vel_range, self.max_vel_range = get_interval_range(self.vel_list)
        self.min_vis_range, self.max_vis_range = get_interval_range(self.vis_list)

        time_cal_score_when_build_graph = time.time() - time_cal_score_when_build_graph
        return time_cal_score_when_build_graph

    @abstractmethod
    def cal_3d_score(self, delta, query_feature, top_k_p, top_k_r):
        pass
