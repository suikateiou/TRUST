from abc import ABC
from src.proximity_graph.proximity_graph import *
from src.common.calculate_function import *


class CoarseGraph(ProximityGraph, ABC):
    def __init__(self, dirs, folder_name, gt_file, traj_len, query_id):
        super().__init__(dirs, folder_name, gt_file, traj_len, query_id)
        self.all_pre_nodes = {}

        self.min_weight = {}
        self.weight_list = {}

        self.velocity_list = {}
        self.velocity_range = {}
        self.velocity_range_index = {}

        self.vision_list = {}
        self.vision_range = {}
        self.vision_range_index = {}

    def cal_3d_score(self, delta, query_feature, top_k_p, top_k_r):
        self.all_edges = {}
        self.all_pre_nodes = {}
        t = time.time()
        f = open(self.dirs + '/coherence_score.csv', 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["start node", "start time", "end node", "end time", "index1", "index2", "dis_score", "time_score", "vis_score", "total_score"])

        for idx, row in enumerate(self.original_scores):
            [sn, st, en, et, idx1, idx2, dis, tim, velocity, vision] = row
            dis_score = cal_score(dis, self.max_dis, self.min_dis)
            tim_score = cal_score(tim, self.max_time, self.min_time)
            vis_score = cal_score(vision, self.max_vision, self.min_vision)
            score = dis_score + tim_score + vis_score

            temp = [int(sn), int(st), int(en), int(et), idx1, idx2, dis_score, tim_score, vis_score, score]
            self.coherence_scores.append(temp)
            csv_writer.writerow(temp)
            
            if score > delta:
                self.candidate_paths.append([idx1, idx2])
                self.scores.append(score)
                # appear times
                w1, w2 = self.top_k[idx1][3], self.top_k[idx2][3]
                self.min_weight[str([idx1, idx2])] = min(w1, w2)
                self.weight_list[str([idx1, idx2])] = [w1, w2]
                # delegate feature of a cluster
                filename = self.dirs + '/avg_features.wyr'
                feature1 = load_avg_feature(filename, self.raw_top_k_set[(self.top_k[idx1][0], self.top_k[idx1][1])])
                feature2 = load_avg_feature(filename, self.raw_top_k_set[(self.top_k[idx2][0], self.top_k[idx2][1])])
                dis1 = get_cos_distance(feature1, query_feature[0])
                dis2 = get_cos_distance(feature2, query_feature[0])

                self.vis_dic[(int(sn), int(st))] = dis1
                self.vis_dic[(int(en), int(et))] = dis2

                self.vision_list[str([idx1, idx2])] = [dis1, dis2]
                if dis1 <= dis2:
                    self.vision_range[str([idx1, idx2])] = (dis1, dis2)
                    self.vision_range_index[str([idx1, idx2])] = (0, 1)
                else:
                    self.vision_range[str([idx1, idx2])] = (dis2, dis1)
                    self.vision_range_index[str([idx1, idx2])] = (1, 0)

                self.velocity_list[str([idx1, idx2])] = [velocity]
                self.velocity_range[str([idx1, idx2])] = (velocity, velocity)
                self.velocity_range_index[str([idx1, idx2])] = ((0, 1), (0, 1))

                next_node = [(int(en), int(et)), idx2, dis_score, tim_score, vis_score, score]
                pre_node = [(int(sn), int(st)), idx1, dis_score, tim_score, vis_score, score]
                if (int(sn), int(st)) not in self.all_edges.keys():
                    self.all_edges[(int(sn), int(st))] = [next_node]
                else:
                    self.all_edges[(int(sn), int(st))].append(next_node)
                if (int(en), int(et)) not in self.all_pre_nodes.keys():
                    self.all_pre_nodes[(int(en), int(et))] = [pre_node]
                else:
                    self.all_pre_nodes[(int(en), int(et))].append(pre_node)
                self.score_pool[str([idx1, idx2])] = score

        for node in self.all_edges.keys():
            self.all_edges[node].sort(key=lambda x: x[0][1], reverse=True)
        for node in self.all_pre_nodes.keys():
            self.all_pre_nodes[node].sort(key=lambda x: x[0][1], reverse=True)
        t = time.time() - t

        return t
