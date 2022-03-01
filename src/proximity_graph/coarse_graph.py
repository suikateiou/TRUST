from abc import ABC
import os
from src.proximity_graph.proximity_graph import *
from src.common.calculate_function import *


class CoarseGraph(ProximityGraph, ABC):
    def __init__(self, dirs, folder_name, gt_file, traj_len, query_id):
        super().__init__(dirs, folder_name, gt_file, traj_len, query_id)
        self.sum_weight = {}
        self.sum_feature = {}
        self.sum_feature2 = {}
        self.sum_velocity = {}
        self.sum_velocity2 = {}
        self.sum_time = {}
        self.sum_time2 = {}
        self.min_weight = {}
        self.weight_list = {}
        self.velocity_list = {}
        self.velocity_range = {}
        self.velocity_range_index = {}
        self.vision_list = {}
        self.vision_range = {}
        self.vision_range_index = {}
        self.all_pre_nodes = {}
        self.vis_dis_dict = {}
        self.image_num_dict = {}

    def cal_3d_score(self, delta, query_feature, top_k_p, top_k_r):
        self.all_edges = {}
        self.all_pre_nodes = {}
        # logging.info("Calculating 3-dimension scores")
        t = time.time()
        f = open(self.dirs + '/coherence_score.csv', 'w')
        csv_writer = csv.writer(f)
        # csv_writer.writerow(
        #     ["start node", "start time", "end node", "end time", "index1", "index2", "dis_score", "time_score",
        #      "vis_score", "total_score"])
        csv_writer.writerow(
            ["start node", "start time", "end node", "end time", "index1", "index2", "dis_score", "time_score",
             "vis_score", "total_score", "label", "judge"])

        # gt_dis, gt_tim, gt_vis = [], [], []
        # noise_dis, noise_tim, noise_vis = [], [], []

        if not os.path.exists(self.dirs + '/../graph_statistics.csv'):
            f1 = open(self.dirs + '/../graph_statistics.csv', 'w')
            csv_writer1 = csv.writer(f1)
            csv_writer1.writerow(["query", "edge num", "recall", "top-k recall"])

        if not os.path.exists(self.dirs + '/../overall_statistics.csv'):
            f2 = open(self.dirs + '/../overall_statistics.csv', 'w')
            csv_writer2 = csv.writer(f2)
            csv_writer2.writerow(["query", "top-k precision", "top-k recall", "graph node number", "graph edge number",
                                  "graph node recall", "graph edge recall"])

        edge_num = 0
        edge_hit = 0
        node_set = set()
        hit_node_set = set()

        # 计算得分
        for idx, row in enumerate(self.original_scores):
            [sn, st, en, et, idx1, idx2, dis, tim, velocity, vision] = row
            dis_score = cal_score(dis, self.max_dis, self.min_dis)
            tim_score = cal_score(tim, self.max_time, self.min_time)
            vis_score = cal_score(vision, self.max_vision, self.min_vision)
            # vis_score = vision
            score_list = [dis_score, tim_score, vis_score]
            score_list.sort()
            if score_list[-1] < 0.87:
                score = 0.6 * score_list[2] + 0.3 * score_list[1] + 0.1 * score_list[0]
            else:
                score = score_list[-1]
            # score = (dis_score + tim_score + vis_score) / 3
            # score = 0.6 * dis_score + 0.3 * tim_score + 0.1 * vis_score

            if (sn, st, en, et) in self.gt_edge:
                label = "GT"
            else:
                label = "NOISE"

            if label == "GT":
                if score > delta:
                    judge = "√"
                else:
                    # logging.info("%.3f, %.3f, %.3f, %.3f" % (dis_score, tim_score, vis_score, score))
                    judge = "X"
            else:
                if score <= delta:
                    judge = "√"
                else:
                    judge = "X"

            # 记录得分
            # temp = [int(sn), int(st), int(en), int(et), idx1, idx2, dis_score, tim_score, vis_score, score]
            temp = [int(sn), int(st), int(en), int(et), idx1, idx2, dis_score, tim_score, vis_score, score, label, judge]
            self.coherence_scores.append(temp)
            csv_writer.writerow(temp)
            
            # print(int(sn), int(en), score, dis_score, tim_score, vis_score)
            
            if score > delta:
                edge_num += 1
                node_set.add((int(sn), int(st)))
                node_set.add((int(en), int(et)))
                if (int(sn), int(st), int(en), int(et)) in self.gt_edge:
                    edge_hit += 1

                if (sn, st) in self.ground_truth:
                    hit_node_set.add((sn, st))
                if (en, et) in self.ground_truth:
                    hit_node_set.add((en, et))

                # if True:
                self.candidate_paths.append([idx1, idx2])
                self.scores.append(score)
                # 出现次数
                w1, w2 = self.top_k[idx1][3], self.top_k[idx2][3]
                self.sum_weight[str([idx1, idx2])] = w1 + w2
                self.min_weight[str([idx1, idx2])] = min(w1, w2)
                self.weight_list[str([idx1, idx2])] = [w1, w2]
                # feature的总和
                filename = self.dirs + '/avg_features.wyr'
                feature1 = load_avg_feature(filename, self.raw_top_k_set[(self.top_k[idx1][0], self.top_k[idx1][1])])
                feature2 = load_avg_feature(filename, self.raw_top_k_set[(self.top_k[idx2][0], self.top_k[idx2][1])])
                dis1 = get_cos_distance(feature1, query_feature[0])
                dis2 = get_cos_distance(feature2, query_feature[0])

                self.vis_dic[(int(sn), int(st))] = dis1
                self.vis_dic[(int(en), int(et))] = dis2

                self.vis_dis_dict[(int(sn), int(st))] = dis1
                self.vis_dis_dict[(int(en), int(et))] = dis2
                self.image_num_dict[(int(sn), int(st))] = self.top_k[idx1][2]
                self.image_num_dict[(int(en), int(et))] = self.top_k[idx2][2]

                self.sum_feature[str([idx1, idx2])] = dis1 + dis2
                self.sum_feature2[str([idx1, idx2])] = dis1 * dis1 + dis2 * dis2
                self.vision_list[str([idx1, idx2])] = [dis1, dis2]
                if dis1 <= dis2:
                    self.vision_range[str([idx1, idx2])] = (dis1, dis2)
                    self.vision_range_index[str([idx1, idx2])] = (0, 1)
                else:
                    self.vision_range[str([idx1, idx2])] = (dis2, dis1)
                    self.vision_range_index[str([idx1, idx2])] = (1, 0)

                # 速度的总和
                self.velocity_list[str([idx1, idx2])] = [velocity]
                self.sum_velocity[str([idx1, idx2])] = velocity
                self.sum_velocity2[str([idx1, idx2])] = velocity * velocity
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
        # 按照时间排序
        for node in self.all_edges.keys():
            self.all_edges[node].sort(key=lambda x: x[0][1], reverse=True)
        for node in self.all_pre_nodes.keys():
            self.all_pre_nodes[node].sort(key=lambda x: x[0][1], reverse=True)
        t = time.time() - t

        # if len(gt_dis):
        #     logging.info(f"GT dis: {np.min(gt_dis):.3f} ~ {np.max(gt_dis):.3f}, avg: {np.mean(gt_dis):.3f}")
        #     logging.info(f"GT tim: {np.min(gt_tim):.3f} ~ {np.max(gt_tim):.3f}, avg: {np.mean(gt_tim):.3f}")
        #     logging.info(f"GT vis: {np.min(gt_vis):.3f} ~ {np.max(gt_vis):.3f}, avg: {np.mean(gt_vis):.3f}")
        #
        # if len(noise_dis):
        #     logging.info(f"NOISE dis: {np.min(noise_dis):.3f} ~ {np.max(noise_dis):.3f}, avg: {np.mean(noise_dis):.3f}")
        #     logging.info(f"NOISE tim: {np.min(noise_tim):.3f} ~ {np.max(noise_tim):.3f}, avg: {np.mean(noise_tim):.3f}")
        #     logging.info(f"NOISE vis: {np.min(noise_vis):.3f} ~ {np.max(noise_vis):.3f}, avg: {np.mean(noise_vis):.3f}")

        f1 = open(self.dirs + '/../graph_statistics.csv', 'a')
        csv_writer1 = csv.writer(f1)
        csv_writer1.writerow([self.query_id, edge_num, len(hit_node_set) / self.gt_num, self.top_k_recall])

        f2 = open(self.dirs + '/../overall_statistics.csv', 'a')
        csv_writer2 = csv.writer(f2)
        csv_writer2.writerow([self.query_id, top_k_p, top_k_r, len(node_set), edge_num, len(hit_node_set) / self.gt_num,
                              edge_hit / (self.gt_num - 1)])

        return t
