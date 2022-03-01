import copy
from queue import PriorityQueue
from src.traj_recovery.basic_solver import *


class CoarseHeapNoRefine(BasicSolver, ABC):
    def __init__(self, traj_len, node_num, video_time, sample_rate, k_num, delta, lam):
        super().__init__(traj_len, node_num, video_time, sample_rate, k_num, delta, lam)
        self.longest_path_list = []
        self.selection_path_score_list = []
        self.shortest_path_file = setting.DATASET_PATH + "trajectory/" + self.folder_name + "/ground_truth/shortest.pkl"
        self.shortest_path = pickle.load(open(self.shortest_path_file, "rb"))

    def generate_coarse_grained_traj(self, query_id, dirs, query_feature, graph):
        logging.info("Generating coarse-grained trajectory")
        longest_path = PriorityQueue()
        heap_num = 50
        time_generate_coarse_grained = time.time()
        partial_path_dic = []
        longest_len = 0
        remove_prefix = set()
        gt_path = []
        gt_info = []
        gt_node_index_list = []
        score_dic = {}

        if len(graph.top_k) < 25:
            flag_high_top_k = True
        else:
            flag_high_top_k = False

        for i in range(len(graph.top_k)):
            # 得到这个点的信息
            node, timestamp, node_weight = graph.top_k[i][0], graph.top_k[i][1], graph.top_k[i][3]
            filename = dirs + '/avg_features.wyr'
            new_feature = load_avg_feature(filename, graph.raw_top_k_set[(node, timestamp)])
            feature_dis = get_cos_distance(new_feature, query_feature[0])
            # 这个点的入度为0
            if (node, timestamp) not in graph.all_pre_nodes.keys():
                partial_path_dic.append([[i]])
            # 入度不为0则寻找前一个点对应的 partial paths
            else:
                # remove_prefix = set()
                pre_nodes = graph.all_pre_nodes[(node, timestamp)]
                all_partial_paths = []
                for pre_node in pre_nodes:
                    velocity = graph.original_scores_dic[(pre_node[0][0], pre_node[0][1], node, timestamp)][4]
                    # 得到前一个点对应的所有可能的 partial path
                    partial_paths = partial_path_dic[pre_node[1]]
                    for curr_path in partial_paths:
                        curr_path_str = str(curr_path)
                        temp = copy.deepcopy(curr_path)
                        temp.append(i)

                        new_path_str = str(temp)
                        flag_remove = False
                        for l in range(2, longest_len + 1):
                            path_str = str(temp[:l])
                            if path_str in remove_prefix:
                                flag_remove = True
                                break
                        if flag_remove:
                            continue

                        # 只有一条边就不检查区间了，直接算得分
                        if len(temp) == 2:
                            all_partial_paths.append(temp)
                            # node_list = [(graph.top_k[int(jjj)][0], graph.top_k[int(jjj)][1]) for jjj in temp]
                            # curr_velocity_range = graph.velocity_range[str(temp)]
                            # curr_vision_range = graph.vision_range[str(temp)]
                            # logging.info(
                            #     "%s, %s, %s" % (str(node_list), str(curr_vision_range), str(curr_velocity_range)))
                        # 有多条边就开始计算 path 的得分
                        elif len(temp) > 2:
                            # 得到原本路径的信息
                            curr_min_weight = graph.min_weight[curr_path_str]
                            curr_weight_list = graph.weight_list[curr_path_str]
                            curr_velocity_list = graph.velocity_list[curr_path_str]
                            curr_velocity_range = graph.velocity_range[curr_path_str]
                            curr_velocity_range_index = graph.velocity_range_index[curr_path_str]
                            curr_vision_list = graph.vision_list[curr_path_str]
                            curr_vision_range = graph.vision_range[curr_path_str]
                            curr_vision_range_index = graph.vision_range_index[curr_path_str]

                            # 得到新点的信息
                            new_weight_list = copy.deepcopy(curr_weight_list)
                            new_weight_list.append(node_weight)
                            new_velocity_list = copy.deepcopy(curr_velocity_list)
                            new_velocity_list.append(velocity)
                            new_vision_list = copy.deepcopy(curr_vision_list)
                            new_vision_list.append(feature_dis)

                            # 检查是否可以剪枝
                            remove_head = -1
                            flag1, flag2 = False, False

                            # 检查新的速度区间
                            if velocity < curr_velocity_range[0]:
                                curr_v_min_index, curr_v_max_index = curr_velocity_range_index[0][0], \
                                                                     curr_velocity_range_index[1][0]
                                if curr_v_min_index < curr_v_max_index:
                                    remove_head = curr_v_max_index
                                    flag1 = True
                                new_velocity_range = (velocity, curr_velocity_range[1])
                                new_velocity_range_index = (
                                    (len(curr_path) - 1, len(curr_path)), curr_velocity_range_index[1])
                            elif velocity > curr_velocity_range[1]:
                                curr_v_min_index, curr_v_max_index = curr_velocity_range_index[0][0], \
                                                                     curr_velocity_range_index[1][0]
                                if curr_v_max_index < curr_v_min_index:
                                    remove_head = curr_v_min_index
                                    flag1 = True
                                new_velocity_range = (curr_velocity_range[0], velocity)
                                new_velocity_range_index = (
                                    curr_velocity_range_index[0], (len(curr_path) - 1, len(curr_path)))
                            else:
                                new_velocity_range = curr_velocity_range
                                new_velocity_range_index = curr_velocity_range_index

                            # 检查新的视觉区间
                            if feature_dis < curr_vision_range[0]:
                                curr_vis_min_index, curr_vis_max_index = curr_vision_range_index[0], \
                                                                         curr_vision_range_index[1]
                                if curr_vis_min_index < curr_vis_max_index:
                                    flag2 = True
                                    remove_head = max(remove_head, curr_vis_max_index)  # 取交集
                                new_vision_range = (feature_dis, curr_vision_range[1])
                                new_vision_range_index = (len(curr_path), curr_vis_max_index)
                            elif feature_dis > curr_vision_range[1]:
                                curr_vis_min_index, curr_vis_max_index = curr_vision_range_index[0], \
                                                                         curr_vision_range_index[1]
                                if curr_vis_max_index < curr_vis_min_index:
                                    flag2 = True
                                    remove_head = max(remove_head, curr_vis_min_index)  # 取交集
                                new_vision_range = (curr_vision_range[0], feature_dis)
                                new_vision_range_index = (curr_vis_min_index, len(curr_path))
                            else:
                                new_vision_range = curr_vision_range
                                new_vision_range_index = curr_vision_range_index

                            # 计算得分
                            new_min_weight = min(curr_min_weight, node_weight)
                            velocity_len = max(new_velocity_range[1] - new_velocity_range[0], 0.01)
                            vision_len = max(new_vision_range[1] - new_vision_range[0], 0.01)
                            vel_score = cal_score(velocity_len, graph.max_vel_range, graph.min_vel_range)
                            vis_score = cal_score(vision_len, graph.max_vis_range, graph.min_vis_range)
                            new_score = new_min_weight * vel_score * vis_score
                            # new_score = 1 / (velocity_len * self.lam + vision_len)  # + time_len * self.lam

                            if remove_head != -1 and flag1 and flag2:
                                remove_prefix.add(str(temp[remove_head:]))
                                longest_len = max(longest_len, len(temp[remove_head:]))

                            # 输出中间信息
                            flag_gt = True
                            node_list_str = []
                            for jjj in temp:
                                info = (graph.top_k[int(jjj)][0], graph.top_k[int(jjj)][1])
                                if info not in self.ground_truth:
                                    flag_gt = False
                                    break
                                else:
                                    node_list_str.append(info)
                            if flag_gt:
                                if len(gt_path) == 0:
                                    gt_path.append(node_list_str)
                                    gt_info.append([new_score, vel_score, vis_score, new_min_weight])
                                    gt_node_index_list.append(temp)
                                elif len(node_list_str) == len(gt_path[0]):
                                    gt_path.append(node_list_str)
                                    gt_info.append([new_score, vel_score, vis_score, new_min_weight])
                                    gt_node_index_list.append(temp)
                                elif len(node_list_str) > len(gt_path[0]):
                                    gt_path = [node_list_str]
                                    gt_info = [[new_score, vel_score, vis_score, new_min_weight]]
                                    gt_node_index_list = [temp]
                                # logging.info("%s, score: %.3f, vel: %.3f, vis: %.3f, weight: %.3f" % (node_list_str, new_score, vel_score, vis_score, new_min_weight))

                            score_dic[str(temp)] = [new_score, vel_score, vis_score, new_min_weight]

                            flag_pass = False
                            if flag_high_top_k and new_min_weight > 3:
                                flag_pass = True
                            if len(temp) > 8 and vel_score > 0.98:
                                flag_pass = True

                            if new_score > self.delta or flag_pass:
                                prune_flag = False
                                # short_path_list = self.shortest_path[
                                #     (graph.top_k[temp[-1]][0], graph.top_k[temp[-2]][0])]
                                # if len(short_path_list) >= 10:
                                #     temp_frame_s, temp_frame_e = int(graph.top_k[temp[-2]][1]), int(
                                #         graph.top_k[temp[-1]][1])
                                #     find_candidate = []
                                #     for short_path_node in short_path_list:
                                #         for candidate_line in graph.top_k:
                                #             candidate_node, candidate_frame = int(candidate_line[0]), int(
                                #                 candidate_line[1])
                                #             if candidate_node == short_path_node and (
                                #                     temp_frame_s <= candidate_frame <= temp_frame_e):
                                #                 find_candidate.append([candidate_node, candidate_frame])
                                #                 temp_frame_s = candidate_frame
                                #                 break
                                #     if len(find_candidate) == 0:
                                #         prune_flag = True
                                if not prune_flag:  # not Prune_flag:
                                    graph.min_weight[new_path_str] = new_min_weight
                                    graph.weight_list[new_path_str] = new_weight_list
                                    graph.velocity_list[new_path_str] = new_velocity_list
                                    graph.velocity_range[new_path_str] = new_velocity_range
                                    graph.velocity_range_index[new_path_str] = new_velocity_range_index
                                    graph.vision_list[new_path_str] = new_vision_list
                                    graph.vision_range[new_path_str] = new_vision_range
                                    graph.vision_range_index[new_path_str] = new_vision_range_index
                                    # 加入小顶堆
                                    if longest_path.qsize() < heap_num:
                                        longest_path.put((len(temp), temp))
                                    else:
                                        heap_min = longest_path.get()
                                        if len(temp) > heap_min[0]:
                                            longest_path.put((len(temp), temp))
                                        else:
                                            longest_path.put(heap_min)
                                    all_partial_paths.append(temp)
                all_partial_paths.append([i])
                partial_path_dic.append(all_partial_paths)
        time_generate_coarse_grained = time.time() - time_generate_coarse_grained

        # logging.info("Longest GT paths:")
        # for idx in range(len(gt_path)):
        #     logging.info("%s, score: %.3f, vel: %.3f, vis: %.3f, node weight: %.3f" % (
        #         str(gt_path[idx]), gt_info[idx][0], gt_info[idx][1], gt_info[idx][2], gt_info[idx][3]))
            # index_list = gt_node_index_list[idx]
            # for node_idx in index_list:
            #     node = graph.top_k[int(node_idx)]
            #     camera_id, timestamp, image_num, weight = node[0], node[1], node[2], node[3]
            #     vis_dis = graph.vis_dis_dict[(camera_id, timestamp)]
            #     logging.info("(%d, %d), #images: %d, weight: %.4f, vis dis: %.4f" % (camera_id, timestamp, image_num, weight, vis_dis))

        paths = []
        scores = {}
        while not longest_path.empty():
            tmp = longest_path.get()[1]
            self.longest_path_list.append(tmp)

        for p in self.longest_path_list:
            full_path = [(graph.top_k[node][0], graph.top_k[node][1]) for node in p]
            paths.append(full_path)
            score_info = score_dic[str(p)]
            scores[str(full_path)] = score_info

        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/coarse_grained_path.csv' % (
            self.delta, self.k)
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["query", "longest path"])
            csv_writer.writerow([query_id, paths])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([query_id, paths])

        return time_generate_coarse_grained, paths, scores

    def path_selection(self, query_id, dirs, query_feature, graph):
        self.longest_path_list = []
        self.time_path_selection, candidate_paths, scores = self.generate_coarse_grained_traj(query_id, dirs,
                                                                                              query_feature, graph)
        return candidate_paths, scores
