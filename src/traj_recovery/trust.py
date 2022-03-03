from src.traj_recovery.basic_solver import *


class CoarseHeapNoRefine(BasicSolver, ABC):
    def __init__(self, traj_len, node_num, video_time, sample_rate, k_num, delta):
        super().__init__(traj_len, node_num, video_time, sample_rate, k_num, delta)
        self.longest_path_list = []
        self.selection_path_score_list = []

    def generate_coarse_grained_traj(self, query_id, dirs, query_feature, graph):
        logging.info("Generating candidate trajectories")
        longest_paths = []
        time_generate_coarse_grained = time.time()
        partial_path_dic = []
        longest_len = 0
        remove_prefix = set()

        for i in range(len(graph.top_k)):
            # get info of this node
            node, timestamp, node_weight = graph.top_k[i][0], graph.top_k[i][1], graph.top_k[i][3]
            filename = dirs + '/avg_features.wyr'
            new_feature = load_avg_feature(filename, graph.raw_top_k_set[(node, timestamp)])
            feature_dis = get_cos_distance(new_feature, query_feature[0])
            # the incoming degree is 0
            if (node, timestamp) not in graph.all_pre_nodes.keys():
                partial_path_dic.append([[i]])
            # the incoming degree > 0
            else:
                pre_nodes = graph.all_pre_nodes[(node, timestamp)]
                all_partial_paths = []
                for pre_node in pre_nodes:
                    velocity = graph.original_scores_dic[(pre_node[0][0], pre_node[0][1], node, timestamp)][4]
                    # get all the possible partial paths of the previous node
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

                        if len(temp) == 2:
                            all_partial_paths.append(temp)
                        elif len(temp) > 2:
                            # get info of the original path
                            curr_min_weight = graph.min_weight[curr_path_str]
                            curr_weight_list = graph.weight_list[curr_path_str]
                            curr_velocity_list = graph.velocity_list[curr_path_str]
                            curr_velocity_range = graph.velocity_range[curr_path_str]
                            curr_velocity_range_index = graph.velocity_range_index[curr_path_str]
                            curr_vision_list = graph.vision_list[curr_path_str]
                            curr_vision_range = graph.vision_range[curr_path_str]
                            curr_vision_range_index = graph.vision_range_index[curr_path_str]

                            # get info of the new node
                            new_weight_list = copy.deepcopy(curr_weight_list)
                            new_weight_list.append(node_weight)
                            new_velocity_list = copy.deepcopy(curr_velocity_list)
                            new_velocity_list.append(velocity)
                            new_vision_list = copy.deepcopy(curr_vision_list)
                            new_vision_list.append(feature_dis)

                            # check if can be pruned
                            remove_head = -1
                            flag1, flag2 = False, False

                            # check the new velocity interval
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

                            # check the new visual interval
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

                            # calculate the holistic score of this new path
                            new_min_weight = min(curr_min_weight, node_weight)
                            velocity_len = max(new_velocity_range[1] - new_velocity_range[0], 0.01)
                            vision_len = max(new_vision_range[1] - new_vision_range[0], 0.01)
                            vel_score = cal_score(velocity_len, graph.max_vel_range, graph.min_vel_range)
                            vis_score = cal_score(vision_len, graph.max_vis_range, graph.min_vis_range)
                            new_score = new_min_weight * vel_score * vis_score

                            if remove_head != -1 and flag1 and flag2:
                                remove_prefix.add(str(temp[remove_head:]))
                                longest_len = max(longest_len, len(temp[remove_head:]))

                            if new_score > self.delta:
                                graph.min_weight[new_path_str] = new_min_weight
                                graph.weight_list[new_path_str] = new_weight_list
                                graph.velocity_list[new_path_str] = new_velocity_list
                                graph.velocity_range[new_path_str] = new_velocity_range
                                graph.velocity_range_index[new_path_str] = new_velocity_range_index
                                graph.vision_list[new_path_str] = new_vision_list
                                graph.vision_range[new_path_str] = new_vision_range
                                graph.vision_range_index[new_path_str] = new_vision_range_index
                                # update the longest paths
                                if len(longest_paths) == 0:
                                    longest_paths.append(temp)
                                else:
                                    if len(longest_paths[0]) == len(temp):
                                        longest_paths.append(temp)
                                    elif len(longest_paths[0]) < len(temp):
                                        longest_paths = [temp]
                                all_partial_paths.append(temp)

                all_partial_paths.append([i])
                partial_path_dic.append(all_partial_paths)
        time_generate_coarse_grained = time.time() - time_generate_coarse_grained

        paths = []
        self.longest_path_list = longest_paths

        for p in self.longest_path_list:
            full_path = [(graph.top_k[node][0], graph.top_k[node][1]) for node in p]
            paths.append(full_path)

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

        return time_generate_coarse_grained, paths

    def path_selection(self, query_id, dirs, query_feature, graph):
        self.longest_path_list = []
        self.time_path_selection, candidate_paths = self.generate_coarse_grained_traj(query_id, dirs, query_feature, graph)
        return candidate_paths
