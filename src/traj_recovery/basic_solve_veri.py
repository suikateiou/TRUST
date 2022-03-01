import shutil
import copy
import heapq
import matplotlib.pyplot as plt
from src.topk.topk import *
from src.topk.camera import *
from src.topk.feature_gallery import *
from src.proximity_graph.coarse_graph import *
from src.common.load_pre_knowledge import *
from src.common.union_find import *


class BasicSolver(object):
    __metaclass__ = ABCMeta

    def __init__(self, traj_len, node_num, video_time, sample_rate, k_num, delta, delta2, lam):
        super(BasicSolver, self).__init__()
        self.traj_len = traj_len
        self.node_num = node_num
        self.video_time = video_time
        self.down_sample_fps = sample_rate
        self.k = k_num
        self.delta = delta
        self.delta2 = delta2
        self.lam = lam

        # 轨迹恢复结果路径
        self.folder_name = "t%02d_c%03d_len%02d" % (video_time, node_num, traj_len)
        if not os.path.exists(setting.OUTPUT_PATH + self.folder_name):
            os.mkdir(setting.OUTPUT_PATH + self.folder_name)
        delta_folder = setting.OUTPUT_PATH + self.folder_name + "/delta_%.2f" % delta2
        if os.path.exists(delta_folder):
            shutil.rmtree(delta_folder)
        os.mkdir(delta_folder)
        self.output_folder = delta_folder + "/top_%d" % k_num
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        # 轨迹恢复可视化路径
        if os.path.exists(setting.OUTPUT_VISUAL_PATH + self.folder_name):
            shutil.rmtree(setting.OUTPUT_VISUAL_PATH + self.folder_name)
        os.mkdir(setting.OUTPUT_VISUAL_PATH + self.folder_name)
        delta_folder_visual = setting.OUTPUT_VISUAL_PATH + self.folder_name + "/delta_%.2f" % delta2
        if not os.path.exists(delta_folder_visual):
            os.mkdir(delta_folder_visual)
        self.output_folder_visual = delta_folder_visual + "/top_%d" % k_num
        if not os.path.exists(self.output_folder_visual):
            os.mkdir(self.output_folder_visual)

        # 加载其他信息
        self.edge_length = load_dis_file(self.folder_name)
        self.pair_info = load_pairs(self.folder_name)
        self.node_dict = self.load_node_dict()
        self.index_gallery = FeatureGallery(node_num, traj_len, video_time)

        # gt信息
        self.ground_truth = set()
        self.gt_num = 0
        self.ground_truth_edge = set()

        # 中间结果
        self.selection_path = []
        self.ans_node_list = []
        self.ans_time_list = []
        self.time_search = 0
        self.time_find_topk = 0
        self.time_build_graph = 0
        self.time_path_selection = 0
        self.time_merge = 0

        self.exceed_num = 0
        self.non_dominated = 0

        # 全部结果
        self.all_inference_time = []
        self.all_p = []
        self.all_r = []
        self.all_f1 = []

        # top-k的相关结果
        self.top_k_precision = 0
        self.top_k_recall = 0
        self.top_k_f1 = 0
        self.all_top_k_p = []
        self.all_top_k_r = []
        self.all_top_k_f1 = []
        self.all_top_k_t = []

    def reset(self):
        self.selection_path = []
        self.ans_node_list = []
        self.ans_time_list = []
        self.time_search = 0
        self.time_find_topk = 0
        self.time_build_graph = 0
        self.time_path_selection = 0
        self.time_merge = 0

        self.exceed_num = 0
        self.non_dominated = 0

    def load_node_dict(self):
        node_dict = {}
        with open(setting.DATASET_PATH + "trajectory/" + self.folder_name + "/ground_truth/node.txt", "r") as f:
            for line in f.readlines():
                node = line[:-1].split(" ")
                node_dict[int(node[0])] = (float(node[1]), float(node[2]))
        return node_dict

    def load_ground_truth(self, query_id):
        carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[query_id]
        gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
        ground_truth = set()
        ground_truth_edge = set()
        with open(gt_file, 'r') as f:
            gt = [line[:-1].split(',') for line in f.readlines()][:self.traj_len]
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            for t in range(st, et + 1):
                ground_truth.add((node, t))
        for idx_a in range(len(gt) - 1):
            idx_b = idx_a + 1
            node1, st1, et1 = int(gt[idx_a][0]), int(gt[idx_a][1]), int(gt[idx_a][2])
            node2, st2, et2 = int(gt[idx_b][0]), int(gt[idx_b][1]), int(gt[idx_b][2])
            for t1 in range(st1, et1 + 1):
                for t2 in range(st2, et2 + 1):
                    ground_truth_edge.add((node1, t1, node2, t2))
        self.ground_truth = ground_truth
        self.gt_num = len(gt)
        self.ground_truth_edge = ground_truth_edge

    def find_top_k(self, query_id, dirs, order, diss):
        # 读出每个视频的时长
        video_len = {}
        path = setting.NODE_FEATURES + self.folder_name + "/partition.txt"

        carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[int(query_id)]
        gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
        with open(gt_file, 'r') as f:
            gt = [line[:-1].split(',') for line in f.readlines()][self.traj_len:]

        with open(path, 'r') as f:
            for line in f.readlines():
                line = line[:-1].split(',')
                video_len[int(line[2])] = int(line[1]) - int(line[0]) + 1

        node_set = set()
        cnt = 0

        top_k = Topk(self.k)
        can_cnt = 0

        time_find_topk = time.time()
        for o_idx, i in enumerate(order):
            if i == -1:
                continue
            can_list = []
            frame, camid, idx_in_frame = get_candidate_info_by_index(i, self.video_time, self.node_num, self.traj_len)
            # camid = self.merge_dict[raw_camid]
            # camid = raw_camid

            if frame % 5 != 0:
                continue

            out_flag = False
            for node_line in gt:
                node = int(node_line[0])
                frame_s, frame_e = int(node_line[1]), int(node_line[2])
                if camid == node and (frame_s <= frame <= frame_e):
                    out_flag = True
            if out_flag:
                continue

            dis = diss[cnt]  # 这里注意下标不是i啦
            temp = Candidate(i, frame, camid, dis, cnt, idx_in_frame)

            can_list.append(temp)

            cnt += 1

            while len(can_list) != 0:
                # 该摄像头首次出现
                temp = can_list.pop(0)
                if camid not in node_set:
                    camera = Camera(camid)
                    camera.add_candidate(temp)
                    node_set.add(camid)
                    top_k.add_camera(camera)
                    # can_cnt += 1    # 这样的话top-k指k个candidate(camera+time)
                # 该摄像头已经有了，需要找到它并且往里面加信息
                else:
                    # 找到该摄像头的是top-k中哪一个
                    idx_in_topk = top_k.find_camera(temp)
                    # 如果发现和已有的间隔太大，就重新为它开一个
                    if idx_in_topk == -1:
                        camera = Camera(camid)
                        camera.add_candidate(temp)
                        top_k.add_camera(camera)
                        # can_cnt += 1  # 这样的话top-k指k个candidate(camera+time)
                    # 发现有符合的时间序列，就直接加进去
                    else:
                        # print(len(top_k.camera_list[idx_in_topk].candidate_list))
                        if len(top_k.camera_list[idx_in_topk].candidate_list) < 20:
                            top_k.add_candidate_to_camera(temp, idx_in_topk)
                        else:
                            can_cnt -= 1
                can_cnt += 1  # 这样的话top-k指k张图片
                if can_cnt >= self.k:
                    break
            if can_cnt >= self.k:
                break
        # self.false_node_set[query_i].clear()
        # self.true_node_set[query_i].clear()
        time_find_topk = time.time() - time_find_topk
        # 整理并记录top-k的信息
        top_k.camera_sort_info()
        top_k.save_top_k_info(dirs)
        # 计算top-k的平均向量
        top_k.save_avg_features(dirs, self.folder_name)
        self.time_find_topk = time_find_topk
        return top_k

    def evaluate_top_k(self, top_k):
        hit_num = 0
        for camera in top_k.camera_list:
            avg_time = int(np.mean(camera.get_timestamps()))
            if (camera.nodeid, avg_time) in self.ground_truth:
                hit_num += 1
        self.top_k_precision = hit_num / len(top_k.camera_list)
        self.top_k_recall = hit_num / self.gt_num
        self.top_k_f1 = 2 * self.top_k_precision * self.top_k_recall / (self.top_k_precision + self.top_k_recall)
        self.all_top_k_p.append(self.top_k_precision)
        self.all_top_k_r.append(self.top_k_recall)
        self.all_top_k_f1.append(self.top_k_f1)
        self.all_top_k_t.append(self.time_find_topk)

    @abstractmethod
    def path_selection(self, query_id, dirs, query_feature, graph):
        pass

    def get_freq_path(self, paths):
        # remove_set = set()
        # for p in paths:
        #     logging.info(p)
        #     for n in p:
        #         if n in self.ground_truth - remove_set:
        #             remove_set.add(n)
        # logging.info("Max hit num: %d" % len(remove_set))

        longest_paths = []
        if len(paths) == 0:
            return longest_paths

        for path in paths:
            if len(longest_paths) == 0:
                longest_paths.append(path)
            else:
                if len(path) == len(longest_paths[0]):
                    longest_paths.append(path)
                elif len(path) > len(longest_paths[0]):
                    longest_paths = [path]

        if len(longest_paths) == 1:
            final_path = longest_paths[0]
        else:
            # 开始计算每条边的频率
            edge_freq = {}
            incoming_node = {}
            all_nodes = set()
            for path in paths:
                # 统计频率
                for idx1 in range(len(path) - 1):
                    idx2 = idx1 + 1
                    n1 = (path[idx1][0], path[idx1][1])
                    n2 = (path[idx2][0], path[idx2][1])
                    edge = (n1[0], n1[1], n2[0], n2[1])
                    all_nodes.add(n1)
                    all_nodes.add(n2)
                    # 记录边的频率
                    if edge not in edge_freq.keys():
                        edge_freq[edge] = 1
                    else:
                        edge_freq[edge] += 1
                    # 记录一个点的入度信息
                    if n2 not in incoming_node.keys():
                        incoming_node[n2] = [n1]
                    else:
                        incoming_node[n2].append(n1)
            # 对所有点按照时间戳进行排序
            all_nodes = list(all_nodes)
            all_nodes.sort(key=lambda x: x[1])
            # 遍历所有点进行处理
            max_freq_table = {}
            max_freq_len = {}
            for idx, curr_node in enumerate(all_nodes):
                if idx == 0:
                    max_freq_table[curr_node] = 0
                    max_freq_len[curr_node] = [curr_node]
                else:
                    if curr_node in incoming_node.keys():
                        for pre_node in incoming_node[curr_node]:
                            curr_edge = (pre_node[0], pre_node[1], curr_node[0], curr_node[1])
                            freq = edge_freq[curr_edge]
                            new_freq_sum = max_freq_table[pre_node] + freq
                            new_path = copy.deepcopy(max_freq_len[pre_node])
                            new_path.append(curr_node)
                            if curr_node not in max_freq_table.keys():
                                max_freq_table[curr_node] = new_freq_sum
                                max_freq_len[curr_node] = new_path
                            else:
                                if new_freq_sum > max_freq_table[curr_node]:
                                    max_freq_table[curr_node] = new_freq_sum
                                    max_freq_len[curr_node] = new_path
                    else:
                        max_freq_table[curr_node] = 0
                        max_freq_len[curr_node] = [curr_node]
            # 寻找freq sum 最高的path
            max_freq_sum = -1
            max_end_node = None
            for key in max_freq_table.keys():
                freq_sum = max_freq_table[key]
                if freq_sum > max_freq_sum:
                    max_freq_sum = freq_sum
                    max_end_node = key
            final_path = max_freq_len[max_end_node]

        # remove_set = set()
        # hit_num = 0
        # for n in final_path:
        #     if n in self.ground_truth - remove_set:
        #         hit_num += 1
        #         remove_set.add(n)
        logging.info(final_path)
        # logging.info("Hit num: %d" % hit_num)
        # logging.info("*" * 40)

        return final_path

    def merge(self, query_id, dirs, paths, graph):
        logging.info('=' * 40)
        logging.info("Merging candidate paths")
        t = time.time()
        union_find = UnionFindSet(range(len(paths)))
        for idx1 in range(len(paths) - 1):
            node_set1 = set(paths[idx1])
            for idx2 in range(idx1 + 1, len(paths)):
                node_set2 = set(paths[idx2])
                if len(node_set1 & node_set2):
                    union_find.union(idx1, idx2)

        partial_paths = []
        for cid in union_find.father_node_set:
            all_path_idx = union_find.get_node(cid)
            candidates = [paths[i] for i in all_path_idx]
            ans_path = self.get_freq_path(candidates)
            heapq.heappush(partial_paths, (-len(ans_path), ans_path))

        final_path = []
        occupied_time = []
        start_time_list = []
        # logging.info("All partial path")
        while len(partial_paths):
            temp = heapq.heappop(partial_paths)[1]
            # logging.info(temp)
            s = temp[0][1]
            e = temp[-1][1]
            if len(final_path) == 0:
                final_path = temp
                occupied_time.append((s, e))
                start_time_list.append(s)
            else:
                index = bisect.bisect_right(start_time_list, s)
                if index == len(start_time_list):
                    occupied_time.append((s, e))
                    start_time_list.append(s)
                    final_path = final_path + temp
                elif index == 0:
                    occupied_time.insert(0, (s, e))
                    start_time_list.insert(0, s)
                    final_path = temp + final_path
                else:
                    prev_time = occupied_time[index - 1]
                    next_time = occupied_time[index]
                    if s > prev_time[1] and e < next_time[0]:
                        occupied_time.insert(index, (s, e))
                        start_time_list.insert(index, s)
                        final_path = final_path + temp
        final_path.sort(key=lambda x: x[1])

        min_vis, max_vis = 10000, -1
        for node in final_path:
            vis = graph.vis_dic[node]
            min_vis = min(min_vis, vis)
            max_vis = max(max_vis, vis)

        t = time.time() - t
        self.time_merge = t

        logging.info("=" * 40)
        logging.info("Merged path")
        logging.info(final_path)
        merge_file = dirs + "/../merged_path.csv"
        if not os.path.exists(merge_file):
            f = open(merge_file, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["query", "merged path"])
            csv_writer.writerow([query_id, final_path])
        else:
            f = open(merge_file, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([query_id, final_path])

        return final_path, (min_vis, max_vis)

    def extend(self, node_pivot, vis_range, partition_index, path_list, vis_range_list, reverse=False):
        if reverse:
            stride = -1
            traverse1, traverse2 = partition_index, -1
        else:
            stride = 1
            traverse1, traverse2 = partition_index, len(path_list)
        # 向后扩展
        ans_path = []
        curr_node = node_pivot
        for idx in range(traverse1, traverse2, stride):
            temp_range = vis_range_list[idx]
            temp_path = path_list[idx]

            # 判断时间是否合理
            if reverse:
                if temp_path[-1][1] >= curr_node[1]:
                    continue
            else:
                if temp_path[0][1] <= curr_node[1]:
                    continue

            # if vis_range[0] <= temp_range[0] and temp_range[1] <= vis_range[1]:
            if temp_range[1] <= vis_range[1]:
                if reverse:
                    ans_path = temp_path + ans_path
                    curr_node = temp_path[0]
                else:
                    ans_path = ans_path + temp_path
                    curr_node = temp_path[-1]
            # logging.info("%s, path vis: %s, main path range: %s" % (temp_path, temp_range, vis_range))

        logging.info("*" * 40)
        logging.info("Add: %s" % ans_path)

        return ans_path

    def refine(self, main_path, vis_range, all_paths, all_path_start_time, all_path_end_time, all_path_vis_range):
        # logging.info("=" * 40)
        # logging.info("All paths for extension:")
        # for p in all_paths:
        #     logging.info(p)

        t = time.time()
        path_sort_by_start = copy.deepcopy(all_paths)
        vis_sort_by_start = copy.deepcopy(all_path_vis_range)
        sorted_indices_start = np.argsort(all_path_start_time)
        path_sort_by_start = path_sort_by_start[sorted_indices_start]
        vis_sort_by_start = vis_sort_by_start[sorted_indices_start]

        path_sort_by_end = copy.deepcopy(all_paths)
        vis_sort_by_end = copy.deepcopy(all_path_vis_range)
        sorted_indices_end = np.argsort(all_path_end_time)
        path_sort_by_end = path_sort_by_end[sorted_indices_end]
        vis_sort_by_end = vis_sort_by_end[sorted_indices_end]

        start_time = main_path[0][1]
        end_time = main_path[-1][1]

        partition_front = bisect.bisect_right(all_path_end_time, start_time) - 1
        partition_back = bisect.bisect_left(all_path_start_time, end_time)

        logging.info("=" * 40)
        logging.info("Extend forward")
        front_path = self.extend(main_path[0], vis_range, partition_front, path_sort_by_end, vis_sort_by_end, reverse=True)

        logging.info("=" * 40)
        logging.info("Extend backward")
        back_path = self.extend(main_path[-1], vis_range, partition_back, path_sort_by_start, vis_sort_by_start)

        logging.info("=" * 40)
        logging.info("Final path:")
        ans_path = front_path + main_path + back_path
        logging.info(ans_path)

        t = time.time() - t
        self.time_refine = t
        return ans_path


    def evaluate(self, query_id, path):
        hit_num = 0
        remove_set = set()
        for item in path:
            if item in self.ground_truth - remove_set:
                hit_num += 1
                remove_set.add(item)
        precision = hit_num / len(path)
        recall = hit_num / self.gt_num
        if precision + recall > 0:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0

        evaluation_file = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/evaluation.csv' % (
            self.delta2, self.k)

        all_time = self.time_search + self.time_find_topk + self.time_build_graph + self.time_path_selection + self.time_merge

        if not os.path.exists(evaluation_file):
            f1 = open(evaluation_file, 'w')
            csv_writer = csv.writer(f1)
            csv_writer.writerow(["query", "gt_len", "ans_len", "hit_num", "precision", "recall", "F1", "time", "topk p", "topk r", "topk f1", "topk t"])
            csv_writer.writerow([query_id, self.gt_num, len(path), hit_num, precision, recall, F1, all_time, self.top_k_precision, self.top_k_recall, self.top_k_f1, self.time_find_topk])
        else:
            f1 = open(evaluation_file, 'a')
            csv_writer = csv.writer(f1)
            csv_writer.writerow([query_id, self.gt_num, len(path), hit_num, precision, recall, F1, all_time, self.top_k_precision, self.top_k_recall, self.top_k_f1, self.time_find_topk])
        return precision, recall, F1

    def save_results(self, query_id):
        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/outputs.csv' % (self.delta2, self.k)
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["query", "node", "frame"])
            csv_writer.writerow([query_id, self.ans_node_list, self.ans_time_list])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([query_id, self.ans_node_list, self.ans_time_list])

    def save_time(self, query_id):
        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/time.csv' % (self.delta2, self.k)
        all_time = self.time_search + self.time_find_topk + self.time_build_graph + self.time_path_selection
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["query", "search", "top-k", "graph", "path selection", "merge", "sum"])
            csv_writer.writerow(
                [query_id, self.time_search, self.time_find_topk, self.time_build_graph, self.time_path_selection,
                 self.time_merge, all_time])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [query_id, self.time_search, self.time_find_topk, self.time_build_graph, self.time_path_selection,
                 self.time_merge, all_time])
        self.all_inference_time.append(all_time)

    def draw_exp_figure(self):
        plt.figure(figsize=(7, 6))
        plt.grid(True, axis='y', color='#D3D3D3')

        plt.plot(self.all_top_k_f1, self.all_top_k_f1, marker='o', color='#F85CBA', linewidth=5, markersize=0)
        plt.scatter(self.all_top_k_f1, self.all_f1, marker='o', color='#6C83F1', linewidth=5, s=10, zorder=3)

        # plt.xlim(-0.02, 5.02)  # 设置x轴的范围
        # plt.ylim(70, 95)
        plt.tick_params(labelsize=14)

        # plt.legend(fontsize=16, loc=1)
        plt.xlabel("TOP-K F1", fontsize=20, labelpad=10)
        plt.ylabel("TRUST F1", fontsize=20, labelpad=12)
        plt.savefig("exp_figures/F1.png", dpi=1200, format='png')

    def process_query(self):
        query_features = np.load(setting.QUERY_FEATURES + 'query_gf.npy').astype('float32')

        # for i, query_feature in enumerate(query_features):
        for i in setting.QUERY_LIST:
            carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[int(i)]
            gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
            self.load_ground_truth(int(i))
            if self.gt_num < 3:
                continue

            self.reset()

            # logging.info("======================== Query: %03d, CarID: %d ========================" % (i, carid))
            dirs = self.output_folder + "/query_%03d" % i  # folder to store the outputs
            if not os.path.exists(dirs):
                os.mkdir(dirs)
            query_feature = np.array([query_features[i]])

            # k-nearest searching
            D, I, self.time_search = self.index_gallery.cal_all_features_dis(query_feature, dirs,
                                                                             setting.FAISS_SEARCH_SPACE)

            # determine top-k images
            top_k = self.find_top_k(i, dirs, I, D)
            self.evaluate_top_k(top_k)

            # build proximity graph
            graph = CoarseGraph(dirs, self.folder_name, gt_file, self.traj_len, i)
            t1 = graph.cal_closeness(self.edge_length)
            t2 = graph.cal_3d_score(self.delta, query_feature, self.top_k_precision, self.top_k_recall)
            self.time_build_graph = t1 + t2

            # score-based path selection
            candidate_paths, all_paths, all_path_start_time, all_path_end_time, all_path_vis_range = self.path_selection(i, dirs, query_feature, graph)

            # merge
            merged_path, vis_range = self.merge(i, dirs, candidate_paths, graph)

            # refine
            if len(merged_path):
                refined_path = self.refine(merged_path, vis_range, all_paths, all_path_start_time, all_path_end_time,
                                           all_path_vis_range)
            else:
                refined_path = []

            # evaluate
            p, r, F1 = self.evaluate(i, refined_path)
            # logging.info("TRUST —— p: %.4f, r: %.4f, f1: %.4f" % (p, r, F1))
            # logging.info("TOP-K —— p: %.4f, r: %.4f, f1: %.4f" % (self.top_k_precision, self.top_k_recall, self.top_k_f1))
            self.all_p.append(p)
            self.all_r.append(r)
            self.all_f1.append(F1)

            # record results and time files
            self.save_results(i)
            self.save_time(i)

        logging.info("=" * 40)
        logging.info(f"TRUST precision: {np.mean(self.all_p):.4f}    TOP-K precision: {np.mean(self.all_top_k_p):.4f}")
        logging.info(f"TRUST recall   : {np.mean(self.all_r):.4f}    TOP-K recall   : {np.mean(self.all_top_k_r):.4f}")
        logging.info(
            f"TRUST F1       : {np.mean(self.all_f1):.4f}    TOP-K F1       : {np.mean(self.all_top_k_f1):.4f}")
        logging.info(
            f"TRUST time     : {np.mean(self.all_inference_time):.4f}    TOP-K time     : {np.mean(self.all_top_k_t):.4f}")

        # self.draw_exp_figure()
        #
        # for i in range(len(self.all_top_k_f1)):
        #     if self.all_top_k_f1[i] > 0.4 and self.all_f1[i] < self.all_top_k_f1[i]:
        #         logging.info("Query to be checked: %d" % setting.RUN_LIST[i])
