import shutil
from src.topk.topk import *
from src.topk.camera import *
from src.topk.feature_gallery import *
from src.proximity_graph.coarse_graph import *
from src.common.load_pre_knowledge import *


class BasicSolver(object):
    __metaclass__ = ABCMeta

    def __init__(self, traj_len, node_num, video_time, sample_rate, k_num, delta, lam):
        super(BasicSolver, self).__init__()
        self.traj_len = traj_len
        self.node_num = node_num
        self.video_time = video_time
        self.down_sample_fps = sample_rate
        self.k = k_num
        self.delta = delta
        self.lam = lam

        # 轨迹恢复结果路径
        self.folder_name = "t%02d_c%03d_len%02d" % (video_time, node_num, traj_len)
        if os.path.exists(setting.OUTPUT_PATH + self.folder_name):
            shutil.rmtree(setting.OUTPUT_PATH + self.folder_name)
        os.mkdir(setting.OUTPUT_PATH + self.folder_name)
        delta_folder = setting.OUTPUT_PATH + self.folder_name + "/delta_%.2f" % delta
        if not os.path.exists(delta_folder):
            os.mkdir(delta_folder)
        self.output_folder = delta_folder + "/top_%d" % k_num
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        # 轨迹恢复可视化路径
        if os.path.exists(setting.OUTPUT_VISUAL_PATH + self.folder_name):
            shutil.rmtree(setting.OUTPUT_VISUAL_PATH + self.folder_name)
        os.mkdir(setting.OUTPUT_VISUAL_PATH + self.folder_name)
        delta_folder_visual = setting.OUTPUT_VISUAL_PATH + self.folder_name + "/delta_%.2f" % delta
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
        for idx_a in range(len(gt)-1):
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

    def merge(self, dirs, paths):
        # 取出一部分
        longest_paths = []
        for path in paths:
            if len(longest_paths) == 0:
                longest_paths.append(path)
            else:
                if len(path) == len(longest_paths[0]):
                    longest_paths.append(path)
                elif len(path) > len(longest_paths[0]):
                    longest_paths = [path]
        # longest_paths = paths
        # 基本情况
        path_file = dirs + "/candidate_paths.csv"
        f1 = open(path_file, 'w')
        csv_writer = csv.writer(f1)
        csv_writer.writerow(["path", "hit num", "precision", "recall", "F1"])

        # 开始计算每条边的频率
        edge_freq = {}
        node_freq = {}
        for path in longest_paths:
            # 记录原始的多条 candidate paths
            hit_num, precision, recall, F1 = self.evaluate_single_path(path)
            f1 = open(path_file, 'a')
            csv_writer = csv.writer(f1)
            csv_writer.writerow([path, hit_num, precision, recall, F1])
            # 统计频率
            for idx1 in range(len(path)-1):
                idx2 = idx1 + 1
                edge = (path[idx1][0], path[idx1][1], path[idx2][0], path[idx2][1])
                if edge not in edge_freq.keys():
                    edge_freq[edge] = 1
                else:
                    edge_freq[edge] += 1
                n1 = (path[idx1][0], path[idx1][1])
                if n1 not in node_freq.keys():
                    node_freq[n1] = 1
                else:
                    node_freq[n1] += 1
                if idx2 == len(path) - 1:
                    n2 = (path[idx2][0], path[idx2][1])
                    if n2 not in node_freq.keys():
                        node_freq[n2] = 1
                    else:
                        node_freq[n2] += 1

        # 记录边的频率等信息
        edge_freq_file = dirs + "/edge_freq.csv"
        f2 = open(edge_freq_file, 'w')
        csv_writer = csv.writer(f2)
        csv_writer.writerow(["edge (n1, t1, n2, t2)", "label", "frequency", "ratio"])
        for key in edge_freq.keys():
            if key in self.ground_truth_edge:
                label = "gt edge"
            else:
                node_cnt = 0
                if (key[0], key[1]) in self.ground_truth:
                    node_cnt += 1
                if (key[1], key[2]) in self.ground_truth:
                    node_cnt += 1
                if node_cnt:
                    label = "contain %d gt node" % node_cnt
                else:
                    label = "noise"
            line = [key, label, edge_freq[key], edge_freq[key] / len(longest_paths)]
            f2 = open(edge_freq_file, 'a')
            csv_writer = csv.writer(f2)
            csv_writer.writerow(line)

        # 记录点的频率信息
        node_freq_file = dirs + "/node_freq.csv"
        f3 = open(node_freq_file, "w")
        csv_writer = csv.writer(f3)
        csv_writer.writerow(["node (camera, timestamp)", "label", "frequency", "ratio"])
        nodes = [key for key in node_freq.keys()]
        nodes.sort(key=lambda x: x[1])
        for node in nodes:
            if node in self.ground_truth:
                label = "GT"
            else:
                label = ""
            line = [node, label, node_freq[node], node_freq[node] / len(longest_paths)]
            f3 = open(node_freq_file, 'a')
            csv_writer = csv.writer(f3)
            csv_writer.writerow(line)

    def evaluate_single_path(self, path):
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
        return hit_num, precision, recall, F1

    def evaluate(self, query_id, paths):
        best_p, best_r, best_F1, best_hit = 0, 0, 0, 0
        best_path = []

        longest_paths = []
        for path in paths:
            if len(longest_paths) == 0:
                longest_paths.append(path)
            else:
                if len(path) == len(longest_paths[0]):
                    longest_paths.append(path)
                elif len(path) > len(longest_paths[0]):
                    longest_paths = [path]

        for path in longest_paths:
            hit_num, precision, recall, F1 = self.evaluate_single_path(path)
            if F1 > best_F1:
                best_F1 = F1
                best_p = precision
                best_r = recall
                best_path = path
                best_hit = hit_num
                self.ans_node_list = [i[0] for i in path]
                self.ans_time_list = [i[1] for i in path]

        evaluation_file = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/evaluation.csv' % (
            self.delta, self.k)

        all_time = self.time_search + self.time_find_topk + self.time_build_graph + self.time_path_selection

        if not os.path.exists(evaluation_file):
            f1 = open(evaluation_file, 'w')
            csv_writer = csv.writer(f1)
            csv_writer.writerow(["query", "gt_len", "ans_len", "hit_num", "precision", "recall", "F1", "time"])
            csv_writer.writerow([query_id, self.gt_num, len(best_path), best_hit, best_p, best_r, best_F1, all_time])
        else:
            f1 = open(evaluation_file, 'a')
            csv_writer = csv.writer(f1)
            csv_writer.writerow([query_id, self.gt_num, len(best_path), best_hit, best_p, best_r, best_F1, all_time])

        extra_info_file = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/extra.csv' % (self.delta, self.k)
        if not os.path.exists(extra_info_file):
            f2 = open(extra_info_file, 'w')
            csv_writer2 = csv.writer(f2)
            csv_writer2.writerow(["query", "path num", "non-dominated num", "precision", "recall", "F1"])
            csv_writer2.writerow([query_id, self.exceed_num, self.non_dominated, best_p, best_r, best_F1])
        else:
            f2 = open(extra_info_file, 'a')
            csv_writer2 = csv.writer(f2)
            csv_writer2.writerow([query_id, self.exceed_num, self.non_dominated, best_p, best_r, best_F1])
        return best_p, best_r, best_F1

    def save_results(self, query_id):
        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/outputs.csv' % (self.delta, self.k)
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
        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/time.csv' % (self.delta, self.k)
        all_time = self.time_search + self.time_find_topk + self.time_build_graph + self.time_path_selection
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["query", "search", "top-k", "graph", "path selection", "sum"])
            csv_writer.writerow(
                [query_id, self.time_search, self.time_find_topk, self.time_build_graph, self.time_path_selection, all_time])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [query_id, self.time_search, self.time_find_topk, self.time_build_graph, self.time_path_selection, all_time])
        self.all_inference_time.append(all_time)

    def process_query(self):
        query_features = np.load(setting.QUERY_FEATURES + 'query_gf.npy').astype('float32')
        for i, query_feature in enumerate(query_features):
            if i in setting.OMIT_LIST:
                continue

            carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[int(i)]
            gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
            self.load_ground_truth(int(i))
            if self.gt_num < 3:
                continue

            self.reset()

            logging.info("======================== Query: %03d, CarID: %d ========================" % (i, carid))
            dirs = self.output_folder + "/query_%03d" % i  # folder to store the outputs
            if not os.path.exists(dirs):
                os.mkdir(dirs)
            query_feature = np.array([query_features[i]])

            # k-nearest searching
            D, I,  self.time_search = self.index_gallery.cal_all_features_dis(query_feature, dirs, setting.FAISS_SEARCH_SPACE)

            # determine top-k images
            top_k = self.find_top_k(i, dirs, I, D)
            self.evaluate_top_k(top_k)

            # build proximity graph
            graph = CoarseGraph(dirs, self.folder_name, gt_file, self.traj_len, i)
            t1 = graph.cal_closeness(self.edge_length)
            t2 = graph.cal_3d_score(self.delta, query_feature, self.top_k_precision, self.top_k_recall)
            self.time_build_graph = t1 + t2

            # score-based path selection
            candidate_paths = self.path_selection(i, dirs, query_feature, graph)

            # merge
            # self.merge(dirs, candidate_paths)

            # evaluate
            best_p, best_r, best_F1 = self.evaluate(i, candidate_paths)
            self.all_p.append(best_p)
            self.all_r.append(best_r)
            self.all_f1.append(best_F1)

            # record results and time files
            self.save_results(i)
            self.save_time(i)

        logging.info("="*40)
        logging.info(f"TRUST precision: {np.mean(self.all_p):.4f}    TOP-K precision: {np.mean(self.all_top_k_p):.4f}")
        logging.info(f"TRUST recall   : {np.mean(self.all_r):.4f}    TOP-K recall   : {np.mean(self.all_top_k_r):.4f}")
        logging.info(f"TRUST F1       : {np.mean(self.all_f1):.4f}    TOP-K F1       : {np.mean(self.all_top_k_f1):.4f}")
        logging.info(f"TRUST time     : {np.mean(self.all_inference_time):.4f}    TOP-K time     : {np.mean(self.all_top_k_t):.4f}")
