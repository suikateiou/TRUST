import shutil
import copy
from src.topk.topk import *
from src.topk.camera import *
from src.topk.feature_gallery import *
from src.proximity_graph.coarse_graph import *
from src.common.load_pre_knowledge import *


class BasicSolver(object):
    __metaclass__ = ABCMeta

    def __init__(self, traj_len, node_num, video_time, sample_rate, k_num, delta):
        super(BasicSolver, self).__init__()
        self.traj_len = traj_len
        self.node_num = node_num
        self.video_time = video_time
        self.down_sample_fps = sample_rate
        self.k = k_num
        self.delta = delta

        # output path for trajectory recovery
        self.folder_name = "t%02d_c%03d_len%02d" % (video_time, node_num, traj_len)
        if not os.path.exists(setting.OUTPUT_PATH + self.folder_name):
            os.mkdir(setting.OUTPUT_PATH + self.folder_name)
        delta_folder = setting.OUTPUT_PATH + self.folder_name + "/delta_%.2f" % delta
        if os.path.exists(delta_folder):
            shutil.rmtree(delta_folder)
        os.mkdir(delta_folder)
        self.output_folder = delta_folder + "/top_%d" % k_num
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        # other information that should be loaded beforehand
        self.edge_length = load_dis_file(self.folder_name)
        self.pair_info = load_pairs(self.folder_name)
        self.node_dict = load_node_dict(self.folder_name)
        self.index_gallery = FeatureGallery(node_num, traj_len, video_time)

        # information of ground truth
        self.ground_truth = set()
        self.gt_num = 0

        # intermediate results
        self.selection_path = []
        self.ans_node_list = []
        self.ans_time_list = []

        # break-down time
        self.time_search = 0
        self.time_find_topk = 0
        self.time_build_graph = 0
        self.time_path_selection = 0
        self.time_merge = 0

        # overall evaluation metrics
        self.all_inference_time = []
        self.all_p = []
        self.all_r = []
        self.all_f1 = []

        # top-k related statistics
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

    def load_ground_truth(self, query_id):
        carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[query_id]
        gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
        ground_truth = set()
        with open(gt_file, 'r') as f:
            gt = [line[:-1].split(',') for line in f.readlines()][:self.traj_len]
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            for t in range(st, et + 1):
                ground_truth.add((node, t))
        self.ground_truth = ground_truth
        self.gt_num = len(gt)

    def find_top_k(self, dirs, order, diss):
        logging.info("Temporal clustering for top-k snapshots")
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

            dis = diss[cnt]  # take care of the index here
            temp = Candidate(i, frame, camid, dis, cnt, idx_in_frame)

            can_list.append(temp)

            cnt += 1

            while len(can_list) != 0:
                # this camera appears for the first time
                temp = can_list.pop(0)
                if camid not in node_set:
                    camera = Camera(camid)
                    camera.add_candidate(temp)
                    node_set.add(camid)
                    top_k.add_camera(camera)
                # this camera has appeared, find it and add snapshot to it
                else:
                    # find the exact camera (in top-k) which the snapshot belongs to
                    idx_in_topk = top_k.find_camera(temp)
                    # the time gap is too large, instantiate a new one
                    if idx_in_topk == -1:
                        camera = Camera(camid)
                        camera.add_candidate(temp)
                        top_k.add_camera(camera)
                    # there is suitable cluster, directly add
                    else:
                        # print(len(top_k.camera_list[idx_in_topk].candidate_list))
                        if len(top_k.camera_list[idx_in_topk].candidate_list) < 20:
                            top_k.add_candidate_to_camera(temp, idx_in_topk)
                        else:
                            can_cnt -= 1
                can_cnt += 1
                if can_cnt >= self.k:
                    break
            if can_cnt >= self.k:
                break

        time_find_topk = time.time() - time_find_topk
        # organize and record the top-k information
        top_k.camera_sort_info()
        top_k.save_top_k_info(dirs)
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
        if self.top_k_precision + self.top_k_recall:
            self.top_k_f1 = 2 * self.top_k_precision * self.top_k_recall / (self.top_k_precision + self.top_k_recall)
        else:
            self.top_k_f1 = 0
        self.all_top_k_p.append(self.top_k_precision)
        self.all_top_k_r.append(self.top_k_recall)
        self.all_top_k_f1.append(self.top_k_f1)
        self.all_top_k_t.append(self.time_find_topk)

    @abstractmethod
    def path_selection(self, query_id, dirs, query_feature, graph):
        pass

    def merge(self, query_id, dirs, longest_paths):
        logging.info("Merging candidate paths")
        t = time.time()
        if len(longest_paths) == 1:
            final_path = longest_paths[0]
        else:
            edge_freq = {}
            incoming_node = {}
            all_nodes = set()
            for path in longest_paths:
                # frequency statistics
                for idx1 in range(len(path) - 1):
                    idx2 = idx1 + 1
                    n1 = (path[idx1][0], path[idx1][1])
                    n2 = (path[idx2][0], path[idx2][1])
                    edge = (n1[0], n1[1], n2[0], n2[1])
                    all_nodes.add(n1)
                    all_nodes.add(n2)
                    if edge not in edge_freq.keys():
                        edge_freq[edge] = 1
                    else:
                        edge_freq[edge] += 1
                    if n2 not in incoming_node.keys():
                        incoming_node[n2] = [n1]
                    else:
                        incoming_node[n2].append(n1)

            # sort all the nodes by their timestamps
            all_nodes = list(all_nodes)
            all_nodes.sort(key=lambda x: x[1])

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
            # find the path with the max sum of frequency
            max_freq_sum = -1
            max_end_node = None
            for key in max_freq_table.keys():
                freq_sum = max_freq_table[key]
                if freq_sum > max_freq_sum:
                    max_freq_sum = freq_sum
                    max_end_node = key
            final_path = max_freq_len[max_end_node]
        t = time.time() - t
        self.time_merge = t

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

        return final_path

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
            self.delta, self.k)

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

    def process_query(self):
        query_features = np.load(setting.QUERY_FEATURES + 'query_gf.npy').astype('float32')

        for i, query_feature in enumerate(query_features):
            carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[int(i)]
            gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
            self.load_ground_truth(int(i))

            self.reset()

            logging.info("======================== Query: %03d, CarID: %d ========================" % (i, carid))
            dirs = self.output_folder + "/query_%03d" % i  # folder to store the outputs
            if not os.path.exists(dirs):
                os.mkdir(dirs)
            query_feature = np.array([query_features[i]])

            # k-nearest searching
            D, I, self.time_search = self.index_gallery.cal_all_features_dis(query_feature, dirs,
                                                                             setting.FAISS_SEARCH_SPACE)

            # determine top-k images
            top_k = self.find_top_k(dirs, I, D)
            self.evaluate_top_k(top_k)

            # build proximity graph
            graph = CoarseGraph(dirs, self.folder_name, gt_file, self.traj_len, i)
            t1 = graph.cal_closeness(self.edge_length)
            t2 = graph.cal_3d_score(self.delta, query_feature, self.top_k_precision, self.top_k_recall)
            self.time_build_graph = t1 + t2

            # score-based path selection
            candidate_paths = self.path_selection(i, dirs, query_feature, graph)

            # merge
            merged_path = self.merge(i, dirs, candidate_paths)

            # evaluate
            p, r, F1 = self.evaluate(i, merged_path)

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
