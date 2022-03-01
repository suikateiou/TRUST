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

        # 中间结果
        self.selection_path = []
        self.ans_node_list = []
        self.ans_time_list = []
        self.time_search = 0
        self.time_find_topk = 0
        self.time_build_graph = 0
        self.time_path_selection = 0

        # 全部结果
        self.all_inference_time = []
        self.all_p = []
        self.all_r = []
        self.all_f1 = []

    def reset(self):
        self.selection_path = []
        self.ans_node_list = []
        self.ans_time_list = []
        self.time_search = 0
        self.time_find_topk = 0
        self.time_build_graph = 0
        self.time_path_selection = 0

    def load_node_dict(self):
        node_dict = {}
        with open(setting.DATASET_PATH + "trajectory/" + self.folder_name + "/ground_truth/node.txt", "r") as f:
            for line in f.readlines():
                node = line[:-1].split(" ")
                node_dict[int(node[0])] = (float(node[1]), float(node[2]))
        return node_dict

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

    @abstractmethod
    def path_selection(self, query_id, dirs, query_feature, graph):
        pass

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
            with open(gt_file, 'r') as f:
                gt = [line[:-1].split(',') for line in f.readlines()]
            if len(gt) < 3:
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
            self.find_top_k(i, dirs, I, D)

            # build proximity graph
            graph = CoarseGraph(dirs, self.folder_name, gt_file, self.traj_len, i)
            t1 = graph.cal_closeness(self.edge_length)
            t2 = graph.cal_3d_score(self.delta, query_feature)
            self.time_build_graph = t1 + t2

            # score-based path selection
            self.path_selection(i, dirs, query_feature, graph)

            # record results and time files
            self.save_results(i)
            self.save_time(i)

        avg_p = np.mean(self.all_p)
        avg_r = np.mean(self.all_r)
        avg_f1 = np.mean(self.all_f1)
        avg_time = np.mean(self.all_inference_time)
        logging.info("="*40)
        logging.info("Average precision: %.4f" % avg_p)
        logging.info("Average recall:    %.4f" % avg_r)
        logging.info("Average F1:        %.4f" % avg_f1)
        logging.info("Average time:      %.4f" % avg_time)
