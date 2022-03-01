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

        # 合并top-k图片时的 time duration
        self.duration = 49

        self.ground_truth = set()
        self.gt_num = 0

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

    def find_top_k(self, query_id, dirs, order, diss):
        carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[int(query_id)]
        gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
        with open(gt_file, 'r') as f:
            gt = [line[:-1].split(',') for line in f.readlines()][self.traj_len:]

        cnt = 0
        top_k = Topk(self.k)
        image_cnt = 0

        image_file = dirs + "/raw_image_info.csv"
        if not os.path.exists(image_file):
            f111 = open(image_file, 'w')
            csv_writer = csv.writer(f111)
            csv_writer.writerow(["camera ID", "timestamp", "whether GT", "whether deleted"])

        all_images = []

        time_find_topk = time.time()
        for o_idx, i in enumerate(order):
            if i == -1:
                continue
            frame, camid, idx_in_frame = get_candidate_info_by_index(i, self.video_time, self.node_num, self.traj_len)
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
            cnt += 1
            all_images.append(temp)

            if image_cnt >= self.k:  # 这样的话top-k指k张图片
                break

        # 统计删除的对错
        delete_num = 0
        delete_right = 0
        delete_wrong = 0

        # 所有图片按照时间戳排序、开始处理
        all_images.sort()
        stored_stack = []
        curr_stack = []
        del_image_set = set()
        while len(all_images):
            temp_image = all_images.pop(0)
            logging.info("frame: %d, camera: %d" % (temp_image.frame, temp_image.nodeid))
            # 当前待处理的队列为空，直接加进去
            if len(curr_stack) == 0:
                curr_stack.append(temp_image)
                logging.info("curr stack is empty, add")
            else:
                # 如果摄像头不改变
                if temp_image.nodeid == curr_stack[0].nodeid:
                    # 没有其他摄像头在前面冒出来，可以直接并进去
                    if len(stored_stack) == 0:
                        curr_stack.append(temp_image)
                        logging.info("same camera + empty stored stack, add")
                    else:
                        # 如果能新加进来的图片在规定的窗口内
                        if temp_image.frame - curr_stack[0].frame <= self.duration:
                            # 合并
                            curr_stack.append(temp_image)
                            # 把岔出去的摄像头全部删掉
                            delete_num += len(stored_stack)
                            for del_item in stored_stack:
                                del_image_set.add((temp_image.frame, temp_image.nodeid, temp_image.idx_in_frame))
                                if (del_item.nodeid, del_item.frame) in self.ground_truth:
                                    delete_wrong += 1
                                    logging.info("!!! error delete (%d, %d)" % (del_item.nodeid, del_item.frame))
                                else:
                                    delete_right += 1
                            stored_stack = []
                            logging.info("same camera, add and clear stored stack")
                        else:
                            logging.info("same camera & out of window")
                # 如果摄像头改变
                else:
                    # 当前的图片还在窗口内，先存起来
                    if temp_image.frame - curr_stack[0].frame <= self.duration:
                        stored_stack.append(temp_image)
                        logging.info("diff camera, add to stored stack")
                    # 已经超过时间窗口了
                    else:
                        # 把这个摄像头的信息转成一个top-k节点
                        camera = Camera(curr_stack[0].nodeid)
                        for item in curr_stack:
                            camera.add_candidate(item)
                        top_k.add_camera(camera)
                        curr_stack = []
                        # 回溯处理
                        for stored_image in stored_stack[::-1]:
                            all_images.insert(0, stored_image)
                        stored_stack = []
                        logging.info("diff camera & out of window, add to top-k and go back")

            # 如果主队列已经空了，但是两个栈还有没处理完的
            if len(all_images) == 0:
                if len(curr_stack):
                    # 把这个摄像头的信息转成一个top-k节点
                    camera = Camera(curr_stack[0].nodeid)
                    for item in curr_stack:
                        camera.add_candidate(item)
                    top_k.add_camera(camera)
                    curr_stack = []
                    logging.info("main stack empty, add to top-k")
                if len(stored_stack):
                    # 回溯处理
                    for stored_image in stored_stack[::-1]:
                        all_images.insert(0, stored_image)
                    stored_stack = []
                    logging.info("main stack empty, go back")

            curr_str = str([(ii.frame, ii.nodeid) for ii in curr_stack])
            store_str = str([(jj.frame, jj.nodeid) for jj in stored_stack])
            logging.info("current stack: %s" % curr_str)
            logging.info("stored  stack: %s" % store_str)
        time_find_topk = time.time() - time_find_topk
        self.time_find_topk = time_find_topk
        # 整理并记录top-k的信息
        top_k.camera_sort_info()
        top_k.save_top_k_info(dirs)
        # 计算top-k的平均向量
        top_k.save_avg_features(dirs, self.folder_name)

        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/del_images.csv' % (self.delta, self.k)
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["query", "del num", "del NOISE", "del GT"])
            csv_writer.writerow([query_id, delete_num, delete_right, delete_wrong])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([query_id, delete_num, delete_right, delete_wrong])

        # 检查原始的top-k图片
        for o_idx, i in enumerate(order):
            if i == -1:
                continue
            frame, camid, idx_in_frame = get_candidate_info_by_index(i, self.video_time, self.node_num, self.traj_len)

            out_flag = False
            for node_line in gt:
                node = int(node_line[0])
                frame_s, frame_e = int(node_line[1]), int(node_line[2])
                if camid == node and (frame_s <= frame <= frame_e):
                    out_flag = True
            if out_flag:
                continue

            if (frame, camid, idx_in_frame) in del_image_set:
                flag_del = 1
            else:
                flag_del = 0
            if (camid, frame) in self.ground_truth:
                flag_gt = 1
            else:
                flag_gt = 0

            f111 = open(image_file, 'a')
            csv_writer = csv.writer(f111)
            csv_writer.writerow([camid, frame, flag_gt, flag_del])

    @abstractmethod
    def path_selection(self, query_id, dirs, query_feature, graph):
        pass

    def evaluate(self, query_id, paths):
        best_p, best_r, best_F1, best_hit = 0, 0, 0, 0
        best_path = []

        for path in paths:
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
            D, I, self.time_search = self.index_gallery.cal_all_features_dis(query_feature, dirs, setting.FAISS_SEARCH_SPACE)

            # determine top-k images
            self.find_top_k(i, dirs, I, D)

            # build proximity graph
            graph = CoarseGraph(dirs, self.folder_name, gt_file, self.traj_len, i)
            t1 = graph.cal_closeness(self.edge_length)
            t2 = graph.cal_3d_score(self.delta, query_feature)
            self.time_build_graph = t1 + t2

            # score-based path selection
            candidate_paths = self.path_selection(i, dirs, query_feature, graph)

            # evaluate
            best_p, best_r, best_F1 = self.evaluate(i, candidate_paths)
            self.all_p.append(best_p)
            self.all_r.append(best_r)
            self.all_f1.append(best_F1)

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
