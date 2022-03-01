import faiss
import random
import time
import os
import numpy as np
import logging
from src.setting import setting


class FeatureGallery(object):
    def __init__(self, node_num, traj_len, video_time):
        super(FeatureGallery, self).__init__()
        self.node_num = node_num
        self.traj_len = traj_len
        self.video_time = video_time
        self.folder_name = "t%02d_c%03d_len%02d" % (video_time, node_num, traj_len)
        self.data_path = setting.NODE_FEATURES + self.folder_name
        self.nlist = 100
        self.m = 32

    def load_gallery_data(self, i, index, d):
        gf_file = self.data_path + '/%d/gf_%d.wyr' % (i, i)
        features = np.fromfile(gf_file, dtype=np.float32)
        features.shape = -1, d
        index.add(features)
        return index

    def pick_train_data(self, node, d, line_cnt, train_data, global_cnt, limit):
        # 加载 gallery 文件
        gf_file = self.data_path + '/%d/gf_%d.wyr' % (node, node)
        features = np.fromfile(gf_file, dtype=np.float32)
        features.shape = -1, d
        with open(self.data_path + "/partition.txt", 'a') as f:
            content = "%d,%d,%d\n" % (line_cnt, line_cnt + features.shape[0] - 1, node)
            f.write(content)
        line_cnt += features.shape[0]

        # 随机选取 train 数据
        num = round(features.shape[0] / 5)
        cnt = 0
        ans = set()
        while cnt < num:
            temp = random.randint(0, features.shape[0] - 1)
            if temp not in ans:
                ans.add(temp)
                cnt += 1

        local_cnt = 0
        local_train_data = np.zeros(shape=[num, d]).astype('float32')
        for i in ans:
            local_train_data[local_cnt] = features[i]
            local_cnt += 1
            if global_cnt < limit:
                train_data[global_cnt] = features[i]
                global_cnt += 1
            else:
                break

        # 保存二进制文件
        train_data_byte = local_train_data.tobytes()
        output_folder = self.data_path + '/%d' % node
        if os.path.exists(output_folder + '/train_data_%d.wyr' % node):
            os.remove(output_folder + '/train_data_%d.wyr' % node)
        with open(output_folder + '/train_data_%d.wyr' % node, "wb") as f:
            f.write(train_data_byte)

        return train_data, line_cnt, global_cnt

    def cal_train_data_num(self):
        all_feature_num = 0
        for node in range(self.node_num):
            frame_file = self.data_path + "/%d/frame_%d.wyr" % (node, node)
            frames = np.fromfile(frame_file, dtype=np.int64)
            all_feature_num += len(frames)
        return all_feature_num // 5

    def build_index(self):
        logging.basicConfig(level=logging.INFO, filename='src/log/build-index.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        dst = self.data_path + '/train_data.wyr'
        if os.path.exists(dst):
            return 0, 0, 0

        d = setting.FEATURE_DIM

        t = time.time()
        all_feature_num = self.cal_train_data_num()
        t = time.time() - t
        logging.info("cal all feature num %d with time %f" % (all_feature_num, t))

        # 挑选训练数据
        logging.info("Merging training features")
        time_pick_train_data = time.time()
        train_data = np.zeros(shape=[all_feature_num, d]).astype('float32')
        line_cnt = 0
        global_cnt = 0
        for node in range(self.node_num):
            train_data, line_cnt, global_cnt = self.pick_train_data(node, d, line_cnt, train_data, global_cnt, all_feature_num)
            logging.info("node %d is ok" % node)
        time_pick_train_data = time.time() - time_pick_train_data
        logging.info("Picking and merging time: %f" % time_pick_train_data)

        # 保存数据
        logging.info("Size of train data: %7d" % len(train_data))
        save_train_data = train_data.tobytes()

        with open(dst, "wb") as f:
            f.write(save_train_data)

        # 建立索引
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, 8)
        logging.info("Training index")
        time_train_index = time.time()
        index.train(train_data)
        time_train_index = time.time() - time_train_index
        logging.info("Training time: %f" % time_train_index)

        logging.info("Loading gallery features and adding index")
        time_build_index = time.time()
        for i in range(self.node_num):
            index = self.load_gallery_data(i, index, d)
            logging.info("node %d is ok" % i)
        time_build_index = time.time() - time_build_index
        faiss.write_index(index, self.data_path + "/features.index")
        return time_pick_train_data, time_train_index, time_build_index

    def cal_all_features_dis(self, query_feature, dirs, k):
        # logging.info("Searching with faiss index")
        time_search = time.time()
        index = faiss.read_index(self.data_path + "/features.index")
        D, I = index.search(query_feature, k)
        # 按照距离升序排序
        D = D[0]
        I = I[0]
        sorted_indices = np.argsort(D)
        D = D[sorted_indices]
        I = I[sorted_indices]
        time_search = time.time() - time_search
        np.save(dirs + '/original_dis.npy', D)
        np.save(dirs + '/order_by_index.npy', I)
        D = (D - D.min()) / (D.max() - D.min())
        np.save(dirs + '/normalized_dis.npy', D)
        return D, I, time_search
