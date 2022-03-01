import csv
import os
import re
from src.common.load_feature import *


# 每一张备选物体对应这样一个对象
class Candidate(object):
    def __init__(self, feature_raw_idx, frame, nodeid, dis, rank, idx_in_frame, node_index_no=-1, index_no=-1):
        super(Candidate, self).__init__()
        self.feature = feature_raw_idx  # 2048维度的向量在原始文件中的索引
        self.frame = frame  # 该图帧号（时间）
        self.nodeid = nodeid  # 对应的点编号
        self.idx_in_frame = idx_in_frame  # 这个物体在其所在帧中的编号
        self.dis = dis  # 和query图片的距离
        self.rank = rank  # feature距离的排名
        self.node_index_no = node_index_no  # 在单个node里feature的序号
        self.index_no = index_no  # 单独为keypoint准备的，因为keypoint是在全局搜索的

    def get_img_name(self):
        return 'c%04d_%07d_%07d.jpg' % (self.nodeid, self.frame, self.idx_in_frame)

    def save_info(self, dirs):
        if os.path.exists(dirs + '/top_k.csv'):
            f = open(dirs + '/top_k.csv', 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([self.frame, self.nodeid, self.dis, self.rank, self.idx_in_frame])
        else:
            f = open(dirs + '/top_k.csv', 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame", "nodeid", "dis", "rank", "idx_in_frame"])
            csv_writer.writerow([self.frame, self.nodeid, self.dis, self.rank, self.idx_in_frame])

    def get_frame(self):
        return self.frame

    def get_feature_raw_index(self):
        return self.feature

    def get_feature(self, folder_name):
        pattern = re.compile(r't([1-9][0-9]*|0)_c0*([1-9][0-9]*|0)_len0*([1-9][0-9]*|0)')
        video_time, node_num, traj_len = map(int, pattern.search(folder_name).groups())
        return get_feaure_by_index(self.feature, video_time, node_num, traj_len)
        # return self.feature

    def get_nodeid(self):
        return self.nodeid

    def get_idx_in_frame(self):
        return self.idx_in_frame

    def get_info(self):
        return self.nodeid, self.frame, self.idx_in_frame

    def __lt__(self, other):
        return self.frame < other.frame
