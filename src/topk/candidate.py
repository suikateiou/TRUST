import csv
import os
import re
from src.common.load_feature import *


class Candidate(object):
    def __init__(self, feature_raw_idx, frame, nodeid, dis, rank, idx_in_frame):
        super(Candidate, self).__init__()
        self.feature = feature_raw_idx
        self.frame = frame
        self.nodeid = nodeid
        self.idx_in_frame = idx_in_frame
        self.dis = dis
        self.rank = rank

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
