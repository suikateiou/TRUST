import numpy as np
import bisect
from src.setting import setting


def find_new_index(raw_index, video_time, node_num, traj_len):
    path = setting.NODE_FEATURES + "t%02d_c%03d_len%02d/partition.txt" % (video_time, node_num, traj_len)
    with open(path, 'r') as f:
        records = [line[:-1].split(',') for line in f.readlines()]
        start = [int(line[0]) for line in records]
    pivot_index = bisect.bisect_right(start, raw_index) - 1
    new_index = raw_index - start[pivot_index]
    file_id = int(records[pivot_index][2])
    return file_id, new_index


def get_candidate_info_by_index(raw_index, video_time, node_num, traj_len):
    file_id, new_index = find_new_index(raw_index, video_time, node_num, traj_len)
    folder_name = "t%02d_c%03d_len%02d" % (video_time, node_num, traj_len)
    prefix = setting.NODE_FEATURES + folder_name + '/%d' % file_id

    with open(prefix + '/frame_%d.wyr' % file_id, 'rb') as f1:
        f1.seek(1 * 8 * new_index)
        data1 = f1.read(1 * 8)
        frame = np.frombuffer(data1, dtype=np.int64)[0]

    camid = file_id

    with open(prefix + '/idx_in_frame_%d.wyr' % file_id, 'rb') as f2:
        f2.seek(1 * 8 * new_index)
        data2 = f2.read(1 * 8)
        idx_in_frame = np.frombuffer(data2, dtype=np.int64)[0]
    return frame, camid, idx_in_frame


def get_feaure_by_index(raw_index, video_time, node_num, traj_len):
    file_id, new_index = find_new_index(raw_index, video_time, node_num, traj_len)
    folder_name = "t%02d_c%03d_len%02d" % (video_time, node_num, traj_len)
    return get_feature_by_file_index(file_id, new_index, folder_name)


def get_feature_by_file_index(file_id, new_index, folder_name):
    filename = setting.NODE_FEATURES + folder_name + "/%d/gf_%d.wyr" % (file_id, file_id)
    f = open(filename, "rb")
    f.seek(2048 * 4 * new_index)
    data = f.read(2048 * 4)
    feature = np.frombuffer(data, dtype=np.float32)
    return feature


def load_avg_feature(filename, index):
    f = open(filename, "rb")
    f.seek(setting.FEATURE_DIM * 4 * index)
    data = f.read(setting.FEATURE_DIM * 4)
    feature = np.frombuffer(data, dtype=np.float32)
    return feature

