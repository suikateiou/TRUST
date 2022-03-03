from src.topk.candidate import *


class Camera(object):
    def __init__(self, nodeid):
        super(Camera, self).__init__()
        self.nodeid = nodeid
        self.candidate_list = []
        self.idx_in_topk = -1

    def add_candidate(self, candidate):
        self.candidate_list.append(candidate)

    def get_nodeid(self):
        return self.nodeid

    def set_idx_in_top_k(self, idx):
        self.idx_in_topk = idx

    def get_idx_in_top_k(self):
        return self.idx_in_topk

    def sort_info(self):
        self.candidate_list.sort(key=lambda x: x.frame)

    def get_timestamps(self):
        return [can.get_frame() for can in self.candidate_list]

    def get_duration_time(self):
        timeline = [can.get_frame() for can in self.candidate_list]
        return np.min(timeline), np.max(timeline)

    def save_frame_and_feature(self, filename):
        timestamps = []
        feature_raw_index = []
        inx_in_frame = []
        for can in self.candidate_list:
            timestamps.append(can.get_frame())
            feature_raw_index.append(can.get_feature_raw_index())
            inx_in_frame.append(can.get_idx_in_frame())
        avg_frame = sum(timestamps) // len(timestamps)
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["camid", "avg timestamp", "appear times", "timestamps", "feature index", "index in frame"])
            csv_writer.writerow([self.nodeid, avg_frame, len(timestamps), timestamps, feature_raw_index, inx_in_frame])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([self.nodeid, avg_frame, len(timestamps), timestamps, feature_raw_index, inx_in_frame])

    def get_avg_feature(self, folder_name):
        all_num = len(self.candidate_list)
        avg_feature = self.candidate_list[(all_num-1) // 2].get_feature(folder_name)
        return avg_feature
