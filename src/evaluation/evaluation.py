import re
import csv
import os
import logging
import numpy as np
import pandas as pd
from src.setting import setting

logging.basicConfig(
    level=logging.INFO,
    filename='src/log/evaluation.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Evaluator(object):
    def __init__(self, traj_len, node_num, video_time, k, delta):
        super(Evaluator, self).__init__()
        self.folder_name = "t%02d_c%03d_len%02d" % (video_time, node_num, traj_len)
        self.traj_len = traj_len
        self.node_num = node_num
        self.video_time = video_time
        self.k = k
        self.delta = delta

    def path_evaluation(self):
        logging.info ("========== %s & top-%d & delta-%.2f ==========" % (self.folder_name, self.k, self.delta))
        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/outputs.csv' % (
        self.delta, self.k)
        all_hit, all_traj, all_gt = 0, 0, 0
        paths = pd.read_csv (filename, header=0)
        precision_list, recall_list = [], []
        for i, row in paths.iterrows ():
            [query, nodes, times] = row
            # 读取输出的结果
            node_list = re.split (r"[, ]", nodes[1:-1])
            node_list = list (filter (lambda x: x != '', node_list))
            time_list = re.split (r"[, ]", times[1:-1])
            time_list = list (filter (lambda x: x != '', time_list))
        
            # 读取 ground truth
            carid = np.load (setting.QUERY_FEATURES + 'query_carid.npy')[int (query)]
            gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
            ground_truth = set ()
            with open (gt_file, 'r') as f:
                gt = [line[:-1].split (',') for line in f.readlines ()]
            for content in gt:
                node, st, et = int (content[0]), int (content[1]), int (content[2])
                for t in range (st, et + 1):
                    ground_truth.add ((t, node))
        
            # 比较结果
            hit = 0
            for i in range (len (node_list)):
                nodeid = int (node_list[i])
                frame = int (time_list[i])
                if (frame, nodeid) in ground_truth:
                    hit += 1
            recall = hit / len (gt)
            if len (node_list) != 0:
                precision = hit / len (node_list)
            else:
                precision = 0
            precision_list.append (precision)
            recall_list.append (recall)
            all_hit += hit
            all_traj += len (node_list)
            all_gt += len (gt)
        
            # 记录数据
            evaluation_file = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/evaluation.csv' % (
            self.delta, self.k)
            if not os.path.exists (evaluation_file):
                f1 = open (evaluation_file, 'w')
                csv_writer = csv.writer (f1)
                csv_writer.writerow (["query", "gt_len", "ans_len", "hit_num", "precision", "recall"])
                csv_writer.writerow ([query, self.traj_len, len (node_list), hit, precision, recall])
            else:
                f1 = open (evaluation_file, 'a')
                csv_writer = csv.writer (f1)
                csv_writer.writerow ([query, self.traj_len, len (node_list), hit, precision, recall])
    
        # avg_precision = all_hit / all_traj
        avg_precision = np.mean (precision_list)
        logging.info ("avg precision: %.4f" % avg_precision)
        # avg_recall = all_hit / all_gt
        avg_recall = np.mean (recall_list)
        logging.info ("avg recall: %.4f\n\n" % avg_recall)
        return avg_precision, avg_recall
    
    def path_evaluation_no_inference(self):
        logging.info("========== %s & top-%d & delta-%.2f ==========" % (self.folder_name, self.k, self.delta))
        filename = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/outputs_no_inference.csv' % (self.delta, self.k)
        all_hit, all_traj, all_gt = 0, 0, 0
        paths = pd.read_csv(filename, header=0)
        precision_list, recall_list = [], []
        for i, row in paths.iterrows():
            [query, nodes, times] = row
            # 读取输出的结果
            node_list = re.split(r"[, ]", nodes[1:-1])
            node_list = list(filter(lambda x: x != '', node_list))
            time_list = re.split(r"[, ]", times[1:-1])
            time_list = list(filter(lambda x: x != '', time_list))

            # 读取 ground truth
            carid = np.load(setting.QUERY_FEATURES + 'query_carid.npy')[int(query)]
            gt_file = setting.DATASET_PATH + "video_gt/%s/%d.txt" % (self.folder_name, carid)
            ground_truth = set()
            with open(gt_file, 'r') as f:
                gt = [line[:-1].split(',') for line in f.readlines()]
            for content in gt:
                node, st, et = int(content[0]), int(content[1]), int(content[2])
                for t in range(st, et+1):
                    ground_truth.add((t, node))

            # 比较结果
            hit = 0
            for i in range(len(node_list)):
                nodeid = int(node_list[i])
                frame = int(time_list[i])
                if (frame, nodeid) in ground_truth:
                    hit += 1
            recall = hit / len(gt)
            if len(node_list) != 0:
                precision = hit / len(node_list)
            else:
                precision = 0
            precision_list.append(precision)
            recall_list.append(recall)
            all_hit += hit
            all_traj += len(node_list)
            all_gt += len(gt)

            # 记录数据
            evaluation_file = setting.OUTPUT_PATH + self.folder_name + '/delta_%.2f/top_%d/evaluation.csv' % (self.delta, self.k)
            if not os.path.exists(evaluation_file):
                f1 = open(evaluation_file, 'w')
                csv_writer = csv.writer(f1)
                csv_writer.writerow(["query", "gt_len", "ans_len", "hit_num", "precision", "recall"])
                csv_writer.writerow([query, self.traj_len, len(node_list), hit, precision, recall])
            else:
                f1 = open(evaluation_file, 'a')
                csv_writer = csv.writer(f1)
                csv_writer.writerow([query, self.traj_len, len(node_list), hit, precision, recall])

        # avg_precision = all_hit / all_traj
        avg_precision = np.mean(precision_list)
        logging.info("avg precision: %.4f" % avg_precision)
        # avg_recall = all_hit / all_gt
        avg_recall = np.mean(recall_list)
        logging.info("avg recall: %.4f\n\n" % avg_recall)
        return avg_precision, avg_recall