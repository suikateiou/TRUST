class Settings:
    def __init__(self):
        # 数据集
        self.DATASET_PATH = "src/data/datasets/"

        # 视频帧率
        self.DOWN_SAMPLE_FPS = 5

        self.NODE_FEATURES = '/home/wyr/carla_data/carla_dataset/node_features/'
        self.QUERY_FEATURES = '/home/wyr/carla_data/carla_dataset/query_50_feature/'

        self.FEATURE_DIM = 2048
        self.FAISS_SEARCH_SPACE = 5000

        # ground truth 是某一辆车的轨迹
        self.GROUND_TRUTH = '/'

        # 算法输出结果的路径
        self.OUTPUT_PATH = "src/data/outputs/"
        self.OUTPUT_VISUAL_PATH = "src/data/visual_outputs/"

        # 建立 graph 和连 path 时用到的阈值
        self.DISCOUNTING_FACTOR = 0.95

        # self.QUERY_LIST = [1, 2, 3, 5, 10, 11, 12, 14, 16, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 31, 32, 35, 36, 37,
        #                    38, 40, 42, 44, 46, 49]

        # self.QUERY_LIST = [1, 2, 3, 5, 10, 11, 12, 14, 16, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 31, 35, 36, 37,
        #                    38, 40, 42, 44, 46]

        # self.QUERY_LIST = range(50)

        # c600
        self.QUERY_LIST = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 23, 24, 26, 27, 29, 30, 31,
                           32, 33, 35, 40, 41, 42, 44, 46, 48, 49]

        # c550
        # self.QUERY_LIST = [1, 3, 4, 7, 8, 10, 11, 15, 16, 17, 18, 19, 20, 25, 26, 27, 29, 30, 31, 33, 35, 37, 38, 40,
        #                    41, 42, 44, 48, 49]

        # 650
        # self.QUERY_LIST = [1, 5, 6, 7, 8, 11, 12, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42, 43, 44, 45, 49]

        self.K = [150]
        self.DELTA = [0.3]


setting = Settings()
print(len(setting.QUERY_LIST))
