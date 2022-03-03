class Settings:
    def __init__(self):
        self.DATASET_PATH = "src/data/datasets/"

        self.DOWN_SAMPLE_FPS = 5

        self.NODE_FEATURES = '/home/wyr/carla_data/carla_dataset/node_features/'
        self.QUERY_FEATURES = '/home/wyr/carla_data/carla_dataset/query_50_feature/'

        self.FEATURE_DIM = 2048
        self.FAISS_SEARCH_SPACE = 5000

        self.OUTPUT_PATH = "src/data/outputs/"


setting = Settings()
