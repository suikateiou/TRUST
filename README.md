<div align=center>
<img src="https://trajectory-recovery.oss-cn-hangzhou.aliyuncs.com/logo.png" alt="logo" style="width:50%;" />
</div>


## Visual Path Inference in Urban-Scale Camera Network

This is the source code of TRUST, a new path inference algorithm tailored for query issued against an urban-scale video database. Given a query image containing the target object, our goal is to recover its historical trajectory from the footprints captured by the surveillance cameras deployed on a road network.  The input  is an image of query vehicle and the output is a sequence of camera ids together with their corresponding timestamps. 

<div align=center>
<img src="https://trajectory-recovery.oss-cn-hangzhou.aliyuncs.com/example.png" style="width:80%;" />
</div>


## Architecture

Top-k retrieval is applied first to generate a set of uncertain images that are visually similar to the target object.We then construct a proximity graph with spatial, temporal and visual similarity to eliminate conflicting candidates and reduce inference space. Afterwards, we propose a filter-and-aggregate framework for path inference.

The file tree organization is shown as:

```
.
├── run.py							# the top module
└── src								# some important functions and data
    ├── data
    │   ├── datasets				# different datasets, take CityFlow for example
    │   │   ├── cityflow
    │   │   │   ├── groundtruth		# groundtruth trajectory of each query vehicle
    │   │   │   ├── node_features	# features extracted and indexed from each camera video
    │   │   │   ├── query			# image and its feature of each query vehicle
    │   │   │   │   ├── features
    │   │   │   │   └── images
    │   │   │   └── roadnetwork		# roadnetwork information of each dataset
    │   │   └── ...
    │   └── outputs					# intermediate, final, and evaluation resutls
    ├── log							# runtime log
    ├── common						# some common utils
    ├── proximity_graph				# proximity graph with 3-dimension scoring
    ├── setting.py					# some basic configurations
    ├── topk						# top-k retrieval of the most visually similar snapshots
    └── traj_recovery				# path filter-and-aggregate framework
```

Node features of each camera are not the complete version here due to the large scale, but they are available [here](https://pan.baidu.com/s/12q_5VThVLze6dQ5MzTZamw) with code `4sj0`.



## Installation

Dependencies are listed and you can install them via Anaconda or Pip.

```
conda create --name trust --file conda.yml
pip install -r requirements.txt
```

You can also refer to the fllowing instructions:

```
conda create -n trust python=3.7
conda activate trust
conda install absl-py
conda install faiss-gpu cudatoolkit=10.1 -c pytorch
conda install -c conda-forge geopy
conda install matplotlib
conda install networkx
conda install -c menpo opencv
conda install pandas
conda install scipy
conda install statsmodels
conda install scikit-learn
conda install folium
```



## Dataset format

Different datasets are named in the standard format as ` t[xx]_c[xxx]_len[xx]` after their video time, camera number, and trajectort length. Corresponding groundruth, query features, camera features and index should be put in the correct directory.  The file tree organization of `node_features` should be:

```
.
├── 1
│   ├── frame_1.wyr
│   ├── gf_1.wyr
│   └── idx_in_frame_1.wyr
├── 2
├── ...
├── features.index
└── partition.txt
```

Under the folder named after `cameraID` such as `1`, there are 3 feature files (in binary form) . Each record in the same line of the 3 files refers to the same vehicle, and them indicate the `frameID` ,  `2048-d feature`, and `index in that frame` respectively.

`features.index` is the index file built for all the features extracted from the cameras above. `partition.txt` is the reversed index to quickly find a certain feature belongs to which camera.



## Run the Code

To start TRUST:

```
CUDA_VISIBLE_DEVICES=0 python run.py --fps 10 --traj_len 5 --node_num 30 --video_time 10 --k 80 --delta 0.45
```

There are several arguments to be assigned:

|  Argument  |                     Definition                      |
| :--------: | :-------------------------------------------------: |
|    fps     | video sampling rate (10 for CityFlow, 5 for others) |
|  traj_len  |   length of groundtruth trajectory in the dataset   |
|  node_num  |          number of cameras in the dataset           |
| video_time |         length of each video in the dataset         |
|     k      |               scale of top-k retrival               |
|   delta    |  threshold for proximity graph and path selection   |

Noted that `traj_len`, `node_num`, and `video_time` are used only for determining dataset.



## Contact

If you have any questions, you can contact us by wangyingrong@zju.edu.cn.

