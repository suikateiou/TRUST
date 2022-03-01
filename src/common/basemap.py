import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.setting import setting


class BaseMap(object):
    def __init__(self, folder_name):
        super(BaseMap, self).__init__()
        nodeosm_path = 'dataset3_0530/trajectory/%s/ground_truth/node.txt' % folder_name
        edgeosm_path = 'dataset3_0530/trajectory/%s/ground_truth/edge.txt' % folder_name
        big_map_path = setting.BIG_MAP
        self.adj, self.graph, self.node_dic, self.coordinate_dic = self._load_map(nodeosm_path, edgeosm_path)
        self.big_map = self._load_big_map(big_map_path)

    def _load_map(self, nodeosm_path, edgeosm_path):
        node_dic = {}
        coordinate_dic = {}
        nodeosm = pd.read_csv(nodeosm_path, header=None, sep=' ')
        nodeosm = nodeosm.values.tolist()

        for node in nodeosm:
            node_dic[int(node[0])] = (round(float(node[2]), 7), round(float(node[1]), 7))
            coordinate_dic[str(node_dic[int(node[0])])] = int(node[0])

        edgeosm = pd.read_csv(edgeosm_path, sep=' ', header=None, usecols=[0, 1, 2, 3])
        node_num = len(nodeosm)
        adj = np.zeros((node_num + 1, node_num + 1))
        adj[edgeosm[1], edgeosm[2]] = edgeosm[3]
        adj[edgeosm[2], edgeosm[1]] = edgeosm[3]

        graph = nx.Graph()
        for i in nodeosm[0]:
            graph.add_node(int(i))
        data = []
        for i, row in edgeosm.iterrows():
            data.append((int(row[1]), int(row[2]), row[3]))
        graph.add_weighted_edges_from(data)

        return adj, graph, node_dic, coordinate_dic

    def _load_big_map(self, big_map_path):
        graph = nx.Graph()
        edges = pd.read_csv(big_map_path, sep=',', header=None)
        edges = edges.values.tolist()
        for edge in edges:
            node1 = str((float(edge[1][1:]), float(edge[2][1:-1])))
            node2 = str((float(edge[3][1:]), float(edge[4][1:-1])))
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(node1, node2)
        return graph

    def get_nth_order_neighbors(self, nodeid, nth_order):
        name = str(self.node_dic[nodeid])
        ans = set()
        if nth_order <= 0:
            return None
        elif nth_order == 1:
            try:
                nb = self.big_map.neighbors(name)
                coordinate_ans = set(nb)
            except nx.exception.NetworkXError:
                coordinate_ans = set()
        else:
            try:
                nb = self.big_map.neighbors(name)
                coordinate_ans = set(nb)
            except nx.exception.NetworkXError:
                coordinate_ans = set()
            for i in range(1, nth_order+1):
                tmp = set()
                for j in coordinate_ans:
                    neighbors = set(self.big_map.neighbors(j))
                    tmp = tmp.union(neighbors)
                coordinate_ans = tmp
        for cor in coordinate_ans:
            if cor in self.coordinate_dic.keys():
                temp = self.coordinate_dic[cor]
                if temp != nodeid:
                    ans.add(temp)
        return ans

    def get_n_neighbors(self, nodeid, n_num):
        ans = set()
        nth_order = 0
        while len(ans) < n_num:
            nth_order += 1
            if nth_order <= 50:
                ans = self.get_nth_order_neighbors(nodeid, nth_order)
            else:
                break
        return ans

    def get_shortest_distance(self, node_id1, node_id2):
        return nx.shortest_path_length(self.graph, source=node_id1, target=node_id2)

    def get_shortest_path_with_gap_distance(self, node_id1, node_id2):
        path = nx.shortest_path(self.graph, source=node_id1, target=node_id2)
        diss = []
        for i in range(len(path) - 1):
            s, d = path[i], path[i + 1]
            diss.append(self.adj[s][d])
        return path, diss

    def get_hop(self, node1, node2):
        name1 = str(self.node_dic[node1])
        name2 = str(self.node_dic[node2])
        try:
            ans = nx.shortest_path(self.big_map, source=name1, target=name2)
            return ans
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def get_straight_dis(self, node1, node2):
        return self.adj[node1, node2]

    def get_nodes(self):
        return self.graph.nodes()

    def get_coordinate_map(self):
        return self.coordinate_dic
