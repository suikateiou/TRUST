class UnionFindSet(object):
    def __init__(self, data_list):
        self.data = set(data_list)
        self.father_dict = {}
        self.size_dict = {}
        self.father_node_set = set(data_list)
        self.clusters = {}

        for node in data_list:
            self.father_dict[node] = node
            self.size_dict[node] = 1
            self.clusters[node] = {node}

    def find_head(self, node):
        father = self.father_dict[node]
        if node != father:
            father = self.find_head(father)
        self.father_dict[node] = father
        return father

    def is_same_set(self, node_a, node_b):
        return self.find_head(node_a) == self.find_head(node_b)

    def union(self, node_a, node_b):
        if node_a is None or node_b is None:
            return None

        a_head = self.find_head(node_a)
        b_head = self.find_head(node_b)

        if a_head != b_head:
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if a_set_size >= b_set_size:
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
                self.father_node_set.remove(b_head)
                self.clusters[a_head] = self.clusters[a_head].union(self.clusters[b_head])
                self.clusters[b_head] = set()
                return b_head  # remove head
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size
                self.father_node_set.remove(a_head)
                self.clusters[b_head] = self.clusters[b_head].union(self.clusters[a_head])
                self.clusters[a_head] = set()
                return a_head

    def get_node_naive(self, father_node):
        ans = set()
        for node in self.data:
            if self.find_head(node) == father_node:
                ans.add(node)
        return ans

    def get_node(self, father_node):
        return self.clusters[father_node]
