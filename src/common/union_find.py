class UnionFindSet(object):
    """并查集"""
    def __init__(self, data_list):
        """初始化两个字典，一个保存节点的父节点，另外一个保存父节点的大小
        初始化的时候，将节点的父节点设为自身，size设为1"""
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
        """使用递归的方式来查找父节点
        在查找父节点的时候，顺便把当前节点移动到父节点上面
        这个操作算是一个优化
        """
        father = self.father_dict[node]
        if node != father:
            father = self.find_head(father)
        self.father_dict[node] = father
        return father

    def is_same_set(self, node_a, node_b):
        """查看两个节点是不是在一个集合里面"""
        return self.find_head(node_a) == self.find_head(node_b)

    def union(self, node_a, node_b):
        """将两个集合合并在一起"""
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


if __name__ == '__main__':
    a = [1,2,3,4,5]
    union_find_set = UnionFindSet(a)
    union_find_set.union(1,2)
    union_find_set.union(3,5)
    union_find_set.union(3,1)
    print(union_find_set.is_same_set(2,5))  # True
    print(union_find_set.get_node(union_find_set.find_head(1)))
    print(union_find_set.get_node_naive(union_find_set.find_head(1)))
    print(union_find_set.father_node_set)
