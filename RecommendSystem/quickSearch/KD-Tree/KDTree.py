import time

import numpy as np

class NodeStruct:
    def __init__(self, nodeVal, lchild=None, rchild=None):
        self.nodeVal = nodeVal
        self.lchild = lchild
        self.rchild = rchild


class KDTree:
    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features

    def KDTImpl(self, root, dataset, dim, lr):
        if not dataset:
            return
        dim = (dim+1) % self.features
        data = sorted(dataset, key=lambda x: x[dim], reverse=False)
        mid = len(data) // 2
        node = NodeStruct(data[mid])
        if lr == 0:
            root.lchild = node
            self.KDTImpl(root.lchild, data[:mid], dim, 0)
            self.KDTImpl(root.lchild, data[mid+1:], dim, 1)
        if lr == 1:
            root.rchild = node
            self.KDTImpl(root.rchild, data[:mid], dim, 0)
            self.KDTImpl(root.rchild, data[mid+1:], dim, 1)

    def initTree(self):
        if not self.dataset:
            print("数据为空.......")
            return
        dim = 0 % self.features
        data = sorted(self.dataset, key=lambda x: x[dim], reverse=False)
        mid = len(data) // 2
        root = NodeStruct(data[mid])
        self.KDTImpl(root, data[:mid], dim, 0)
        self.KDTImpl(root, data[mid+1:], dim, 1)
        print("Init KDTree Done ......")
        return root

    def preOrderTraversalTree(self, root):
        if root:
            print(root.nodeVal, end=" | ")
            self.preOrderTraversalTree(root.lchild)
            self.preOrderTraversalTree(root.rchild)

    def searchNN(self, root, data, dim, nnCount, nnList):
        # print("-----", nnList)
        dim = dim % self.features
        if root:
            # 先遍历至叶节点，再进行回溯，递归就是典型的 "回溯"
            if data[dim] < root.nodeVal[dim]:
                self.searchNN(root.lchild, data, dim+1, nnCount, nnList)
            else:
                self.searchNN(root.rchild, data, dim+1, nnCount, nnList)

            # 使用欧式距离进行度量
            distance = np.sqrt(np.sum(np.square(np.array(data)-np.array(root.nodeVal))))
            # 若叶节点，则直接将元素进行添加
            if len(nnList) == 0:
                nnList.append((round(distance, 8), root.nodeVal))
            else:
                if distance < nnList[-1][0]:  # 则说明另一侧区域可能存在最近邻值，需要探索；当前节点的距离比结果列表中的最大距离还小
                    if len(nnList) < nnCount:
                        nnList.append((round(distance, 8), root.nodeVal))
                    else:
                        nnList[-1] = (round(distance, 8), root.nodeVal)
                    # 探索另外一个区域,先对结果进行排序
                    nnList.sort(key=lambda x: x[0])
                    if data[dim] < root.nodeVal[dim]:
                        self.searchNN(root.rchild, data, dim+1, nnCount, nnList)  # 若先前探索左区域，则现在探索右区域；反之亦然
                    else:
                        self.searchNN(root.lchild, data, dim+1, nnCount, nnList)
                else:  # 直接回溯; 当前节点与目标形成的距离比结果列表的最大距离还要大，则说明该节点另外区域不存在最近邻的值了
                    if len(nnList) < nnCount:  # 如果结果列表个数不够，直接添加；如果足够，不做修改
                        nnList.append((round(distance, 8), root.nodeVal))
            # nnList 排序，便于下一步进行替换
            nnList.sort(key=lambda x: x[0])
        return nnList


if __name__ == '__main__':
    start = time.process_time()
    dataSource = [np.random.randint(1, 100, 200).tolist() for _ in range(1000000)]
    # dataSource = [(2, 3, 100), (5, 4, 70), (9, 6, 55), (4, 7, 200), (8, 1, 44), (7, 2, 0)]
    kdt = KDTree(dataSource, len(dataSource[0]))
    root = kdt.initTree()
    second = time.process_time()
    print("构建KDTree时间%.3f" % (second-start))  # 占据99%时间
    nnList = kdt.searchNN(root, np.random.randint(1, 100, 200).tolist(), 0, 100, [])
    end = time.process_time()
    print("搜索KDTree时间%.3f" % (end-second))  # 占据1%时间
    print("程序运行时间{}".format(end-start))
    print([ele[0] for ele in nnList])
