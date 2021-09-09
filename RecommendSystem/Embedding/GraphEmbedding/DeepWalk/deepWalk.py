import itertools
import random
import time
import networkx as nx
from gensim.models import Word2Vec
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class DeepWalk:
    def __init__(self, dataset, walkNums, walkLen, vecDim):
        super(DeepWalk, self).__init__()
        self.dataset = dataset
        self.walkNums = walkNums
        self.walkLen = walkLen
        self.vecDim = vecDim


    def generateGraph(self):
        ''' 将原始列表变成图的形式
        :param dataset: 数据集[[]]
        :return: Graph
        '''
        nodes = []
        edges = []
        for row in range(len(self.dataset)):
            nodes += self.dataset[row]
            for col in range(1, len(self.dataset[row])):
                edges.append((self.dataset[row][col-1], self.dataset[row][col]))
        nodes = list(set(nodes))
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        print("Graph build done.")
        return G
        # nx.draw(G, with_labels=True)
        # plt.show()


    def generateUserSequential(self, walkNums, walkLen):
        '''
        生成新的用户行为序列，对于数据量巨大的，可以使用joblibd的Parallel, delayed来进行并行处理, itertools.chain合并结果
        :return: 新的用户序列
        '''
        G = self.generateGraph()
        walks = []
        nodes = list(G.nodes)
        random.shuffle(nodes)
        for _ in range(walkNums):
            for node in nodes:
                walks.append(self.randomWalk(G, node, walkLen))
        return walks

    # 并行生成新行为序列
    def parallelGenerateSequential(self):
        walks = Parallel(n_jobs=2)(delayed(self.generateUserSequential)(num, self.walkLen) for num in self.partition(self.walkNums, 2))
        walks = list(itertools.chain(*walks))
        print("newUserSequential build done.")
        return walks

    def partition(self, walkNums, workers):
        if walkNums % workers == 0:
            return [walkNums//workers] * workers
        else:
            return [walkNums//workers] * workers + [walkNums%workers]

    def randomWalk(self, G:nx.Graph, startNode:str, walkLen:int):
        '''
        :param G: 生成的图
        :param startNode: 随机的起始节点
        :param walkLen: 每个节点开始生成的序列长度
        :return: 起始节点生成的序列
        '''
        walk = [startNode]
        while len(walk) < walkLen:
            currNode = walk[-1]
            neighbors = list(G.neighbors(currNode))
            if len(neighbors) > 0:
                walk.append(random.choice(neighbors))  # 无向无权图随机选择一个节点
            else:
                break
        return walk


    def generateVec(self, filename):
        '''
        使用Word2vec生成 dense embedding
        :param userSequential: deepWalk生成的用户序列
        :return:
        '''
        userSequential = self.parallelGenerateSequential()
        model = Word2Vec(sentences=userSequential, hs=0, sg=1, vector_size=self.vecDim, min_count=2, window=2)
        # model.wv.save_word2vec_format("./{}.vector".format(filename))
        model.save("./{}.model".format(filename))
        print("DeepWalk vector build done.")


if __name__ == '__main__':
    start = time.time()
    dataset = [[random.choice(range(0, 21)) for _ in range(random.choice(range(1, 10)))]for _ in range(5)]
    deepWalk = DeepWalk(dataset, 10, 5)
    deepWalk.generateVec("deepwalk")

