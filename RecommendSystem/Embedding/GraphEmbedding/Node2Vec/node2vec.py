import itertools
import random
from gensim.models import Word2Vec
import networkx as nx
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


class Node2Vec:
    def __init__(self, dataset, p, q, walkNums, walkLen):
        super(Node2Vec, self).__init__()
        self.dataset = dataset
        self.p = p
        self.q = q
        self.walkNums = walkNums
        self.walkLen = walkLen

    def generateGraph(self):
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
        # nx.draw(G, with_labels=True)
        # plt.show()
        print("Graph build done.")
        return G


    def generateUserSequential(self, walkNums, walklen):
        G = self.generateGraph()
        walks = []
        for _ in range(walkNums):
            nodes = list(G.nodes)
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                while len(walk) < walklen:
                    neighbors = list(G.neighbors(walk[-1]))  # 找当前路径的最后一个节点的邻居
                    if len(neighbors) > 0:
                        if len(walk) == 1:
                            walk.append(random.choice(neighbors))
                        else:
                            # 计算邻居节点的跳转概率
                            random.shuffle(neighbors)
                            selectNode = {}
                            for nb in neighbors:
                                if nb == walk[-2]:  # dt=0
                                    selectNode[nb] = (1/self.p) * 1
                                elif G.has_edge(walk[-2], nb):  # dt=1
                                    selectNode[nb] = 1
                                else:  # dt=2
                                    selectNode[nb] = (1/self.q) * 1
                            selectNode = sorted(selectNode.items(), key=lambda x: x[1], reverse=True)
                            walk.append(selectNode[0][0])
                    else:
                        break
                walks.append(walk)
        return walks

    def partition(self, walkNums, workers):
        if walkNums % workers == 0:
            return [walkNums//workers] * workers
        else:
            return [walkNums//workers] * workers + [walkNums%workers]

    def parallelGenerateSequential(self):
        walks = Parallel(n_jobs=2)(delayed(self.generateUserSequential)(num, self.walkLen) for num in self.partition(self.walkNums, 2))
        walks = list(itertools.chain(*walks))
        print(walks)
        print("new Sequential build done.")
        return walks

    def generateVec(self, filename):
        walks = self.parallelGenerateSequential()
        model = Word2Vec(sentences=walks, sg=1, window=3, min_count=2)
        # model.wv.save_word2vec_format("./node2vec.vector")
        model.save("./{}.model".format(filename))
        print("模型保存完毕。。。。。。")


if __name__ == '__main__':
    dataset = [[random.choice(range(0, 11)) for _ in range(random.choice(range(1, 20)))]for _ in range(5)]
    node2vec = Node2Vec(dataset=dataset, p=0.8, q=0.2, walkNums=10, walkLen=10)
    node2vec.generateVec("node2vec")


