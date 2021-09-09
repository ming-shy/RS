'''
    矩阵分解，采用标准的梯度下降算法，损失函数加入正则化项（防止过拟合）
'''
import numpy as np
import matplotlib.pyplot as plt


class MF:
    def __init__(self, G, maxIters, K, threshold, alpha, beta):
        super(MF, self).__init__()
        self.G = G  # 共现矩阵users-items
        self.maxIters = maxIters  # 最大迭代次数
        self.K = K  # 隐向量的个数
        self.threshold = threshold  # 结束训练的阈值
        self.alpha = alpha  # 学习率
        self.beta = beta  # 正则化参数

    def mf(self):
        res = []  # 记录损失函数的值
        # 随机生成P Q矩阵
        users = len(self.G)
        items = len(self.G[0])
        P = np.random.uniform(-1, 1, (users, self.K))
        Q = np.random.uniform(-1, 1, (items, self.K))
        # 迭代次数
        for iter in range(self.maxIters):
            for userIdx in range(users):
                for itemIdx in range(items):
                    if self.G[userIdx, itemIdx] > 0:
                        # 更新P Q矩阵的值
                        eij = self.G[userIdx, itemIdx]-np.matmul(P[userIdx, :], Q[itemIdx, :].T)  # 真实值与预测值的差值
                        P[userIdx, :] -= 2*self.alpha*(-1*eij * Q[itemIdx, :]+self.beta*P[userIdx, :])
                        Q[itemIdx, :] -= 2*self.alpha*(-1*eij * P[userIdx, :]+self.beta*Q[itemIdx, :])
            # 每一次迭代之后当前的损失函数
            loss = 0
            for userIdx in range(users):
                for itemIdx in range(items):
                    if self.G[userIdx, itemIdx] > 0:  # 损失函数是针对训练集
                        loss += pow(self.G[userIdx, itemIdx]-np.matmul(P[userIdx, :], Q[itemIdx, :].T), 2)
                        for k in range(self.K):  # L2正则化项
                            loss += self.beta*(pow(P[userIdx, k], 2) + pow(Q[itemIdx, k], 2))
            print("迭代次数%d/%d,损失函数值为：%.6f" % (iter+1, self.maxIters, loss))
            if len(res) > 1 and abs(loss - res[-1]) < self.threshold:
                break
            res.append(loss)
        print("训练后的评分矩阵", np.matmul(P, Q.T))
        print("用户矩阵", P)
        print("物品矩阵", Q)
        return P, Q, res

    def displayLoss(self, res):
        plt.plot(range(len(res)), res, color='r', linewidth=3)
        plt.title("Convergence curve")
        plt.xlabel("generation")
        plt.ylabel("loss")
        plt.show()


if __name__ == '__main__':
    G = [
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ]
    G = np.array(G)
    print("原评分矩阵:%s" % G)
    mf = MF(G, 5000, 10, 0.0001, 0.01, 0.1)
    P, Q, res = mf.mf()
    mf.displayLoss(res)