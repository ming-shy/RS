'''
    用于召回阶段的 快速近邻搜索
    常用于对embedding进行快速搜索，
    需要调参：映射的次数times、每一桶的宽度bukWidth、求候选集时的划分band
'''
import numpy as np


# 获取签名矩阵
def buildBucketMat(dataset, times, bukWidth):
    '''
    :param dataset:shape(n:物品数,m:维度)
    :param times: 重复的次数
    :param bukWidth:每一桶的宽度   b: 0~bukWidth间一个均匀分布的变量
    :return: mat.shape(times,n)
    '''
    n = dataset.shape[0]
    m = dataset.shape[1]
    tmpb = []
    resMat = np.zeros((times, n))
    for row in range(times):
        # 随机生成一个m维向量
        v = np.random.uniform(1, 10, m)
        # 随机生成一个b
        b = np.random.uniform(0, bukWidth)
        while b in tmpb:
            b = np.random.uniform(0, bukWidth)
        tmpb.append(b)
        for col in range(n):
            resMat[row, col] = (np.sum(dataset[col, :] * v, axis=0) + b) // bukWidth   # 哈希函数进行映射
    return resMat


# 得到resMat之后，使用与minHash类似的处理的方法：划分band，再求出相邻点的候选集合，再使用距离度量函数进行 NN 的寻找


if __name__ == '__main__':
    dataset = np.array([np.random.randint(0, 10, 5) for _ in range(10)])
    resMat = buildBucketMat(dataset, 7, 4)
    print(resMat)