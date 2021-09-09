'''
    常用于 协同过滤 + 快速近邻搜索
    使用矩阵行置换的方式（填写第一次置换出现1的下标），针对的可以是共现矩阵（针对用户的，针对物品的），占据97%的时间
    搜索占据3%的时间
'''
import random
import time

import numpy as np


# 获取签名矩阵
def minHash(dataset, times):
    '''
    :param dataset:mat.shape(m:维数,n:用户数)
    :param times: 置换的次数
    :return: 置换t次后生成的列表，list.shape(t,n)
    '''
    n = len(dataset[0])
    m = len(dataset)
    resLst = np.array([[-1]*n for _ in range(times)])
    for t in range(times):
        count = 0
        idxSet = [i for i in range(m)]
        while count < n and len(idxSet) > 0:
            randonNum = random.choice(idxSet)  # 随机一个数
            row = dataset[randonNum]
            for i, ele in enumerate(row):
                if resLst[t][i] == -1 and ele == 1:
                    resLst[t][i] = randonNum
                    count += 1
            idxSet.remove(randonNum)  # 置换过的下标不再参与下一次选取
    return resLst


def searchNN(dataset, res, band, tarUserIdx):
    # 使用 Or 的方式求每一桶的相似用户，然后汇总去重
    simUser = []
    rows = len(res) // band
    for i in range(band):
        for userIdx in range(len(res[0])):
            if userIdx != tarUserIdx:
                if res[i*rows: min((i+1)*rows, len(res)), tarUserIdx].tolist() == res[i*rows: min((i+1)*rows, len(res)), userIdx].tolist():
                    simUser.append(userIdx)
    simUser = list(set(simUser))
    # 计算候选集的相似度
    sortSimUser = {}
    for user in simUser:
        sortSimUser[user] = np.sqrt(np.sum((dataset[:, tarUserIdx]-dataset[:, user])^2, axis=0))
    sortSimUser = sorted(sortSimUser.items(), key=lambda x: x[1], reverse=False)
    print("相似的用户：{}".format([ele[0] for ele in sortSimUser]))
    print("相似的用户个数：{}".format(len(sortSimUser)))


if __name__ == '__main__':
    start = time.process_time()
    dataset = [[random.choice([0, 1]) for _ in range(200)] for _ in range(100000)]
    dataset = np.array(dataset).T
    res = minHash(dataset, 100)
    second = time.process_time()
    print("构建minHash矩阵时间%.3f" % (second-start))
    searchNN(dataset, res, 20, 0)
    print("搜索minHash近邻时间%.3f" % (time.process_time()-second))
    print("程序总运行时间：%.3f" % (time.process_time()-start))