'''
    算法思想与Word2Vec是一致的，只不过在构造训练集时，摒弃了词窗口的概念，对于一个用户的历史行为，两两形成训练集
    优化softmax:使用负采样的方法，其损失函数使用多分类交叉熵，负采样方法如下：
'''
import time
import torch
import torch.nn.functional as F


def test():
    start = time.process_time()
    for _ in range(10000):
        train = torch.rand(20000)
        # ---选取负采样的值--- #
        train = train[[1, 36, 360, 3600, 8600, 16000]]
        res = F.softmax(train, dim=0)
        new_train = torch.zeros(20000)
        new_train[[1, 36, 360, 3600, 8600, 16000]] = res
        new_train = new_train.view([1, -1])
        # --- 负采样结束  --- #
        target = torch.LongTensor([1])
        loss = F.cross_entropy(new_train, target)
    print("运行时间：", time.process_time()-start)
    print("负采样损失函数值：", loss)

    start = time.process_time()
    for _ in range(10000):
        train = torch.rand(20000)
        res = F.softmax(train, dim=0)
        res = res.view([1, -1])
        target = torch.LongTensor([1])
        loss = F.cross_entropy(res, target)
    print("运行时间：", time.process_time()-start)
    print("不进行负采样的损失函数值：", loss)


if __name__ == '__main__':
    test()