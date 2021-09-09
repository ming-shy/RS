import torch.utils.data as Data
import torch
import numpy as np


class loadDataset(Data.Dataset):
    def __init__(self, path):  # 设置数据集的一些初始信息  结构化数据：可以一次性加载到内存，  对于非结构化数据：可以加载其路径，进行存储，然后使用的时候再读取
        super(loadDataset, self).__init__()
        self.dataset = np.loadtxt(path, encoding='utf-8',delimiter=',')

    def __len__(self):  # 返回数据集的长度
        return len(self.dataset)

    def __getitem__(self, item):  # 根据索引item，读取数据
        self.X = torch.from_numpy(self.dataset[item, :-1]).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array([self.dataset[item, -1]])).type(torch.FloatTensor)
        return self.X, self.y