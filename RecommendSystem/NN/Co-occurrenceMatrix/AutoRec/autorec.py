'''
利用自编码器的思想，对共现矩阵进行拟合，
加载数据集，可以使用torch.utils.data,重写Dataset的类，实现三个方法 __init__  __len__  __getitem__
'''
import torch
import torch.nn as nn
import torch.utils.data as Data


class LoadDataset(Data.Dataset):
    def __init__(self):
        # 加载数据集
        pass

    def __len__(self):
        # 返回数据集的长度
        pass

    def __getitem__(self, item):
        # 返回一条数据需要返回的信息
        pass


class AutoRec(nn.Module):
    def __init__(self, epoch, batchSize, feaDim, hiddenDim):
        super(AutoRec, self).__init__()
        self.epoch = epoch
        self.batchSize = batchSize
        self.feaDim = feaDim
        self.hiddenDim = hiddenDim
        self.hidden = nn.Sequential(
            # nn.Dropout(p=0.2),  # 添加Dropout，降低过拟合的风险（原理就是随机删除一些神经元）
            nn.Linear(self.feaDim, self.hiddenDim),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(self.hiddenDim, self.feaDim),
            nn.ReLU(),
        )

    def forward(self, input):
        out = self.hidden.forward(input)
        out = self.out.forward(out)
        return out

    def train(self, dataset):
        lossfun = nn.MSELoss()
        optimzer = torch.optim.Adam(self.parameters(), lr=0.001)
        loader = Data.DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
        for epoch in range(self.epoch):
            for step, (batchX, batchY) in enumerate(loader):
                out = self.forward(batchX)
                loss = lossfun.forward(out, batchX)
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()


if __name__ == '__main__':
    autorec = AutoRec(1, 50, 10, 5)
    print(autorec)

