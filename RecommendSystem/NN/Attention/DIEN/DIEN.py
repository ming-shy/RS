"""
    DIN模型的增强版，DIN只考虑了用户历史行为，并没有考虑其历史行为的兴趣变化，因此DIEN考虑了用户行为的序列信息；
    常见的序列模型：RNN  -> LSTM  ->  GRU（效果类似LSTM,但比LSTM的参数少）
"""
import torch
import torch.nn as nn
import torch.optim as Optim
import torch.utils.data as Data
from GRU import GRU


# 自定义数据集，将其写入loader中
class LoadDataset(Data.Dataset):
    def __init__(self):
        super(LoadDataset, self).__init__()
        self.dataset = [[torch.rand(3), torch.rand(3), torch.rand(3), torch.rand(11)] for _ in range(100)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):  # item是索引
        return self.dataset[item], torch.randint(2, (1, 1)).reshape(1).type(torch.FloatTensor)


class DIEN(nn.Module):
    def __init__(self, batchSize, m, mFeas, layers, totalFeaDim):
        super(DIEN, self).__init__()
        self.batchSize = batchSize
        self.m = m  # 历史行为个数
        self.mFeas = mFeas  # 每一个行为的embedding维数，即特征数
        self.layers = layers  # 上层MLP的层数
        self.totalFeaDim = totalFeaDim  # 总特征的维度
        self.GRU = GRU(self.batchSize, self.m, self.mFeas)
        self.AUGRU = GRU(self.batchSize, self.m, self.mFeas, isAttention=True)
        self.MLP = nn.ModuleDict([
            ["mlp{}".format(i), nn.Sequential(nn.Linear(self.totalFeaDim//i, self.totalFeaDim//(i+1)), nn.ReLU())]
            for i in range(1, self.layers+1)])
        self.finalOut = nn.Sequential(
            nn.Linear(self.totalFeaDim//(self.layers+1), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        r"""
        :param input: [[Tensor], [Tensor], [Tensor], .... ],前m个为用户的历史行为，第m个为目标item，最后为其他特征
        :return:
        """
        targetItem = input[self.m]
        GRUInput = torch.unsqueeze(input[0], dim=1)
        for i in range(1, self.m):
            GRUInput = torch.cat((GRUInput, torch.unsqueeze(input[i], dim=1)), dim=1)
        # GRU的输出是三维的，自定义的GRU
        GRUOut = self.GRU.forward(GRUInput, False)
        AUGRUOut = self.AUGRU.forward(GRUOut, targetItem)
        # 取AUGRU的最后一个输出用于参与后面的MLP的训练
        AUGRUOut = AUGRUOut[:, -1]
        # 拼接 AUGRU的输出 targetItem 其他特征
        otherFea = input[-1]
        totalFea = torch.cat((AUGRUOut, targetItem, otherFea), dim=1)
        # 上层MLP的训练
        for i in range(1, self.layers+1):
            totalFea = self.MLP["mlp{}".format(i)].forward(totalFea)
        # 最终的输出
        out = self.finalOut.forward(totalFea)
        return out


if __name__ == '__main__':
    ds = LoadDataset()
    loader = Data.DataLoader(ds, batch_size=50)
    dien = DIEN(batchSize=50, m=2, mFeas=3, layers=3, totalFeaDim=17)
    lossfun = nn.BCELoss()
    optim = Optim.Adam(dien.parameters(), lr=0.01)
    print(dien.get_parameter("finalOut.0.weight"))
    for step, (x, y) in enumerate(loader):
        out = dien.forward(x)
        loss = lossfun.forward(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(dien.get_parameter("finalOut.0.weight"))