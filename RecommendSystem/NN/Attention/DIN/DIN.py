"""
    阿里从工业的角度引入注意力机制的一次实践，对所有类别特征都进行Embedding处理（降低输入的维度），对用户历史行为的物品与候选物品之间使用注意力机制
"""
import torch
import torch.nn as nn
import torch.optim as Optim
import torch.utils.data as Data


# 自定义数据集，将其写入loader中
class LoadDataset(Data.Dataset):
    def __init__(self):
        super(LoadDataset, self).__init__()
        self.dataset = [[torch.rand(3), torch.rand(3), torch.rand(3), torch.rand(3), torch.rand(3)] for _ in range(2)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):  # item是索引
        return self.dataset[item], torch.randint(2, (1, 1)).reshape(1).type(torch.FloatTensor)


class DIN(nn.Module):
    def __init__(self, m, feaDim, aDim, outLayers, aLayers):
        super(DIN, self).__init__()
        self.m = m  # 用户最近的历史行为个数
        self.feaDim = feaDim  # 特征总维度
        self.aDim = aDim  # 注意力网络输入的总维度
        self.outLayers = outLayers  # 输出层的层数
        self.aLayers = aLayers  # 注意力网络的层数
        # 上层的MLP
        self.outMLP = nn.ModuleDict([["out{}".format(i), nn.Sequential(nn.Linear(self.feaDim//i, self.feaDim//(i+1)), nn.ReLU())] for i in range(1, self.outLayers+1)])
        # 注意力网路
        self.aMLP = nn.Sequential(
            nn.Linear(self.aDim, self.aDim//2),
            nn.ReLU(),
            nn.Linear(self.aDim//2, 1),
        )
        # 最后的输出层
        self.finalOut = nn.Sequential(
            nn.Linear(self.feaDim//(self.outLayers+1), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        :param input: [[Tensor],[Tensor],[Tensor],......],前m为用户的历史行为，第m+1个为待推荐物品，剩下的为基本特征信息
        :return:
        """
        # 假设选取用户最近的m个用户历史行为
        targetItem = input[self.m]
        # print(targetItem.size())
        tmp = torch.zeros((targetItem.size(0), targetItem.size(1)))  # 定义一个定时变量，用于计算加权和
        for i in range(self.m):  # 前m个为用户的历史行为数据
            # 求历史行为与目标物品的互操作，这里使用内积
            a = torch.cat([input[i], torch.multiply(input[i], targetItem), targetItem], dim=1)
            aij = self.aMLP.forward(a)  # 注意力得分
            tmp = tmp+torch.multiply(aij, input[i])
        # 组合经过注意力机制的特征、基本特征、目标特征
        otherFea = torch.cat([input[self.m+i+1] for i in range(len(input[self.m+1:]))], dim=1)
        inputFea = torch.cat([tmp, targetItem, otherFea], dim=1)
        for i in range(1, self.outLayers+1):
            inputFea = self.outMLP['out{}'.format(i)].forward(inputFea)
        out = self.finalOut.forward(inputFea)
        return out


if __name__ == '__main__':
    din = DIN(2, 12, 9, 3, 2)
    dataset = LoadDataset()
    loader = Data.DataLoader(dataset, batch_size=50)
    lossfun = nn.BCELoss()
    optim = Optim.Adam(din.parameters(), lr=0.01)
    print(din.get_parameter("finalOut.0.weight"))
    for epoch in range(100):
        for step, (x, y) in enumerate(loader):
            out = din.forward(x)
            loss = lossfun.forward(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    print(din.get_parameter("finalOut.0.weight"))
