"""FNN model build.
    First, pretrain FM model to get its parameters;
    second,these parameters as embedding layer will become MLP’s input
"""
import torch 
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as Optim


class FM(nn.Module):
    """当前FM模型主要针对sparse特征，为后续MLP输入做准备
    """
    def __init__(self, feas, k):
        super(FM, self).__init__()
        self.feas = feas  # one-hot的特征数
        self.k = k  # 隐向量的维度
        self.cross = nn.Parameter(torch.rand(self.feas, 1, self.k))  # 为每一个特征生成一个隐向量
        self.linear = nn.Linear(self.feas, 1)

    def getParas(self):
        linearWeight = self.get_parameter("linear.weight")
        linearBias = self.get_parameter("linear.bias")
        cross = self.get_parameter("cross")
        return linearBias, linearWeight, cross

    def forward(self, input):
        linear = self.linear.forward(input)
        # 计算二阶部分
        cross = torch.zeros(self.feas, self.feas)
        for i in range(self.feas):
            for j in range(self.feas):
                cross[i, j] = torch.mm(self.cross[i], self.cross[j].t())
        cross = torch.triu(cross)  # 仅保留上三角矩阵，由于特征交叉后的计算值是一个对称矩阵
        cross = torch.sum(cross)
        # 整合一阶和二阶部分
        out = torch.add(linear, cross)
        # 激活函数
        out = nn.Sigmoid().forward(out)
        return out

    def train(self, input):
        print(self.get_parameter("cross"))
        y = torch.ones(1, 1)
        lossfun = nn.BCELoss()
        optim = Optim.Adam(self.parameters())
        for i in range(100):
            out = self.forward(input)
            loss = lossfun.forward(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(self.get_parameter("cross"))


class FNN(nn.Module):
    def __init__(self, feas):
        super(FNN, self).__init__()
        self.feas = feas
        self.fnn = nn.Sequential(
            nn.Linear(self.feas, self.feas),
            nn.Tanh(),
            nn.Linear(self.feas, self.feas//2),
            nn.Tanh(),
            nn.Linear(self.feas//2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """前向传播
            得到FM对sparse特征的参数；
            整合dense和加工后的sparse，共同作为MLP的输入
        :param input: 预训练得到的参数
        :return:
        """
        a, b, c = FM(10, 5).getParas()


if __name__ == '__main__':
    fm = FM(5, 2)
    input = torch.randn(1, 5)
    fm.getParas()