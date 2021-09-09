'''
在FM的基础上对特征交叉时，给每个特征学习的隐向量进行了增强
'''
import torch
import torch.nn as nn
import torch.utils.data as Data


class FFM(nn.Module):
    def __init__(self, epoch, feaDim, latentDim):
        super(FFM, self).__init__()
        self.epoch = epoch
        self.feaDim = feaDim
        self.latentDim = latentDim
        # 定义该模型所需要的一些参数
        self.linear = nn.Linear(feaDim, 1, bias=True)
        self.cross = nn.Parameter(torch.rand(self.feaDim, self.feaDim, self.latentDim))
        self.activation = nn.Sigmoid()

    def buildFFM(self, input):
        linear = self.linear.forward(input)  # 一阶部分就是线性回归
        crossPart1 = torch.mm(input.transpose(0, 1), input)  # 二阶部分，特征的交叉值
        crossPart2 = torch.zeros(self.feaDim, self.feaDim)
        for i in range(self.feaDim):  # 二阶部分的权重求取
            for j in range(self.feaDim):
                crossPart2[i, j] = torch.dot(self.cross[i, j, :], self.cross[j, i, :])
        cross = torch.sum(torch.multiply(crossPart1, crossPart2))  # 组合二阶部分
        output = torch.add(linear, cross)  # 组合一阶与二阶部分
        output = self.activation.forward(output)  # 激活函数，将值映射值（0，1）
        return output

    def forward(self, input):
        output = self.buildFFM(input)
        return output

    def trainFFM(self, loader):
        lossfun = nn.BCELoss()
        optimzer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(self.epoch):
            for step, (batchX, batchY) in enumerate(loader):
                out = self.forward(batchX)
                loss = lossfun.forward(out, batchY)
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()


if __name__ == '__main__':
    ffm = FFM(1, 5, 2)
    ffm.buildFFM(torch.rand(1, 5))


