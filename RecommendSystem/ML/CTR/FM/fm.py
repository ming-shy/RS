'''
对物品的评分思想，改变为对物品的点击率，损失函数可为交叉熵
可以使用pytorch实现，由于torch内置了nn.Parameters()作为 nn.Module 中的可训练参数使用，而Tensor虽然可以设置为自动求导，但是不参与训练使用的
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


class FM(nn.Module):
    def __init__(self, epoch, featureDim, latentDim):
        super(FM, self).__init__()
        self.epoch = epoch  # 迭代的epoch次数
        self.featureDim = featureDim  # 特征维度
        self.latentDim = latentDim  # 隐向量维度
        self.linear = nn.Linear(featureDim, 1, bias=True)  # FM的线性分
        self.cross = nn.Parameter(torch.rand((self.featureDim, self.latentDim)))  # FM特征交叉部分,自定义参数参数模型的训练
        self.activation = nn.Sigmoid()

    def combineFM(self, input):
        linearPart = self.linear.forward(input)
        crossPart1 = torch.mm(input.transpose(0, 1), input)
        crossPart2 = torch.mm(self.cross, self.cross.transpose(0, 1))
        cross = torch.sum(torch.triu(torch.mul(crossPart1, crossPart2), diagonal=1))  # 进保留上三角矩阵（可包含主对角元素）就和
        output = torch.add(linearPart, cross)
        output = self.activation.forward(output)  # 使用激活函数输出其概率
        return output

    def forward(self, input):
        output = self.combineFM(input)
        return output

    # 添加正则化项，避免过拟合
    def addRegular(self):
        l2Loss = torch.tensor(0.0, requires_grad=True)
        for name, paras in self.named_parameters():
            if 'bias' not in name:
                l2Loss = l2Loss + torch.sum(torch.pow(paras, 2))
        return l2Loss

    def trainFM(self, loader, lossfun, optimzer):
        for epoch in range(self.epoch):
            for step, (batchX, batchY) in enumerate(loader):
                output = self.forward(batchX)
                loss = lossfun.forward(output, batchY)
                # print(loss.data.numpy())
                # 还可以添加正则化项
                loss = loss + self.addRegular()  # 尽量不使用 += 的形式，影响其反向传播求导的过程
                print(loss)
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
                # for name, paras in self.named_parameters():
                    # if 'bias' in name:
                    #     print(paras)
        # 训练完毕，保存模型
        torch.save(self, "../../model/FM.h5")
        print("model save done.")



if __name__ == '__main__':
    fm = FM(5, 5, 2)
    X = np.random.uniform(-1, 1, size=(1000, 5))
    y = np.random.randint(0, 2, (1000, 1))
    inputX = torch.from_numpy(X).type(torch.FloatTensor)
    inputY = torch.from_numpy(y).type(torch.FloatTensor)
    dataset = Data.TensorDataset(inputX, inputY)
    loader = Data.DataLoader(dataset, batch_size=50, shuffle=True)
    lossfun = nn.BCELoss()
    optimzer = torch.optim.Adam(params=fm.parameters(), lr=0.001)
    fm.trainFM(loader, lossfun, optimzer)
