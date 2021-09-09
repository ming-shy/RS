'''
GMF + MLP 的组合体：相当于使用PQ矩阵进行训练
可以对不同的参数设置不同的优化器（包括优化器的不同种类，同一优化器的不同参数）
    userParas = id(neuralCF.get_parameter('userP'))  # id() 为传入的对象分配一个唯一id
    itemParas = id(neuralCF.get_parameter('itemQ'))
    gmfParas = filter(lambda p: id(p) in [userParas, itemParas], neuralCF.parameters())  # 使用唯一id过滤参数
    mlpParas = filter(lambda p: id(p) not in [userParas, itemParas], neuralCF.parameters())
'''
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim


class NeuralCF(nn.Module):
    def __init__(self, users, items, latentDim):
        super(NeuralCF, self).__init__()
        self.users = users
        self.items = items
        self.latentDim = latentDim
        self.userP = nn.Parameter(torch.rand(self.users, self.latentDim))  # 生成P矩阵
        self.itemQ = nn.Parameter(torch.rand(self.items, self.latentDim))  # 生成Q矩阵
        self.mlp = nn.Sequential(  # 定义MLP层
            nn.Linear(2*self.latentDim, self.latentDim),
            nn.ReLU(),
            nn.Linear(self.latentDim, self.latentDim//2),
            nn.ReLU(),
            nn.Linear(self.latentDim//2, self.latentDim//4),
            nn.ReLU(),
        )
        self.outLayer = nn.Linear(self.latentDim+self.latentDim//4, 1)   # 定义输出层

    # 组合GMF + MLP
    def gmfMlp(self, input:list):  # 输入包含P、Q下标的列表
        userVec = self.userP[input[0], :]
        itemVec = self.itemQ[input[1], :]
        gmf = torch.multiply(userVec, itemVec)  # 元素乘积  三种矩阵乘法：dot(内积), mm(二阶矩阵乘法),multiply(元素积)
        concat = torch.cat((userVec, itemVec))
        mlpOut = self.mlp.forward(concat)
        concat = torch.cat((gmf, mlpOut))
        out = self.outLayer.forward(concat)
        return out

    # 前向传播
    def forward(self, input):
        return self.gmfMlp(input)


if __name__ == '__main__':
    neuralCF = NeuralCF(10, 8, 4)
    print(neuralCF.forward([0, 1]))  # 制作数据集X为PQ对应的下标，Y为具体的评分