'''
wide部分使模型具有记忆能力，可以选择较简单的模型和高相关的特征
'''
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optimzer


class WideAndDeep(nn.Module):
    def __init__(self, epoch, wideInputDim, deepInputDim):
        super(WideAndDeep, self).__init__()
        self.epoch = epoch
        self.wideInputDim = wideInputDim
        self.deepInputDim = deepInputDim
        self.wide = nn.Linear(self.wideInputDim, 1)
        self.deep = nn.Sequential(
            nn.Linear(self.deepInputDim, self.deepInputDim//2),
            nn.ReLU(),
            nn.Linear(self.deepInputDim//2, self.deepInputDim//4),
            nn.ReLU(),
            nn.Linear(self.deepInputDim//4, 1),
        )

    def forward(self, wideInput, deepInput):
        wideOut = self.wide.forward(wideInput)
        deepOut = self.deep.forward(deepInput)
        out = nn.Sigmoid.forward(torch.add(wideOut, deepOut))
        return out

    # 划分参数,wide部分的优化器使用FTRL，deep部分的优化器使用Adam
    def splitParas(self):
        tmpParas1 = id(self.get_parameter("wide.weight"))
        tmpParas2 = id(self.get_parameter("wide.bias"))
        wideParas = filter(lambda p: id(p) in [tmpParas1, tmpParas2], self.parameters())
        deepParas = filter(lambda p: id(p) not in [tmpParas1, tmpParas2], self.parameters())
        return wideParas, deepParas

    # wide部分使用FTRL优化器
    def FTRL(self, params):
        pass

    def train(self, loader):
        lossfun = nn.BCELoss()
        wideParas, deepParas = self.splitParas()
        deepOptim = optimzer.Adam([{'params': deepParas}])  # 为deep部分的参数设置Adam优化器
        # FTRL优化器
        for epoch in range(self.epoch):
            for step, (batchX, batchY) in enumerate(loader):
                out = self.forward(batchX[0], batchX[1])
                loss = lossfun.forward(out)
                # 损失函数加上正则化
                deepOptim.zero_grad()
                loss.backward()

                deepOptim.step()


if __name__ == '__main__':
    wideAndDeep = WideAndDeep(5, 5)
    wideAndDeep.train([])