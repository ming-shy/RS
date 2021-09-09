'''
wide&deep的改进版deep&cross，简称DCN，deep部分未做改动，对wide部分进行了改进，使用了一个cross网络，
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimzer
import torch.utils.data as Data


class DeepAndCross(nn.Module):
    def __init__(self, crossDepth, deepDepth, crossInputDim, deepInputDim):
        super(DeepAndCross, self).__init__()
        self.crossDepth = crossDepth
        self.deepDepth = deepDepth
        self.crossInputDim = crossInputDim
        self.deepInputDim = deepInputDim
        # 给cross层创建参数
        self.crossWeight = nn.Parameter(torch.rand((self.crossDepth, 1, self.crossInputDim), requires_grad=True))
        self.crossBias = nn.Parameter(torch.rand((self.crossDepth, 1, self.crossInputDim), requires_grad=True))
        # 使用ModuleDict([[],[]])创建深度为n的模型，可以指定每一层的名字，等价于 使用默认名字的ModuleList([,,,])
        self.deep = nn.ModuleDict([
            ["deep"+str(i), nn.Sequential(
                nn.Linear(self.deepInputDim//i, self.deepInputDim//(i+1)),
                nn.ReLU()
            )] for i in range(1, self.deepDepth+1)
        ])
        self.out = nn.Sequential(
            nn.Linear(self.crossInputDim+(self.deepInputDim//(self.deepDepth+1)), 1),
            nn.Sigmoid(),
        )

    def forward(self, input):  # input由cross和deep两部分组成
        """前向传播
        :param input: [2-array,2-array]
        :return: 前向传播的最终输出值
        """
        # 求cross部分的输出
        lLayerInput = input[0].t()
        for i in range(self.crossDepth):
            lLayerInput = torch.add(torch.add(torch.mm(torch.mm(input[0].t(), lLayerInput.t()), self.crossWeight[i].t()), self.crossBias[i].t()), lLayerInput)
        crossOut = lLayerInput.t()
        # 求deep部分输出, ModuleDict()形式的求值
        deepOut = input[1]
        for i in range(1, self.deepDepth+1):
            deepOut = self.deep["deep"+str(i)].forward(deepOut)
        # 连接cross和deep的输出
        cat = torch.cat((crossOut, deepOut), dim=1)
        # 最终的输出
        out = self.out.forward(cat)
        return out

    def train(self, input):
        """测试参数是否进行了更新
        """
        print(self.get_parameter('crossWeight'))
        lossfun = nn.BCELoss()
        optim = optimzer.Adam(self.parameters(), lr=0.001)
        y = torch.from_numpy(np.array([[1]])).type(torch.FloatTensor)
        print(y)
        out = self.forward(input)
        loss = lossfun.forward(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(self.get_parameter("crossWeight"))


if __name__ == '__main__':
    deepAndCross = DeepAndCross(5, 5, 10, 100)
    input = [torch.rand(1, 10), torch.rand(1, 100)]
    deepAndCross.train(input)