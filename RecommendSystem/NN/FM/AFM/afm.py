"""
    对NFM的改进，加入了注意力机制（对特征交叉池化层进行求加权和）
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as Optim


class AFM(nn.Module):
    def __init__(self, feas, k, deepDepth):
        super(AFM, self).__init__()
        self.feas = feas
        self.k = k
        self.deepDepth = deepDepth
        self.wide = nn.Linear(self.feas, 1)  # wide部分
        self.cross = nn.Parameter(torch.rand(self.feas, self.k))  # 为特征设置隐向量
        # 注意力网络
        self.attention = nn.Sequential(
            nn.Linear(self.k, self.k),
            nn.ReLU(),
            nn.Linear(self.k, 1, bias=False),
        )
        self.deep = nn.ModuleDict([["deep{}".format(i), nn.Sequential(nn.Linear(self.k//i, self.k//(i+1)), nn.ReLU())] for i in range(1, self.deepDepth+1)])
        self.out = nn.Linear(self.feas//(self.deepDepth+1), 1)

    def forward(self, input):
        linear = self.wide.forward(input)
        # 计算交叉向，进行特征交叉池化层
        tmp = torch.zeros((input.size(0), self.k))
        for step in range(input.size(0)):   # batchSize
            a = torch.zeros(1)
            b = torch.zeros(self.k)
            for i in range(input.size(1)):
                for j in range(i, input.size(1)):
                    if input[step, i] == 0:
                        break
                    if input[step, j] == 0:
                        continue
                    # 对特征交叉项保留
                    a = torch.cat((a, self.attention(input[step, i]*input[step, j]*torch.multiply(self.cross[i], self.cross[j]))), dim=0)
                    b = torch.cat((b, input[step, i]*input[step, j]*torch.multiply(self.cross[i], self.cross[j])), dim=0)
            a = nn.Softmax(dim=0).forward(a)  # softmax获取注意力得分
            tmp[step] = torch.sum(torch.multiply(a.reshape(-1, 1), b.reshape(-1, self.k)), dim=0)  # 求加权和
        for i in range(1, self.deepDepth+1):
            tmp = self.deep['deep{}'.format(i)].forward(tmp)
        deepOut = self.out.forward(tmp)
        out = nn.Sigmoid().forward(linear+deepOut)  # 整合线性部分与深度部分
        return out



if __name__ == '__main__':
    x = torch.rand(2, 5)
    y = torch.rand(2, 1)
    dataset = Data.TensorDataset(x, y)
    nfm = AFM(5, 3, 2)
    loader = Data.DataLoader(dataset, batch_size=50)
    lossfun = nn.BCELoss()
    optim = Optim.Adam(nfm.parameters(), lr=0.01)
    print(nfm.get_parameter("cross"))
    for _ in range(10):
        for i, (x, y) in enumerate(loader):
            out = nfm.forward(x)
            loss = lossfun.forward(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    print(nfm.get_parameter("cross"))