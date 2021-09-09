"""
    对FM二阶部分使用MLP进行改进，如果把FM一阶部分当成wide，则该方法也可以看做是对wide&deep的deep部分的改进
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as Optim


class NFM(nn.Module):
    def __init__(self, feas, k, deepDepth):
        super(NFM, self).__init__()
        self.feas = feas
        self.k = k
        self.deepDepth = deepDepth
        self.wide = nn.Linear(self.feas, 1)
        self.cross = nn.Parameter(torch.rand(self.feas, self.k))
        self.deep = nn.ModuleDict([["deep{}".format(i), nn.Sequential(nn.Linear(self.k//i, self.k//(i+1)), nn.ReLU())] for i in range(1, self.deepDepth+1)])
        self.out = nn.Linear(self.feas//(self.deepDepth+1), 1)

    def forward(self, input):
        linear = self.wide.forward(input)
        # 计算交叉向，进行特征交叉池化层
        tmp = torch.zeros((input.size(0), self.k))
        for step in range(input.size(0)):
            for i in range(input.size(1)):
                for j in range(i, input.size(1)):
                    if input[step, i] == 0:
                        break
                    if input[step, j] == 0:
                        continue
                    # 无差别的对特征交叉后的结果进行求和
                    tmp[step] = tmp[step] + input[step, i]*input[step, j]*torch.multiply(self.cross[i], self.cross[j])
        for i in range(1, self.deepDepth+1):
            tmp = self.deep['deep{}'.format(i)].forward(tmp)
        deepOut = self.out.forward(tmp)
        out = nn.Sigmoid().forward(linear+deepOut)
        return out



if __name__ == '__main__':
    # x = torch.rand(2, 5)
    # y = torch.rand(2, 1)
    # dataset = Data.TensorDataset(x, y)
    # nfm = NFM(5, 3, 2)
    # loader = Data.DataLoader(dataset, batch_size=50)
    # lossfun = nn.BCELoss()
    # optim = Optim.Adam(nfm.parameters())
    # print(nfm.get_parameter("cross"))
    # for i, (x, y) in enumerate(loader):
    #     out = nfm.forward(x)
    #     loss = lossfun.forward(out, y)
    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()
    # print(nfm.get_parameter("cross"))
    N, L = 4950, 100
    # 运用等差数列求解答案，假设由几个值构成的，可以反推等差数列首项为多少值
    for i in range(L, 101):
        start = (2*N-i*(i-1))/(2*i)
        if start > 0 and start % 1 == 0:
            res = [str(j) for j in range(int(start), int(start)+i)]
            print(" ".join(res))
            exit()
    print("No")