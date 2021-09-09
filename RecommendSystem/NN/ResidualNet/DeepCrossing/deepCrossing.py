import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
# from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


EPOCH = 10
BATCHSIZE = 50
LR = 0.01


class ResidualUnit(nn.Module):  # 残差单元
    def __init__(self, iDim, hDim):
        super(ResidualUnit, self).__init__()
        self.iDim = iDim
        self.hDim = hDim
        self.residualUnit = nn.Sequential(
            nn.Linear(self.iDim, self.hDim),
            nn.ReLU(),
            nn.Linear(self.hDim, self.iDim),
        )

    def forward(self, input):  # 残差单元前向传播的输出
        out = self.residualUnit.forward(input)
        out += input
        out = nn.ReLU().forward(out)
        return out


class loadDataset(Data.Dataset):
    def __init__(self, filename):  # 设置数据集的一些初始信息  结构化数据：可以一次性加载到内存，  对于非结构化数据：可以加载其路径，进行存储，然后使用的时候再读取
        super(loadDataset, self).__init__()
        self.dataset = np.loadtxt(filename, delimiter=',')

    def __len__(self):  # 返回数据集的长度
        return len(self.dataset)

    def __getitem__(self, item):  # 根据索引item，读取数据
        self.X = torch.from_numpy(self.dataset[item, :-1]).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array([self.dataset[item, -1]])).type(torch.FloatTensor)
        return self.X, self.y


class DeepCrossingNet(nn.Module):
    def __init__(self, inputDim, outputDim, layers):
        super(DeepCrossingNet, self).__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layers = layers
        self.lossFun = nn.BCELoss()
        # 对残差单元进行堆叠，组合磨料
        self.residualLayer = nn.ModuleList([
            ResidualUnit(self.inputDim, i) for i in range(10)
        ])
        self.outLayer = nn.Sequential(
            nn.Linear(self.inputDim, self.outputDim),
            nn.Sigmoid(),
        )

    def forward(self, input):
        for residual in self.residualLayer:  # 使用残差单元堆叠的残差网络
            input = residual.forward(input)
        out = self.outLayer.forward(input)
        out = torch.squeeze(out, dim=1)
        return out

    def fit(self, optimizer):
        dataset = loadDataset("data.txt")
        # dataset = Data.TensorDataset(torch.from_numpy(np.array(X)).type(torch.FloatTensor), torch.from_numpy(np.array(y)).type(torch.FloatTensor))
        loader = Data.DataLoader(dataset, BATCHSIZE, True)
        for epoch in range(1, EPOCH+1):
            for step, (trainx, trainy) in enumerate(loader):
                predict = self.forward(trainx)
                predict = predict.view(-1, 1)
                loss = self.lossFun.forward(predict, trainy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 20 == 0:
                    print("epoch_{}({}/10), step_{}({}/{}),*****loss:{}******".format(epoch, epoch, step, step, len(loader), loss.data.numpy()))
        return self

    def saveModel(self, filename):
        torch.save(self, "./%s.h5" %(filename))

    def predict(self, testX):
        testX = [[9, 2, 3, 140, 5, 6], [9, 2, 3, 140, 5, 6]]
        testX = torch.FloatTensor(testX)
        res = self.forward(testX).detach().numpy()
        res[res < 0.5] = 0
        res[res >= 0.5] = 1
        print(res)


if __name__ == '__main__':
    deepCross = DeepCrossingNet(6, 1, 150)
    print(deepCross)
    # optimizer = torch.optim.Adam(deepCross.parameters())
    # # print(deepCross)
    # deepCross.fit(optimizer)
    # deepCross.saveModel()