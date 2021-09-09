# pytorch实现的LS-PLM:大规模分段线性模型，又称MLR混合逻辑回归
# LS-PLM由softmax函数和LR组成，softmax用于将样本分组，LR对分组的样本进行预测，二者相乘，所有组相加 -> 最后结果
# 如何自定义损失函数，加上正则项？
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data


class LS_PLM(nn.Module):
    def __init__(self, n=1, m=12, alpha=1e-3, beta=1e-3, loss_fun=nn.BCELoss(), optimier="Adam", epoch=100, batchsize=50):
        super(LS_PLM, self).__init__()
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.loss_fun = loss_fun
        self.epoch = epoch
        self.batchize = batchsize
        self.optimizer = optimier
        self.softmax = nn.Sequential(
            nn.Linear(self.n, self.m),
            nn.Softmax(dim=0)
        )
        self.logistic = nn.Sequential(
            nn.Linear(self.n, self.m),
            nn.Sigmoid()
        )
        self.activation = nn.Sigmoid()

    # LS-PLM的核心，前向传播，即目标函数的定义
    def forward(self, input):
        '''
        :param input: shape(batchSize,n_features)
        :return:shape(1, batchsize)
        '''
        softmax_out = self.softmax.forward(input)
        logistic_out = self.logistic.forward(input)
        combine_out = torch.mul(softmax_out, logistic_out)
        combine_out = torch.sum(combine_out, dim=1)
        combine_out = self.activation.forward(combine_out).type(torch.FloatTensor)
        return combine_out

    # 自定义L1损失,在LS-PLM中对所有参数进行L1正则化
    def l1Loss(self):
        l1_loss = torch.tensor(0.0, requires_grad=True)
        for name, params in self.named_parameters():
            if 'bias' not in name:
                l1_loss = l1_loss + torch.sum(torch.abs(params))
        l1_loss = 0.5 * self.alpha * l1_loss
        return l1_loss

    # 自定义L2损失，在LS-PLM中对每一个特征的所有参数进行L2正则化，在对其进行L1正则化,L_21
    def l2Loss(self):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        squre_res = []
        for name, params in self.named_parameters():
            if 'bias' not in name:
                tmp = torch.sum(torch.pow(params, 2), dim=0)
                squre_res.append(tmp)
        squre_res = torch.stack((squre_res[0], squre_res[1]), dim=0)  # 合并tensor
        l2_loss = self.beta * (l2_loss + torch.sum(torch.sqrt(torch.sum(squre_res, dim=0))))
        return l2_loss

    def fit(self, X, y):
        '''
        :param X: List[List[]]:训练集
        :param y: List:标签
        :return: self
        '''
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        # 将X,y放入dataloader中
        dataset = Data.TensorDataset(torch.from_numpy(np.array(X)).type(torch.FloatTensor), torch.from_numpy(np.array(y)).type(torch.FloatTensor))
        loader = Data.DataLoader(dataset, batch_size=self.batchize, shuffle=True)
        for epoch in range(self.epoch):
            for idx, (batch_X, batch_y) in enumerate(loader):
                out = self.forward(batch_X)
                loss = self.loss_fun(out, batch_y)  # 损失函数计算的值，可在其后增加正则化项
                print(loss)
                loss = loss + self.l1Loss() + self.l2Loss()
                print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict_proba(self, X):
        '''
        :param X:list[list[]]
        :return:
        '''
        X = torch.from_numpy(np.array(X)).type(torch.FloatTensor)
        return self.forward(X)

    def predict(self, X):
        X = torch.from_numpy(np.array(X)).type(torch.FloatTensor)
        out = self.forward(X)
        out[out >= 0.5] = 1.0
        out[out < 0.5] = 0.0
        return out


if __name__ == '__main__':
    train_X = [[0.2, 0.5, 0.6], [0.1, 0.75, 1.26], [2.6, 1.6, 3.9]]
    train_y = [1, 0, 0]
    ls_plm = LS_PLM(n=3)
    ls_plm.fit(train_X, train_y)
    res = ls_plm.predict_proba([[0.2, 0.6, 0.7], [3.6, 4.6, 9.5]])
    res1 = ls_plm.predict([[0.2, 0.6, 0.7], [3.6, 4.6, 9.5]])
    print(res.detach().numpy())
    print(res1.detach().numpy())

