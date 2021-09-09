"""
    属于wide&deep的改进版，将wide部分替换为FM模型；
    另外的改进就是：wide和deep部分是共享embedding输入（相比sparse特征的输入，计算量大大减少了），增强了wide部分的表达
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as Optim


class FM(nn.Module):
    def __init__(self, feas, k):
        super(FM, self).__init__()
        self.feas = feas  # embedding层拼接数值特征总的特征数量
        self.k = k
        self.cross = nn.Parameter(torch.randn(self.feas, self.k))
        self.linear = nn.Linear(self.feas, 1)  # FM的线性部分

    def forward(self, input):
        """
        :param input: shape(batchSize, feas)
        :return:
        """
        linear = self.linear.forward(input)
        cross = torch.triu(torch.multiply(torch.mm(input.t(), input), torch.mm(self.cross, self.cross.t())))
        cross = torch.sum(cross)
        out = torch.add(linear, cross)
        return out


class Deep(nn.Module):
    def __init__(self, feas, deepDepth):
        super(Deep, self).__init__()
        self.feas = feas
        self.deepDepth = deepDepth
        self.deep = nn.ModuleDict([["deep{}".format(i), nn.Sequential(nn.Linear(self.feas//i, self.feas//(i+1)), nn.ReLU())]for i in range(1, self.deepDepth+1)]
        )
        self.out = nn.Linear(self.feas//(self.deepDepth+1), 1)

    def forward(self, input):
        for i in range(1, self.deepDepth+1):
            input = self.deep['deep{}'.format(i)].forward(input)
        out = self.out.forward(input)
        return out


class DeepFM(nn.Module):
    def __init__(self, feaDim, k, deepDepth, feaCols):
        super(DeepFM, self).__init__()
        torch.manual_seed(2021)
        self.feaDim = feaDim  # 特征的维度
        self.k = k  # FM隐向量的维度
        self.deepDepth = deepDepth  # Deep的层数
        self.feaCols = feaCols  # [a:[], b:{},{},{},.[..]],a为数值特征对应的列名，b为类别特征【每个值由字典组成{"catDim", "embedDim"}】
        self.embedLayer = {
            "embed{}".format(i): nn.Embedding(fea["catDim"]+1, fea["embedDim"], ) for i, fea in enumerate(self.feaCols[1])  # 第一行为默认值
        }

    def forward(self, input):
        """
        :param input: shape(batchsize, feas)
        :return:
        """
        denseFea = input[:, :len(self.feaCols[0])]
        # sparse特征embedding化
        sparseFeaIdx = input[:, len(self.feaCols[0]):]
        # Embedding的访问方式Embedding(tensor)
        sparseEmbed = torch.cat([self.embedLayer["embed{}".format(i)](sparseFeaIdx[:, i]) for i in range(sparseFeaIdx.size(1))], dim=1)
        # 拼接dense和sparse
        embedFea = torch.cat((denseFea, sparseEmbed), dim=1)
        feaDim = embedFea.size(1)
        fm = FM(feaDim, self.k)
        deep = Deep(feaDim, self.deepDepth)
        # 组合FM和Deep输出
        fmOut = fm.forward(embedFea)
        deepOut = deep.forward(embedFea)
        out = nn.Sigmoid().forward(0.5*(fmOut+deepOut))
        return out


if __name__ == '__main__':
    x = torch.randint(1, 5, (1, 5))
    print(x)
    denseFea = ["a", "b", "c"]
    sparseFea = [{"catDim": 10, "embedDim": 4}, {"catDim": 10, "embedDim": 4}]
    feaCol = [denseFea, sparseFea]
    deepFM = DeepFM(10, 5, 5, feaCol)
    deepFM.forward(x)