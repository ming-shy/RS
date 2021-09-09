import torch
import torch.nn as nn


class GRU(nn.Module):
    r"""自定义实现GRU:核心计算公式如下：

        .. math::
            \begin{matrix}
                更新门：u_t=\sigma(w_u\{x_t,h_{t-1}\}+b_u)  \\
                重置门：r_t=\sigma(w_r\{x_t,h_{t-1}\}+b_r) \\
                \tilde{h}_t=tanh(w_h\{x_t,r_t\odot h_{t-1}\}+b_h)  \\
                h_t=(1-u_t)\odot h_{t-1}+u_t\odot \tilde{h}_t
            \end{matrix}

    """
    def __init__(self, batchSize, seqLength, seqFeas, isAttention=False):
        super(GRU, self).__init__()
        self.batchSize = batchSize   # batchSize的大小
        self.seqLength = seqLength  # 序列长度，即有多少个t
        self.seqFeas = seqFeas  # 每个序列包含的特征数
        self.isAttention = isAttention  # 注意力得分
        # 更新门
        self.update = nn.Sequential(
            nn.Linear(2*self.seqFeas, self.seqFeas),
            nn.Sigmoid()
        )
        # 重置门
        self.reset = nn.Sequential(
            nn.Linear(2*self.seqFeas, self.seqFeas),
            nn.Sigmoid()
        )
        # 经过重置门，上一时刻的输出变化量
        self.hChange = nn.Sequential(
            nn.Linear(2*self.seqFeas, self.seqFeas),
            nn.Tanh()
        )
        # 注意力机制，具体的层数可以按照实际情况进行调整
        self.attention = nn.Sequential(
            nn.Linear(3*self.seqFeas, 3*self.seqFeas//2),
            nn.ReLU(),
            nn.Linear(3*self.seqFeas//2, 3*self.seqFeas//4),
            nn.ReLU(),
            nn.Linear(3*self.seqFeas//4, 1),
        )

    def forward(self, input, targetItem):
        r"""
        :param input: shape(batchSize, seqLength, seqFeas)
        :param targetItem: shape(batcSize, seqFeas) 目标item的embedding
        :return: 每一seq的输出
        """
        batchSize = input.size(0)
        seqLength = input.size(1)
        seqFeas = input.size(2)
        res = torch.zeros(batchSize, 1, seqFeas)
        h = input[:, 0]  # 上一时刻的输出，最开始的时刻，即为第一个值
        for i in range(seqLength):
            x_t = input[:, i]  # 当前时刻的输入,二维的
            cat = torch.cat((x_t, h), dim=1)
            u = self.update.forward(cat)
            r = self.reset.forward(cat)
            # 计算注意力得分
            if self.isAttention:
                # x_t 与 targetItem 进行互操作，元素积
                attFea = torch.cat((x_t, torch.multiply(x_t, targetItem), targetItem), dim=1)
                aij = self.attention.forward(attFea)
                u = aij * u  # AUGRU：将注意力得分与更新门进行乘积，完成兴趣进化层的定义
            hChange = self.hChange.forward(torch.cat((x_t, torch.multiply(r, h)), dim=1))
            h = torch.multiply((1-u), h) + torch.multiply(u, hChange)  # 当前时刻的输出
            # 对结果进行拼接
            res = torch.cat((res, h.reshape(batchSize, 1, seqFeas)), dim=1)
        res = res[:, 1:, :]  # 移除初始0的行
        return res


if __name__ == '__main__':
    targetItem = torch.randn(2, 5)
    gru = GRU(2, 3, 5, attention=True)
    input = torch.rand(2, 3, 5)
    print(gru.forward(input, targetItem))