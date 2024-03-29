import torch
import torch.nn as nn

class AdaptiveAggreModel(nn.Module):
    def __init__(self):
        super(AdaptiveAggreModel, self).__init__()
        #self.W_1 = nn.Parameter(torch.randn(512, 1))
        #self.b_1 = nn.Parameter(torch.randn(1))

        '''
        这个地方的模型可以看看再修正一下，多加上几个线性层以及激活函数
        '''
        self.linear1 = nn.Linear(128, 1)  #
        # self.linear1 = nn.Linear(256, 128)  #
        # self.linear2 = nn.Linear(128, 64)
        # self.linear3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()


        #最后看看是否需要把上面的变换矩阵W和偏置b修改为nn.Linear的形式

    def forward(self, object_feat, neight_feat):
        #learn_parameter = torch.sigmoid(torch.matmul(object_feat * neight_feat, self.W_1) + self.b_1)
        learn_parameter = torch.sigmoid(self.linear1(object_feat * neight_feat))

        # learn_parameter = torch.relu(self.linear1(object_feat * neight_feat))
        # learn_parameter = torch.relu(self.linear2(learn_parameter))
        # learn_parameter = self.sigmoid(self.linear3(learn_parameter))

        learn_parameter_expand = learn_parameter.expand(-1, object_feat.size(1))
        # temp = 1 - learn_parameter_expand #测试temp当中的结果
        # tmp1 = learn_parameter_expand * object_feat
        # tmp2 = temp * neight_feat
        S_f = learn_parameter_expand * object_feat + (1 - learn_parameter_expand) * neight_feat
        return S_f

