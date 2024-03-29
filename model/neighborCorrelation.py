import torch
import torch.nn as nn

class NeighborCorrelation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeighborCorrelation, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    #参数h表示传过来的特征
    def forward(self, feat):
        feat_transformed = self.linear(feat)
        return feat_transformed