import torch
import torch.nn as nn
from operators import TsCorr, TsCov, TsSum, TsStddev, TsMean, TsCv, TsMax, TsMin, TsAdd, TsSub, TsWeight

class AlphaNetV2(nn.Module):
    def __init__(self, num_features):
        super(AlphaNetV2, self).__init__()
        min = int(240/10)

        # 输入层
        self.input_layers = nn.ModuleList([
            TsWeight(min=min, stride=min),
        ])

        # 特征提取层
        self.feature_layers = nn.ModuleList([
            TsCorr(min=min, stride=min),

            TsStddev(min=min, stride=min),
            TsSum(min=min, stride=min),
            # TsMean(min=min, stride=min),
        ])

        # 交叉计算层
        self.cross_layers = nn.ModuleList([
            TsAdd(min=1, stride=1),
            TsSub(min=1, stride=1),
        ])

        # 因子数量
        out_input_length =  (num_features * (num_features - 1) // 2) + num_features 
        out_feature_length = (out_input_length * (out_input_length - 1) // 2) * 1 + out_input_length * 2
        out_cross_length = (out_feature_length * (out_feature_length - 1) // 2) * 2

        # 批标准化层
        self.bn = nn.BatchNorm1d(out_cross_length)

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(out_cross_length, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self._init_weights()

    def forward(self, x):
        input_outputs = torch.cat([layer(x) for layer in self.input_layers], dim=1)
        x = torch.cat([x, input_outputs], dim=1)

        feature_outputs = torch.cat([layer(x) for layer in self.feature_layers], dim=1)
        cross_outputs = torch.cat([layer(feature_outputs) for layer in self.cross_layers], dim=1)

        x = self.bn(cross_outputs).squeeze(-1)
        x = self.fc_layers(x).squeeze()
        return x
        
    def _init_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight) 
                nn.init.normal_(layer.bias, std=1e-6)
        