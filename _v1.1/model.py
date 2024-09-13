import torch
import torch.nn as nn
from operators import TsCorr, TsCov, TsSum, TsStddev, TsMean, TsCv, TsMax, TsMin, TsAdd

class AlphaNetV1(nn.Module):
    def __init__(self, num_features, min, rolling):
        super(AlphaNetV1, self).__init__()

        # 特征提取层
        self.feature_layers = nn.ModuleList([
            TsCorr(min=min, stride=min),

            TsStddev(min=min, stride=min),
            TsSum(min=min, stride=min),
            TsMean(min=min, stride=min),
        ])

        # 交叉计算层
        self.cross_layers = nn.ModuleList([
            TsAdd(min=1, stride=1),
        ])

        # 滚动层
        self.rooling_layers = nn.ModuleList([
            TsMean(min=rolling, stride=1),
            TsStddev(min=rolling, stride=1),
            TsCv(min=rolling, stride=1)
        ])

        # 因子数量
        out_feature_length = (num_features * (num_features - 1) // 2) * 1  + num_features * 3
        out_cross_length = (out_feature_length * (out_feature_length - 1) // 2) * 1
        out_rolling_length = out_cross_length * 3

        # 批标准化层
        self.bn = nn.BatchNorm1d(out_rolling_length)

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(out_rolling_length, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self._init_weights()

    def forward(self, x):
        feature_outputs = torch.cat([layer(x) for layer in self.feature_layers], dim=1)
        cross_outputs = torch.cat([layer(feature_outputs) for layer in self.cross_layers], dim=1)
        rolling_outputs = torch.cat([layer(cross_outputs) for layer in self.rooling_layers], dim=1)

        x = self.bn(rolling_outputs).squeeze(-1)
        x = self.fc_layers(x).squeeze()
        return x
        
    def _init_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight) 
                nn.init.normal_(layer.bias, std=1e-6)
        