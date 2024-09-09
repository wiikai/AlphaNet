import torch
import torch.nn as nn
from operators import TsCorr, TsCov, TsSum, TsStddev, TsMean, TsCv, TsMax, TsMin

class AlphaNetV1(nn.Module):
    def __init__(self, num_features, size, rolling):
        super(AlphaNetV1, self).__init__()

        out_feature_length =  num_features * (num_features - 1) // 2

        # 特征提取层
        self.feature_layers = nn.ModuleList([
            nn.Sequential(TsCorr(size=size, stride=1)),
            nn.Sequential(TsCov(size=size, stride=1)),

            nn.Sequential(TsStddev(size=size, stride=1)),
            nn.Sequential(TsSum(size=size, stride=1)),
            nn.Sequential(TsMean(size=size, stride=1)),
            nn.Sequential(TsCv(size=size, stride=1))
        ])

        # 池化层
        self.pooling_layers = nn.ModuleList([
            TsMean(size=rolling, stride=1),
            TsStddev(size=rolling, stride=1)
        ])

        # 批标准化层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(out_feature_length),
            nn.BatchNorm1d(out_feature_length),

            nn.BatchNorm1d(num_features),
            nn.BatchNorm1d(num_features),
            nn.BatchNorm1d(num_features),
            nn.BatchNorm1d(num_features)
        ])

        # 全连接层
        input_dim = (out_feature_length * 2 + num_features * 4) + 2 * (out_feature_length * 2 + num_features * 4)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self._init_weights()

    def forward(self, x):
        feature_outputs = [layer(x) for layer in self.feature_layers]
        results = [bn(feature[:, :, 0]) for feature, bn in zip(feature_outputs, self.bn_layers)]

        for pooling_layer in self.pooling_layers:
            results += [bn_layer(pooling_layer(feature)).squeeze(-1) for feature, bn_layer in zip(feature_outputs, self.bn_layers)]
                
        # 展平所有的特征输出
        x = torch.cat([f.view(f.size(0), -1) for f in results], dim=1)
        x = self.fc_layers(x)
        
        return x
        
    def _init_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight) 
                nn.init.normal_(layer.bias, std=1e-6)
        