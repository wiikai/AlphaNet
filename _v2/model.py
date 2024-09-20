import torch
import torch.nn as nn
from operators import TsCorr, TsCov, TsSum, TsStddev, TsMean, TsCv, TsMax, TsMin, TsAdd, TsSub, TsWeight

class AlphaNetV2(nn.Module):
    def __init__(self, num_features, min):
        super(AlphaNetV2, self).__init__()
        
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
            # TsSub(min=1, stride=1),
        ])

        # 因子数量
        out_input_length =  (num_features * (num_features - 1) // 2) + num_features 
        out_feature_length = (out_input_length * (out_input_length - 1) // 2) * 1 + out_input_length * 2
        out_cross_length = (out_feature_length * (out_feature_length - 1) // 2) * 1

        # 批标准化层
        self.bn = nn.BatchNorm1d(out_cross_length)

        # 全连接层
        self.lstm = nn.LSTM(
                input_size=out_cross_length, 
                hidden_size=30, 
                num_layers=1,
                batch_first=True
            )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(30, 1)
        )

        self._init_weights()

    def forward(self, x):
        input_outputs = torch.cat([layer(x) for layer in self.input_layers], dim=1)
        x = torch.cat([x, input_outputs], dim=1)

        feature_outputs = torch.cat([layer(x) for layer in self.feature_layers], dim=1)
        cross_outputs = torch.cat([layer(feature_outputs) for layer in self.cross_layers], dim=1)
        x = self.bn(cross_outputs).transpose(1, 2)

        lstm_out, (h_n, c_n) = self.lstm(x)  # 输入 x 大小为 [batch_size, time_step, lstm_input_dim]
        last_output = lstm_out[:, -1, :] # 最后一个时间步长的隐藏层

        x = self.fc_layers(last_output).squeeze()
        return x
        
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  
                param.data.fill_(0)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(layer.bias, std=1e-6)
        