import torch
import torch.nn as nn

class TsCorr(nn.Module):
    def __init__(self, size, stride):
        super(TsCorr, self).__init__()
        self.size = size
        self.stride = stride
        self.eps = 1e-8

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features * (num_features - 1) // 2, out_seq_length, device=x.device)

        k = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                for t in range(out_seq_length):
                    start = t * self.stride
                    end = start + self.size
                    segment_i = x[:, i, start:end]
                    segment_j = x[:, j, start:end]
                    
                    mean_i = segment_i.mean(dim=1, keepdim=True)
                    mean_j = segment_j.mean(dim=1, keepdim=True)
                    segment_i_centered = segment_i - mean_i
                    segment_j_centered = segment_j - mean_j
                    
                    corr = (segment_i_centered * segment_j_centered).sum(dim=1) / (
                        torch.sqrt((segment_i_centered**2).sum(dim=1)) * torch.sqrt((segment_j_centered**2).sum(dim=1)) + self.eps
                    )
                    output[:, k, t] = corr
                k += 1

        return output

class TsCov(nn.Module):
    def __init__(self, size=10, stride=1):
        super(TsCov, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features * (num_features - 1) // 2, out_seq_length, device=x.device)

        k = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                for t in range(out_seq_length):
                    start = t * self.stride
                    end = start + self.size
                    segment_i = x[:, i, start:end]
                    segment_j = x[:, j, start:end]
                    
                    mean_i = segment_i.mean(dim=1, keepdim=True)
                    mean_j = segment_j.mean(dim=1, keepdim=True)
                    segment_i_centered = segment_i - mean_i
                    segment_j_centered = segment_j - mean_j
                    
                    cov = (segment_i_centered * segment_j_centered).sum(dim=1) / (self.size - 1)
                    output[:, k, t] = cov
                k += 1

        return output
    
class TsStddev(nn.Module):
    def __init__(self, size=20, stride=1):
        super(TsStddev, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.size
                segment = x[:, i, start:end]
                
                mean_segment = segment.mean(dim=1, keepdim=True)
                segment_centered = segment - mean_segment
                
                stddev = torch.sqrt((segment_centered**2).sum(dim=1) / (self.size - 1))
                output[:, i, t] = stddev

        return output

class TsMean(nn.Module):
    def __init__(self, size=20, stride=1):
        super(TsMean, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.size
                segment = x[:, i, start:end]
                
                mean_segment = segment.mean(dim=1)
                output[:, i, t] = mean_segment

        return output

class TsCv(nn.Module):
    def __init__(self, size=20, stride=1):
        super(TsCv, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.size
                segment = x[:, i, start:end]
                
                mean_segment = segment.mean(dim=1, keepdim=True)
                segment_centered = segment - mean_segment

                stddev = torch.sqrt((segment_centered**2).sum(dim=1) / (self.size - 1))
                cv = stddev/mean_segment.squeeze()
                output[:, i, t] = cv

        return output
    
class TsSum(nn.Module):
    def __init__(self, size=20, stride=1):
        super(TsSum, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.size
                segment = x[:, i, start:end]
                
                sum_segment = segment.sum(dim=1)
                output[:, i, t] = sum_segment

        return output
    
class TsMax(nn.Module):
    def __init__(self, size=20, stride=1):
        super(TsMax, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.size
                segment = x[:, i, start:end]
                
                max_segment = segment.max(dim=1)[0]
                output[:, i, t] = max_segment

        return output

class TsMin(nn.Module):
    def __init__(self, size=20, stride=1):
        super(TsMin, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.size) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.size
                segment = x[:, i, start:end]
                
                min_segment = segment.min(dim=1)[0]
                output[:, i, t] = min_segment

        return output

