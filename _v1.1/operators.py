import torch
import torch.nn as nn

class TsCorr(nn.Module):
    def __init__(self, min, stride):
        super(TsCorr, self).__init__()
        self.min = min
        self.stride = stride
        self.eps = 1e-8

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features * (num_features - 1) // 2, out_seq_length, device=x.device)

        k = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                for t in range(out_seq_length):
                    start = t * self.stride
                    end = start + self.min
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
    def __init__(self, min, stride):
        super(TsCov, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features * (num_features - 1) // 2, out_seq_length, device=x.device)

        k = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                for t in range(out_seq_length):
                    start = t * self.stride
                    end = start + self.min
                    segment_i = x[:, i, start:end]
                    segment_j = x[:, j, start:end]
                    
                    mean_i = segment_i.mean(dim=1, keepdim=True)
                    mean_j = segment_j.mean(dim=1, keepdim=True)
                    segment_i_centered = segment_i - mean_i
                    segment_j_centered = segment_j - mean_j
                    
                    cov = (segment_i_centered * segment_j_centered).sum(dim=1) / (self.min - 1)
                    output[:, k, t] = cov
                k += 1

        return output
    
class TsStddev(nn.Module):
    def __init__(self, min, stride):
        super(TsStddev, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.min
                segment = x[:, i, start:end]
                
                mean_segment = segment.mean(dim=1, keepdim=True)
                segment_centered = segment - mean_segment
                
                stddev = torch.sqrt((segment_centered**2).sum(dim=1) / (self.min - 1))
                output[:, i, t] = stddev

        return output

class TsMean(nn.Module):
    def __init__(self, min, stride):
        super(TsMean, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.min
                segment = x[:, i, start:end]
                
                mean_segment = segment.mean(dim=1)
                output[:, i, t] = mean_segment

        return output

class TsCv(nn.Module):
    def __init__(self, min, stride):
        super(TsCv, self).__init__()
        self.min = min
        self.stride = stride
        self.eps = 1e-8

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.min
                segment = x[:, i, start:end]
                
                mean_segment = segment.mean(dim=1, keepdim=True)
                segment_centered = segment - mean_segment

                stddev = torch.sqrt((segment_centered**2).sum(dim=1) / (self.min - 1))
                cv = stddev/(mean_segment.squeeze() + self.eps)
                cv = torch.where(torch.isnan(cv), torch.zeros_like(cv), cv)
                output[:, i, t] = cv

        return output
    
class TsSum(nn.Module):
    def __init__(self, min, stride):
        super(TsSum, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.min
                segment = x[:, i, start:end]
                
                sum_segment = segment.sum(dim=1)
                output[:, i, t] = sum_segment

        return output
    
class TsMax(nn.Module):
    def __init__(self, min, stride):
        super(TsMax, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.min
                segment = x[:, i, start:end]
                
                max_segment = segment.max(dim=1)[0]
                output[:, i, t] = max_segment

        return output

class TsMin(nn.Module):
    def __init__(self, min, stride):
        super(TsMin, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features, out_seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.min
                segment = x[:, i, start:end]
                
                min_segment = segment.min(dim=1)[0]
                output[:, i, t] = min_segment

        return output

class TsAdd(nn.Module):
    def __init__(self, min, stride):
        super(TsAdd, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features * (num_features - 1) // 2, out_seq_length, device=x.device)

        k = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                for t in range(out_seq_length):
                    start = t * self.stride
                    end = start + self.min
                    segment_i = x[:, i, start:end]
                    segment_j = x[:, j, start:end]
                    
                    add = (segment_i + segment_j).squeeze()
                    output[:, k, t] = add
                k += 1

        return output

class TsSub(nn.Module):
    def __init__(self, min, stride):
        super(TsSub, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features * (num_features - 1) // 2, out_seq_length, device=x.device)

        k = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                for t in range(out_seq_length):
                    start = t * self.stride
                    end = start + self.min
                    segment_i = x[:, i, start:end]
                    segment_j = x[:, j, start:end]
                    
                    add = (segment_i - segment_j).squeeze()
                    output[:, k, t] = add
                k += 1

        return output

class TsWeight(nn.Module):
    def __init__(self, min, stride):
        super(TsWeight, self).__init__()
        self.min = min
        self.stride = stride

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features * (num_features - 1) // 2, seq_length, device=x.device)

        k = 0  
        for i in range(num_features):
            for j in range(i + 1, num_features): 
                for t in range(out_seq_length):
                    start = t * self.stride
                    end = start + self.min

                    segment_i = x[:, i, start:end]
                    segment_j = x[:, j, start:end]
                    
                    sum_segment = segment_i.sum(dim=1, keepdim=True)
                    weight = segment_i / sum_segment

                    weighted_product = weight * segment_j
                    output[:, k, start:end] = weighted_product

                k += 1 

        return output
    
class TsLnret(nn.Module):
    def __init__(self, min, stride):
        super(TsLnret, self).__init__()
        self.min = min  
        self.stride = stride  

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape
        out_seq_length = (seq_length - self.min) // self.stride + 1
        output = torch.zeros(batch_size, num_features, seq_length, device=x.device)

        for i in range(num_features):
            for t in range(out_seq_length):
                start = t * self.stride
                end = start + self.min
                segment = x[:, i, start:end]

                lnret = torch.log(segment[:, 1:] / segment[:, :-1])
                output[:, i, start+1:end] = lnret

        output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        return output
