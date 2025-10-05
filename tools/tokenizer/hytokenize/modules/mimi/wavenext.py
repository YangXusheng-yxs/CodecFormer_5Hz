import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNextEncoder(nn.Module):
    def __init__(self, 
                 hop_length: int,
                 channels: int,
                 out_channels: int,
                 **kwargs,
                 ):
        super().__init__()
        
        self.dimension = out_channels
        self.hop_length = hop_length
        self.linear_1 = torch.nn.Linear(hop_length, channels, bias=False)
        self.linear_2 = torch.nn.Linear(channels, out_channels)

        # W init
        nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        nn.init.trunc_normal_(self.linear_2.weight, std=0.02)
        
        
    def forward(self, x):
        B = x.shape[0] # [B, 1, T]
        x = x.reshape(B, -1, self.hop_length)
        x = self.linear_1(x)
        x = self.linear_2(x)
        
        return x.transpose(1, 2) # [B, C, T']
    

class WaveNextDecoder(nn.Module):
    def __init__(self, 
                 hop_length: int,
                 channels: int,
                 in_channels: int,
                 tanh: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        
        self.dimension = in_channels
        self.hop_length = hop_length
        self.act = nn.Tanh() if tanh else nn.Identity()
        self.linear_1 = torch.nn.Linear(in_channels, channels)
        self.linear_2 = torch.nn.Linear(channels, hop_length, bias=False)

        # W init
        nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        nn.init.trunc_normal_(self.linear_2.weight, std=0.02)
        
        
    def forward(self, x):
        B, C, T = x.shape
        x = x.transpose(1, 2)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = x.transpose(1, 2)
        x = x.reshape(B, 1, -1)
        x = torch.clip(self.act(x), min=-1.0, max=1.0)
        return x 
        
