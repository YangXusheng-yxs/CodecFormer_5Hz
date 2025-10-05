import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum, Tensor
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class Linear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        w_init_gain: str = 'linear',
        activation = None,
        **kwargs
    ):
        super(Linear, self).__init__(
            in_channels,
            out_channels,
            bias=bias
        )

        self.activation = activation if activation is not None else nn.Identity()
        self.output_dim = out_channels
        if w_init_gain is not None:
            if isinstance(w_init_gain, str):
                gain = nn.init.calculate_gain(w_init_gain)
            else:
                gain = w_init_gain
            nn.init.xavier_uniform_(
                    self.weight, gain=gain)
        if bias:
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x, **kwargs):
        return self.activation(super(Linear, self).forward(x))


class MaxPool1d(nn.MaxPool1d):
    def __init__(
        self,
        kernel_size: int, 
        stride = None, 
        padding = None,
        dilation: int = 1, 
        return_indices: bool = False, 
        ceil_mode: bool = False,
        causal: bool = False,
        pad_value: float = -1e5,
        **kwargs
    ):
        self.causal = causal
        self.pad_value = pad_value # small pad value to ensure keep the first element
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super(MaxPool1d, self).__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

    def forward(self, x):
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0), value=self.pad_value).squeeze(2)

        return super(MaxPool1d, self).forward(x)


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        padding = None,
        causal: bool = False,
        bn: bool = False,
        activation = None,
        w_init_gain = None,
        input_transpose: bool = False,
        **kwargs
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        outputs = self.activation(self.bn(super(Conv1d, self).forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(Conv1d, self).extra_repr(), self.causal, self.transpose)


def WNConv1d(*args, **kwargs):
    return weight_norm(Conv1d(*args, **kwargs))


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = 'zeros',
        causal: bool = False,
        input_transpose: bool = False,
        **kwargs
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, \
                    "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."

        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )

        self.causal = causal
        self.stride = stride
        self.transpose = input_transpose

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, :-self.stride]
        return x.transpose(1, 2) if self.transpose else x

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(ConvTranspose1d, self).extra_repr(), self.causal, self.transpose)



def random_masking(mask, mask_prob, ignore_first=True):
    assert mask.ndim == 2
    lens = mask.shape[-1]
    rand = torch.randn(mask.shape, device=mask.device)
    if ignore_first:
        rand[:, 0] = -torch.finfo(rand.dtype).max # Ignore the first item
    num_mask = min(int(lens * mask_prob), lens - 1)
    indices = rand.topk(num_mask, dim=-1).indices
    new_mask = ~torch.zeros(mask.shape, device=mask.device).scatter(1, indices, 1.).bool()
    return new_mask


def get_mask_from_lengths(lengths, max_len=None, r=1, random_mask=0.):
    if max_len is None:
        max_len = torch.max(lengths).item()
    if max_len % r != 0:
        max_len = max_len + r - max_len % r
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    mask = (ids < lengths.unsqueeze(1)).bool()
    if random_mask > 0.:
        mask = mask.logical_and(random_masking(mask, random_mask))
    return mask

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(BaseModel, self).__init__()
        self.config = config
        self.integrity_test()

    def forward(self, feature):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, feature):
        return self.forward(feature)

    def remove_weight_norm(self):
        pass

    def integrity_test(self):
        pass


class ProjectorMLP(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.upsample_factor = config.get("upsample_factor", 1)
        self.downsample_factor = config.get("downsample_factor", 1)
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.cond_dim = config.get("cond_dim", -1)
        self.input_type = config.get("input_type", "token")


        if self.upsample_factor > 1:
            self.up_proj = nn.Sequential(
                ConvTranspose1d(
                    self.input_dim,
                    self.input_dim,
                    stride = self.upsample_factor,
                    kernel_size = self.upsample_factor * 2)
            )
        else:
            self.up_proj = nn.Identity()
            
        if self.downsample_factor > 1:
            self.ds_proj = Conv1d(self.input_dim, self.input_dim, self.downsample_factor, self.downsample_factor)
        else:
            self.ds_proj = nn.Identity()
            
        self.cond_proj = None
        if self.cond_dim > 0:
            self.cond_proj = Linear(self.cond_dim, self.input_dim)
            
        self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        
    def forward(self, x, x_lens=None, condition=None):
        if x_lens is None:
            x_lens = torch.zeros(x.shape[0], dtype=torch.long, device=x.device) + x.shape[1]
        
        # up part
        x_lens = x_lens * self.upsample_factor
        hidden_feats = self.up_proj(x)
        
        # down part
        x_lens = x_lens // self.downsample_factor
        hidden_feats = self.ds_proj(x.transpose(1, 2))
        hidden_feats = rearrange(hidden_feats, 'b d n -> b n d')
        mask = get_mask_from_lengths(x_lens, max_len=hidden_feats.shape[1])
        
        if self.cond_proj is not None and condition is not None:
            if condition.dim() == 2:
                condition = condition.unsqueeze(1)
            condition = self.cond_proj(condition)
            hidden_feats = hidden_feats + condition
        
        outputs = self.mlp(hidden_feats)
        
        return outputs, x_lens
        