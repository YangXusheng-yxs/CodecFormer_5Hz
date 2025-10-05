import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import resample
from einops import rearrange
from pathlib import Path

from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMModel
)

import numpy as np
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional, Tuple, List, Dict, Any, Union

from tools.tokenizer.hytokenize.quantization.residual_vq import ResidualVQ as ResidualFactorVQ
from ..utils import exists, get_mask_from_lengths, eval_decorator
from .seanet import SEANetEncoder, SEANetDecoder
from .wavenext import WaveNextEncoder, WaveNextDecoder
from .transformer import ProjectedTransformer
from .resample_net import ConvDownsample1d, ConvTrUpsample1d


class MimiCodecEncWithFactorVQ(nn.Module):
    def __init__(
        self,
        *,
        seanet_conf: Dict[str, Any],
        transformer_conf: Optional[Dict[str, Any]] = {},
        vq_conf: Optional[Dict[str, Any]] = None,
        ds_rate: int = 1,
        token_pad_id = -1,
        init_model: Optional[str] = None,
        sample_rate: int = 24000,
        audio_do_normalize: bool = False,
        **kwargs
    ):
        super().__init__()
        self.ds_rate = ds_rate
        self.token_pad_id = token_pad_id
        self.sample_rate = sample_rate
        self.audio_do_normalize = audio_do_normalize

        # VQ
        if vq_conf is None:
            vq_conf = {
                "num_quantizers": 1,
                "dim": 1024,
                "codebook_size": 8192,
                "codebook_dim": 8,
                "threshold_ema_dead_code": 2,
                "commitment": 0.25,
                "weight_init": False,
                "full_commit_loss": False,
            }
        self.vq = ResidualFactorVQ(**vq_conf)
        
        # SEANet
        self.encoder = SEANetEncoder(**seanet_conf)
        self.hidden_size = self.encoder.dimension
        self.hop_size = self.encoder.hop_length
        
        # transformer
        self.encoder_transformer = None
        if transformer_conf:
            transformer_conf["input_dimension"] = self.hidden_size
            transformer_conf["output_dimensions"] = [vq_conf["dim"]]
            self.encoder_transformer = ProjectedTransformer(device=torch.device("cpu"), **transformer_conf)
        else:
            self.encoder_transformer = nn.Conv1d(self.hidden_size, vq_conf["dim"], 1)

        # projection
        if ds_rate > 1:
            self.proj = ConvDownsample1d(
                                ds_rate,
                                dimension=vq_conf["dim"],
                                learnt=True,
                                causal=seanet_conf.get("causal", False),
                            )
        else:
            self.proj = nn.Identity()
            
        # init model
        if init_model is not None:
            model_dict = torch.load(init_model, map_location="cpu")
            self.load_state_dict(model_dict)

    def train(self, mode=True):
        super().train(mode)

    def zero_unit_norm(self, wav_input, wav_lens, eps=1e-7):
        mask = get_mask_from_lengths(wav_lens, max_len=wav_input.shape[1]).to(wav_input.dtype)
        wav_input = wav_input * mask
        mean = wav_input.sum(dim=-1, keepdims=True) / wav_lens.unsqueeze(1)
        sqr_mean = (wav_input * wav_input).sum(dim=-1, keepdims=True) / wav_lens.unsqueeze(1)
        var = sqr_mean - mean * mean
        normed = (wav_input - mean) / (var + eps).sqrt()
        normed = normed * mask
        return normed

    def get_hidden_feat(
        self,
        x,
        x_lens = None,
        input_sample_rate = None
    ):
        assert x.dim() == 2, f"wav input for mimi encoder should with dim==2"
        if x_lens is None:
            x_lens = torch.tensor([x.shape[1]] * x.shape[0], device=x.device).int()
        # 计算需要填充的长度，使其成为 self.hop_size 1280 的整数倍
        batch_size = x.shape[0]
        current_length = x.shape[1]
        remainder = current_length % self.hop_size
        if remainder != 0:
            pad_length = self.hop_size - remainder
            x = torch.nn.functional.pad(x, (0, pad_length))  # 在时间轴上填充
            x_lens = x_lens + pad_length  # 更新音频长度
        # resample
        if input_sample_rate is not None and input_sample_rate != self.sample_rate:
            x = resample(x, orig_freq=input_sample_rate, new_freq=self.sample_rate)
            x_lens = (x_lens / input_sample_rate * self.sample_rate).ceil().int()
            x_lens = torch.clamp(x_lens, max=x.shape[1])
        # normalize
        if self.audio_do_normalize:
           x = self.zero_unit_norm(x, x_lens)
        padding_mask = get_mask_from_lengths(x_lens, max_len=x.shape[1])
        
        feat = self.encoder(x.unsqueeze(1))
        if isinstance(self.encoder_transformer, ProjectedTransformer):
            (feat,) = self.encoder_transformer(feat) # [B, C, T]
        elif isinstance(self.encoder_transformer, nn.Conv1d):
            feat = self.encoder_transformer(feat) # [B, C, T]

        feat_lens = x_lens // self.hop_size
        feat_lens = torch.clamp(feat_lens, max=feat.shape[-1])
        return feat, feat_lens

    def forward(
        self,
        wav_input,
        wav_lens = None,
        input_sample_rate = None,
        **kwargs
    ):
        all_hidden_feats, feat_lens = self.get_hidden_feat(
                wav_input, x_lens=wav_lens, input_sample_rate=input_sample_rate)
        hidden_feats = self.proj(all_hidden_feats)
        token_lens = feat_lens // self.ds_rate
        mask = get_mask_from_lengths(token_lens, max_len=hidden_feats.shape[-1])
        hidden_feats = hidden_feats.masked_fill(~mask.unsqueeze(1), 0.0)
        
        quantized, indices, vq_loss = self.vq(hidden_feats)
        
        vq_loss_items = {"vq_loss": vq_loss.sum().item()}
        tokens = indices.masked_fill(~mask, self.token_pad_id)
        quantized = quantized.masked_fill(~mask.unsqueeze(1), 0.0)
        res = {
            "quantized": quantized.transpose(1, 2),
            "tokens": tokens,
            "token_lens": token_lens,
            "vq_loss": vq_loss.sum(),
            "vq_loss_items": vq_loss_items,
            "all_hidden_feats": all_hidden_feats,
            "hidden_feat_lens": feat_lens,
            "before_quantize": hidden_feats.transpose(1, 2)
        }
        return res

    @eval_decorator
    @torch.no_grad()
    def extract_token(
        self,
        wav_input,
        wav_lens = None,
        input_sample_rate = None,
        **kwargs
    ):
        res = self.forward(wav_input, wav_lens=wav_lens, input_sample_rate=input_sample_rate, **kwargs)
        token, token_lens = res["tokens"], res.get("token_lens", None)
        return token, token_lens

    @eval_decorator
    @torch.no_grad()
    def extract_feat_before_quantize(
        self,
        wav_input,
        wav_lens = None,
        input_sample_rate = None,
        **kwargs
    ):
        res = self.forward(wav_input, wav_lens=wav_lens, input_sample_rate=input_sample_rate, **kwargs)
        before_quantize = res["before_quantize"]
        feat_lens = res.get("token_lens", None)
        return before_quantize, feat_lens


class MimiCodecDec(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        seanet_conf: Dict[str, Any],
        transformer_conf: Optional[Dict[str, Any]] = {},
        up_rate: int = 1,
        gin_channels: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.gin_channels = gin_channels
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
            
        # projection
        self.up_rate = up_rate
        if up_rate > 1:
            self.proj = ConvTrUpsample1d(
                                up_rate,
                                dimension=in_channels,
                                learnt=True,
                                causal=seanet_conf.get("causal", False),
                                channel_wise=True,
                            )
        else:
            self.proj = nn.Identity()
            
        # SEANet
        self.decoder = SEANetDecoder(**seanet_conf)
        self.hidden_size = self.decoder.dimension
        self.hop_size = self.decoder.hop_length
        
        # transformer
        self.decoder_transformer = None
        if transformer_conf:
            transformer_conf["input_dimension"] = in_channels
            transformer_conf["output_dimensions"] = [self.hidden_size]
            self.decoder_transformer = ProjectedTransformer(device=torch.device("cpu"), **transformer_conf)
        else:
            self.decoder_transformer = nn.Conv1d(in_channels, self.hidden_size, 1)

    def forward(self, x, g=None, **kwargs):
        x = x.transpose(1, 2) # [B, C, T]
        if g is not None:
            assert hasattr(self, 'cond')
            if g.dim() == 2:
                g = g.unsqueeze(-1)
            x = x + self.cond(g)
            
        x = self.proj(x) # [B, C, T]
        
        if isinstance(self.decoder_transformer, ProjectedTransformer):
            (x,) = self.decoder_transformer(x) # [B, C, T]
        elif isinstance(self.decoder_transformer, nn.Conv1d):
            x = self.decoder_transformer(x) # [B, C, T]
        
        hidden_feat = None
        if kwargs.get("return_hidden_feat", False):
            hidden_feat = x.transpose(1, 2) # [B, T, C]
        x = self.decoder(x)
        
        if kwargs.get("return_hidden_feat", False):
            return x, hidden_feat
        
        return x

