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

from ..utils import exists, get_mask_from_lengths, eval_decorator


class WavLMTeacher(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_path,
        layer: Optional[int] = None,
        use_regressor: bool = False,
        regress_layers: Optional[List[int]] = None,
        output_dim: Optional[int] = None,
        init_model: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.layer = layer
        self.use_regressor = use_regressor
        self.regress_layers = regress_layers
        self.output_dim = output_dim
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_path)
        self.wavlm = WavLMModel.from_pretrained(checkpoint_path)
        self.num_encoder_layers = self.wavlm.config.num_hidden_layers
        self.hidden_size = self.wavlm.config.hidden_size
        if self.output_dim is None:
            self.output_dim = self.hidden_size
        
        if self.layer is None:
            self.layer = self.wavlm.config.num_hidden_layers
        self.layer = min(self.layer, self.wavlm.config.num_hidden_layers)
        self.sample_rate = self.feature_extractor.sampling_rate
        self.hop_size = np.prod(self.wavlm.config.conv_stride)

        self.wavlm = self.wavlm.eval()

        # init model
        if init_model is not None:
            model_dict = torch.load(init_model, map_location="cpu")
            self.load_state_dict(model_dict)

    def train(self, mode=True):
        super().train(mode)
        self.wavlm.eval()

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
        assert x.dim() == 2, f"wav input for wavlm encoder should with dim==2"
        if x_lens is None:
            x_lens = torch.tensor([x.shape[1]] * x.shape[0], device=x.device).int()
        # resample
        if input_sample_rate is not None and input_sample_rate != self.sample_rate:
            x = resample(x, orig_freq=input_sample_rate, new_freq=self.sample_rate)
            x_lens = (x_lens / input_sample_rate * self.sample_rate).ceil().int()
            x_lens = torch.clamp(x_lens, max=x.shape[1])
        # normalize, Wav2VecFeatureExtractor conducted on cpu, would be very slow
        if self.feature_extractor.do_normalize:
           x = self.zero_unit_norm(x, x_lens)
        padding_mask = get_mask_from_lengths(x_lens, max_len=x.shape[1])
        feat = self.wavlm(
            x,
            attention_mask = padding_mask,
            output_hidden_states = True
        )["hidden_states"]
        # feat_lens = x_lens // self.hop_size - 1  # valid conv
        feat_lens = self.wavlm._get_feat_extract_output_lengths(x_lens)
        feat_lens = torch.clamp(feat_lens, max=feat[0].shape[1])
        return feat, feat_lens

    @eval_decorator
    @torch.no_grad()
    def forward(
        self,
        wav_input,
        wav_lens = None,
        input_sample_rate = None,
        **kwargs
    ):

        all_hidden_feats, feat_lens = self.get_hidden_feat(
                wav_input, x_lens=wav_lens, input_sample_rate=input_sample_rate)
        if self.use_regressor:
            # average
            regressed_feats = torch.mean(torch.stack([all_hidden_feats[x] for x in self.regress_layers], dim=0), dim=0)
        else:
            regressed_feats = all_hidden_feats[self.layer]

        res = {
            "all_hidden_feats": all_hidden_feats,
            "hidden_feat_lens": feat_lens,
            "regressed_feats": regressed_feats,
        }
        return res
