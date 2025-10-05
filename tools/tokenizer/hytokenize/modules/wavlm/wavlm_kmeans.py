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

from ..utils import exists, get_mask_from_lengths, eval_decorator
from ..kmeans import ApplyKmeans

import logging
import numpy as np
logging.root.setLevel(logging.ERROR)


class WavLMWithKmeans(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_path,
        kmeans_path,
        layer = None,
        seq_len_multiple_of = None,
        token_pad_id = -1
    ):
        super().__init__()
        model_path = Path(checkpoint_path)
        kmeans_path = Path(kmeans_path)
        assert model_path.exists(), f'path {checkpoint_path} does not exist'
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'

        self.layer = layer
        self.seq_len_multiple_of = seq_len_multiple_of
        self.token_pad_id = token_pad_id

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_path)
        self.wavlm = WavLMModel.from_pretrained(checkpoint_path)
        self.kmeans = ApplyKmeans(kmeans_path)
        if self.layer is None:
            self.layer = self.wavlm.config.num_hidden_layers
        self.layer = min(self.layer, self.wavlm.config.num_hidden_layers)
        self.sample_rate = self.feature_extractor.sampling_rate
        self.hop_size = np.prod(self.wavlm.config.conv_stride)

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @property
    def cluster_centers_(self):
        return rearrange(self.kmeans.C, 'd c -> c d')

    def zero_unit_norm(self, wav_input, wav_lens, eps=1e-7):
        mask = get_mask_from_lengths(wav_lens, max_len=wav_input.shape[1]).to(wav_input.dtype)
        wav_input = wav_input * mask
        mean = wav_input.sum(dim=-1, keepdims=True) / wav_lens.unsqueeze(1)
        sqr_mean = (wav_input * wav_input).sum(dim=-1, keepdims=True) / wav_lens.unsqueeze(1)
        var = sqr_mean - mean * mean
        normed = (wav_input - mean) / (var + eps).sqrt()
        normed = normed * mask
        return normed

    @torch.no_grad()
    def forward(
        self,
        wav_input,
        wav_lens = None,
        input_sample_rate = None,
        **kwargs
    ):
        self.wavlm.eval()
        assert wav_input.dim() == 2
        if not exists(wav_lens):
            wav_lens = torch.tensor([wav_input.shape[1]] * wav_input.shape[0], device=wav_input.device).int()
        if exists(input_sample_rate) and input_sample_rate != self.sample_rate:
            wav_input = resample(wav_input, input_sample_rate, self.sample_rate)
            wav_lens = (wav_lens / input_sample_rate * self.sample_rate).ceil().int()
            wav_lens = torch.clamp(wav_lens, max=wav_input.shape[1])

        if exists(self.seq_len_multiple_of):
            d = wav_input.size(-1)
            d = d // self.seq_len_multiple_of * self.seq_len_multiple_of
            wav_input = wav_input[..., :d]
            wav_lens = wav_lens // self.seq_len_multiple_of * self.seq_len_multiple_of
            wav_lens = torch.clamp(wav_lens, max=d)
        # feature extractor conducted on cpu, may be slow
        # input_values = self.feature_extractor(wav_input, return_tensors='pt', sampling_rate=self.sample_rate).input_values
        # wav_input = input_values.squeeze(0).to(wav_input.device)
        if self.feature_extractor.do_normalize:
           wav_input = self.zero_unit_norm(wav_input, wav_lens)
        padding_mask = get_mask_from_lengths(wav_lens, max_len=wav_input.shape[1])
        res = self.wavlm(
            wav_input,
            attention_mask = padding_mask,
            output_hidden_states = True
        )
        feats = res['hidden_states'][self.layer]
        token_lens = wav_lens // self.hop_size - 1  # valid conv
        token_lens = torch.clamp(token_lens, max=feats.shape[1])
        b, n, dim = feats.shape
        feats = rearrange(feats, 'b n d -> (b n) d')
        tokens = self.kmeans(feats)
        tokens = rearrange(tokens, '(b n) -> b n', b=b)
        token_pad_mask = ~get_mask_from_lengths(token_lens, max_len=tokens.shape[1])
        tokens = tokens.masked_fill(token_pad_mask, self.token_pad_id)
        return tokens, token_lens


    @eval_decorator
    @torch.no_grad()
    def extract_token(
        self,
        wav_input,
        wav_lens = None,
        input_sample_rate = None,
        **kwargs
    ):
        return self.forward(
            wav_input,
            wav_lens = wav_lens,
            input_sample_rate = input_sample_rate,
            **kwargs
        )
