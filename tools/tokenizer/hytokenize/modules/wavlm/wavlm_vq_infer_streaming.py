import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import resample
from einops import rearrange
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor
from .modeling_wavlm_streaming import WavLMModel as StreamWavLMModel

import numpy as np
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional, Tuple, List, Dict, Any, Union

from ..utils import exists, get_mask_from_lengths, eval_decorator
from ..modules import RegressLinear
from ..vector_quantize_pytorch import VectorQuantize
from modules.commons.layers import Linear, Conv1d
from modules.commons.mask import subsequent_chunk_mask


class WavLMWithVQ_StreamInfer(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_path,
        layer: Optional[int] = None,
        use_regressor: bool = False,
        regress_layers: Optional[List[int]] = None,
        vq_conf: Optional[Dict[str, Any]] = {},
        ssl_ds_rate: int = 1,
        token_pad_id = -1,
        encoder_trainable = False,
        init_model: Optional[str] = None,
        chunk_size = 16,  # each block contains `chunk_size` tokens
        conv_pos_causal = -1,
        feat_norm_mode = 0,  # 0 for full context, 1 for chunk context, 2 for specified context
        feat_norm_context = -1,
        **kwargs
    ):
        super().__init__()
        self.layer = layer
        self.use_regressor = use_regressor
        self.ssl_ds_rate = ssl_ds_rate
        self.token_pad_id = token_pad_id
        self.encoder_trainable = encoder_trainable

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_path)
        self.wavlm = StreamWavLMModel.from_pretrained(checkpoint_path)
        self.num_encoder_layers = self.wavlm.config.num_hidden_layers
        self.hidden_size = self.wavlm.config.hidden_size
        if self.layer is None:
            self.layer = self.wavlm.config.num_hidden_layers
        self.layer = min(self.layer, self.wavlm.config.num_hidden_layers)
        self.sample_rate = self.feature_extractor.sampling_rate
        self.hop_size = np.prod(self.wavlm.config.conv_stride)
        # if not self.encoder_trainable:
        #     self.wavlm = self.wavlm.eval()
        # samples chunk infos
        self.chunk_size = chunk_size
        self.conv_pos_causal = conv_pos_causal
        self.block_size = chunk_size * self.hop_size + self.wavlm.feature_extractor.right_context
        self.single_window_size = self.hop_size + self.wavlm.feature_extractor.right_context
        self.block_hop_size = chunk_size * self.hop_size
        # norm context mode
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_context = feat_norm_context
        self.feat_norm_min_std = kwargs.get("feat_norm_min_std", 0.2)

        # regressor
        if self.use_regressor:
            self.regress_linear = RegressLinear(
                num_layers = self.num_encoder_layers,
                hidden_size = self.hidden_size,
                regress_layers = regress_layers,
                hidden_contain_input = True
            )
        # VQ
        if vq_conf is None:
            vq_conf = {"dim": 256, "codebook_size": 1024}
        vq_conf["heads"] = 1  # set VQ to has 1 code by default
        vq_conf["separate_codebook_per_head"] = False
        vq_conf["codebook_dim"] = None
        self.vq = VectorQuantize(**vq_conf)
        # projection
        self.ssl_proj = Conv1d(self.hidden_size, self.vq.dim, ssl_ds_rate, ssl_ds_rate)
        # init model
        if init_model is not None:
            model_dict = torch.load(init_model, map_location="cpu")
            self.load_state_dict(model_dict)

        # # Register gradient scaling hook 
        # if self.encoder_trainable:
        #     self.grad_scale_for_wavlm = kwargs.get('grad_scale_for_wavlm', 10.0)
        #     self.register_gradient_hook()

    @property
    def groups(self):
        return self.vq.heads

    @property
    def codebook_size(self):
        return self.vq.codebook_size

    def train(self, mode=True):
        super().train(mode)
        if not self.encoder_trainable:
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

    def zero_unit_norm_context(self, wav_input, wav_lens, context_size, eps=1e-7):
        context_lens = torch.clamp(wav_lens, max=context_size)
        context_wavs = wav_input[:, :context_size].clone()
        context_mask = get_mask_from_lengths(context_lens, max_len=context_size).to(wav_input.dtype)
        context_wavs = context_wavs * context_mask
        mean = context_wavs.sum(dim=-1, keepdims=True) / context_lens.unsqueeze(1)
        sqr_mean = (context_wavs * context_wavs).sum(dim=-1, keepdims=True) / context_lens.unsqueeze(1)
        var = sqr_mean - mean * mean
        mask = get_mask_from_lengths(wav_lens, max_len=wav_input.shape[1]).to(wav_input.dtype)
        wav_input = wav_input * mask
        normed = (wav_input - mean) / torch.clamp((var + eps).sqrt(), min=self.feat_norm_min_std)
        normed = normed * mask
        return normed

    @eval_decorator
    @torch.no_grad()
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
            if self.feat_norm_mode == 0 or (self.feat_norm_mode == 2 and self.feat_norm_context <= 0):
                x = self.zero_unit_norm(x, x_lens)
            elif self.feat_norm_mode == 2:
                x = self.zero_unit_norm_context(x, x_lens, self.feat_norm_context)
        padding_mask = get_mask_from_lengths(x_lens, max_len=x.shape[1])
        # forward
        feat = [None for _ in range(self.num_encoder_layers + 1)]
        conv_pos_cache = None
        past_key_values = None
        feat_lens = torch.zeros(x.shape[0], device=x.device).long()
        for start in range(0, x.shape[1], self.block_hop_size):
            block = x[:, start:start+self.block_size]
            if block.shape[1] < self.single_window_size:
                break
            attn_mask = padding_mask[:, start:start+self.block_size]
            block_lens = attn_mask.sum(dim=1).int()
            if self.feature_extractor.do_normalize and self.feat_norm_mode == 1:
                block = self.zero_unit_norm(block, block_lens)
            block_feat_lens = self.wavlm._get_feat_extract_output_lengths(block_lens)
            block_feat_lens = torch.clamp(block_feat_lens, min=0)
            stream_out = self.wavlm(
                block,
                attention_mask = attn_mask,
                conv_pos_cache = conv_pos_cache,
                past_key_values = past_key_values,
                chunk_size = self.chunk_size if self.conv_pos_causal < 0 else self.conv_pos_causal,
                output_hidden_states = True
            )
            conv_pos_cache = stream_out.conv_pos_cache
            past_key_values = stream_out.past_key_values
            for i in range(len(feat)):
                if feat[i] is None:
                    feat[i] = stream_out.hidden_states[i]
                else:
                    feat[i] = torch.cat([feat[i], stream_out.hidden_states[i]], dim=1)
            feat_lens = feat_lens + block_feat_lens
        feat_lens = torch.clamp(feat_lens, max=feat[0].shape[1])
        return feat, feat_lens

    def forward(
        self,
        wav_input,
        wav_lens = None,
        input_sample_rate = None,
        **kwargs
    ):
        encoder_fw_context = nullcontext if self.encoder_trainable else torch.no_grad
        with encoder_fw_context():
            all_hidden_feats, feat_lens = self.get_hidden_feat(
                    wav_input, x_lens=wav_lens, input_sample_rate=input_sample_rate)
        if self.use_regressor:
            regressed_feats = self.regress_linear(all_hidden_feats)
        else:
            regressed_feats = all_hidden_feats[self.layer]
        hidden_feats = self.ssl_proj(regressed_feats.transpose(1, 2))
        hidden_feats = rearrange(hidden_feats, 'b d n -> b n d')
        token_lens = feat_lens // self.ssl_ds_rate
        mask = get_mask_from_lengths(token_lens, max_len=hidden_feats.shape[1])
        quantized, indices, vq_loss, loss_breakdown = self.vq(
                hidden_feats, mask=mask, return_loss_breakdown=True)
        tokens = indices.masked_fill(~mask, self.token_pad_id)
        vq_loss_items = {}
        for k, v in loss_breakdown._asdict().items():
            if v.item() > 0.0:
                vq_loss_items[k] = v.item()
        res = {
            "quantized": quantized,
            "tokens": tokens,
            "token_lens": token_lens,
            "vq_loss": vq_loss,
            "vq_loss_items": vq_loss_items,
            "all_hidden_feats": all_hidden_feats,
            "hidden_feat_lens": feat_lens,
            "regressed_feats": regressed_feats,
            "before_quantize": hidden_feats
        }
        return res

    @eval_decorator
    @torch.no_grad()
    def stream_chunk_forward(
        self,
        chunk_input: torch.Tensor,
        chunk_lens: torch.Tensor,
        input_sample_rate: int,
        conv_pos_cache: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None
    ):
        # input wav should be resampled and normed here
        assert chunk_input.dim() == 2, f"wav input for wavlm encoder should with dim==2"
        assert input_sample_rate == self.sample_rate, f"given wav sample_rate should be {self.sample_rate}, got {input_sample_rate}"
        padding_mask = get_mask_from_lengths(chunk_lens, max_len=chunk_input.shape[1])
        chunk_token_lens = self.wavlm._get_feat_extract_output_lengths(chunk_lens)
        chunk_token_lens = torch.clamp(chunk_token_lens, min=0)
        chunk_out = self.wavlm(
            chunk_input,
            attention_mask = padding_mask,
            conv_pos_cache = conv_pos_cache,
            past_key_values = past_key_values,
            chunk_size = self.chunk_size if self.conv_pos_causal < 0 else self.conv_pos_causal,
            output_hidden_states = True
        )
        conv_pos_cache = chunk_out.conv_pos_cache
        past_key_values = chunk_out.past_key_values
        all_hidden_states = chunk_out.hidden_states
        # tokenize
        if self.use_regressor:
            regressed_feats = self.regress_linear(all_hidden_states)
        else:
            regressed_feats = all_hidden_states[self.layer]
        hidden_feats = self.ssl_proj(regressed_feats.transpose(1, 2))
        hidden_feats = rearrange(hidden_feats, 'b d n -> b n d')
        chunk_token_lens = chunk_token_lens // self.ssl_ds_rate
        mask = get_mask_from_lengths(chunk_token_lens, max_len=hidden_feats.shape[1])
        quantized, indices, vq_loss, loss_breakdown = self.vq(
                hidden_feats, mask=mask, return_loss_breakdown=True)
        tokens = indices.masked_fill(~mask, self.token_pad_id)
        res = {
            "quantized": quantized,
            "tokens": tokens,
            "token_lens": chunk_token_lens,
            "conv_pos_cache": conv_pos_cache,
            "past_key_values": past_key_values
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
        token, token_lens = res["tokens"], res["token_lens"]
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
        feat_lens = res["token_lens"]
        return before_quantize, feat_lens
