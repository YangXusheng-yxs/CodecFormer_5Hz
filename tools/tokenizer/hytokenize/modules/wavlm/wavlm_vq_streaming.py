import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import resample
from einops import rearrange
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor
from .modeling_wavlm import WavLMModel

import numpy as np
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional, Tuple, List, Dict, Any, Union

from ..utils import exists, get_mask_from_lengths, eval_decorator
from ..modules import RegressLinear
from ..vector_quantize_pytorch import VectorQuantize
from modules.commons.layers import Linear, Conv1d
from modules.commons.mask import subsequent_chunk_mask


def add_optional_chunk_mask(max_len, masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            max_num_right_chunks: int = 0,
                            max_dynamic_chunk_size: int = 25):
    """ Apply optional mask for encoder.

    Args:
        max_len (int): max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        max_num_right_chunks: max num right chunks allowed to attend when >0,
            right_chunks will be sampled in [1, max_num_right_chunks] when use dynamic,
            otherwise right_chunks will be set as max_num_right_chunks

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
            num_right_chunks = 0
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
            num_right_chunks = torch.randint(0, 1+max_num_right_chunks, (1,)).item()
        else:
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            # if chunk_size > max_len // 2:
            #     chunk_size = max_len
            # else:
            #     chunk_size = chunk_size % 25 + 1
            chunk_size = chunk_size % max_dynamic_chunk_size + 1
            if use_dynamic_left_chunk:
                max_left_chunks = (max_len - 1) // chunk_size
                num_left_chunks = torch.randint(0, max_left_chunks,
                                                (1, )).item()
            num_right_chunks = 0
        chunk_masks = subsequent_chunk_mask(max_len, chunk_size, num_left_chunks,
                                            masks.device, num_right_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(max_len, static_chunk_size, num_left_chunks,
                                            masks.device, max_num_right_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


class WavLMWithVQ_streaming(nn.Module):
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
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        chunk_size: int = 0,
        num_left_chunks: int = -1,
        num_right_chunks: int = 0,
        max_dynamic_chunk_size: int = 25,
        conv_pos_causal: int = -1,
        feat_norm_context: int = -1,
        **kwargs
    ):
        super().__init__()
        self.layer = layer
        self.use_regressor = use_regressor
        self.ssl_ds_rate = ssl_ds_rate
        self.token_pad_id = token_pad_id
        self.encoder_trainable = encoder_trainable

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_path)
        self.wavlm = WavLMModel.from_pretrained(checkpoint_path)
        self.num_encoder_layers = self.wavlm.config.num_hidden_layers
        self.hidden_size = self.wavlm.config.hidden_size
        if self.layer is None:
            self.layer = self.wavlm.config.num_hidden_layers
        self.layer = min(self.layer, self.wavlm.config.num_hidden_layers)
        self.sample_rate = self.feature_extractor.sampling_rate
        self.hop_size = np.prod(self.wavlm.config.conv_stride)
        if not self.encoder_trainable:
            self.wavlm = self.wavlm.eval()
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
        # chunk streaming relative
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.chunk_size = chunk_size
        self.num_left_chunks = num_left_chunks
        self.num_right_chunks = num_right_chunks
        self.max_dynamic_chunk_size = max_dynamic_chunk_size
        self.conv_pos_causal = conv_pos_causal
        self.feat_norm_context = feat_norm_context
        self.feat_norm_min_std = kwargs.get("feat_norm_min_std", 0.2)

        # init model
        if init_model is not None:
            model_dict = torch.load(init_model, map_location="cpu")
            self.load_state_dict(model_dict)

        # Register gradient scaling hook 
        if self.encoder_trainable:
            self.grad_scale_for_wavlm = kwargs.get('grad_scale_for_wavlm', 10.0)
            self.register_gradient_hook()

    def register_gradient_hook(self):
        def scale_gradients(grad):
            return grad / self.grad_scale_for_wavlm

        for param in self.wavlm.parameters():
            param.register_hook(scale_gradients)

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
            if self.feat_norm_context <= 0:
                x = self.zero_unit_norm(x, x_lens)
            else:
                x = self.zero_unit_norm_context(x, x_lens, self.feat_norm_context)
        padding_mask = get_mask_from_lengths(x_lens, max_len=x.shape[1])
        # get attention bias
        feat_lens = self.wavlm._get_feat_extract_output_lengths(x_lens)
        max_step = self.wavlm._get_feat_extract_output_lengths(x.shape[1])
        feat_attn_mask = get_mask_from_lengths(feat_lens, max_len=max_step).unsqueeze(1)
        # attention_bias
        chunk_mask = add_optional_chunk_mask(max_step, feat_attn_mask,
                self.use_dynamic_chunk, self.use_dynamic_left_chunk,
                self.chunk_size, self.chunk_size, self.num_left_chunks,
                self.num_right_chunks, self.max_dynamic_chunk_size)
        attention_bias = torch.zeros_like(chunk_mask, dtype=x.dtype).masked_fill(~chunk_mask, float("-inf"))
        # forward
        chunk_size = self.chunk_size if self.conv_pos_causal < 0 else self.conv_pos_causal
        feat = self.wavlm(
            x,
            attention_mask = padding_mask,
            attention_bias = attention_bias,
            chunk_size = chunk_size,
            output_hidden_states = True
        )["hidden_states"]
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
            if v.item() != 0.0:
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
