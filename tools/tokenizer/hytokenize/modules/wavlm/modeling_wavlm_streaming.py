import math
import random
import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import einsum
from einops import rearrange, repeat, reduce

from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_peft_available,
    logging,
)
from transformers.models.wavlm.configuration_wavlm import WavLMConfig


class StreamWavLMConfig(WavLMConfig):
    def __init__(self, use_cache=False, **kwargs):
        super().__init__(**kwargs)
        self.use_cache = use_cache

@dataclass
class StreamWavLMOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    conv_pos_cache: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None


@dataclass
class StreamWavLMOutputWithFeat(StreamWavLMOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    conv_pos_cache: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None
    extract_features: torch.FloatTensor = None

# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask

"""
Since conv layer in encoder use `valid` conv format, not `pad` conv format
and the kernel may not be odd, here we do not add cache for conv layer,
otherwise use pre-computed chunk_size and hop_size for conv encoder
"""
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->WavLM
class WavLMNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->WavLM
class WavLMLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->WavLM
class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

'''
add caches for streaming
'''
class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.right_context = config.num_conv_pos_embeddings // 2
        self.conv_groups = config.num_conv_pos_embedding_groups
        self.conv_padding = config.num_conv_pos_embeddings // 2

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = WavLMSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]
        self.use_cache = config.use_cache
        self.cache_size = config.num_conv_pos_embeddings // 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        chunk_size = -1,
        caches: Optional[torch.Tensor] = None
    ):
        offset = 0
        if self.use_cache and caches is not None:
            used_caches = caches[:, -self.cache_size:, :]
            offset = used_caches.shape[1]
            hidden_states = torch.cat([used_caches, hidden_states], dim=1)
        new_caches = None
        if self.use_cache:
            new_caches = hidden_states[:, -self.cache_size:, :]
        hidden_states = hidden_states.transpose(1, 2)
        if chunk_size >= 0 and chunk_size < self.right_context:
            conv_weight = self.conv.weight.clone()
            conv_weight[..., (chunk_size - self.right_context):] = 0.0
            hidden_states = F.conv1d(hidden_states, conv_weight, bias=self.conv.bias,
                                     padding=self.conv_padding, groups=self.conv_groups)
        else:
            hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = hidden_states[..., offset:]
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states, new_caches

# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->WavLM
class WavLMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class WavLMFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [WavLMGroupNormConvLayer(config, layer_id=0)] + [
                WavLMNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [WavLMLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True
        # receptive_field & hop_size
        self.receptive_field = self.get_receptive_field(config)
        self.hop_size = self.get_hop_size(config)
        self.right_context = self.receptive_field - self.hop_size
        self.use_cache = config.use_cache

    def get_hop_size(self, config):
        hop_size = 1
        for s in config.conv_stride:
            hop_size = hop_size * s
        return hop_size

    def get_receptive_field(self, config):
        o = 1
        kernels, strides = config.conv_kernel, config.conv_stride
        kernels = kernels[::-1]
        strides = strides[::-1]
        for i, (k, s) in enumerate(zip(kernels, strides)):
            o = (o - 1) * s + k
        return o

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
    ):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


class WavLMFeatureExtractor(WavLMFeatureEncoder):
    def __init__(self, config):
        super().__init__(config)
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->WavLM
class WavLMFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class WavLMAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_buckets: int = 320,
        max_distance: int = 800,
        has_relative_position_bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # [batch_size * num_heads, time_ste, dimension]
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        q_len = q.shape[-2]
        kv_seq_len = k.shape[-2]
        prefix_len = 0
        if past_key_values is not None:
            kv_seq_len += past_key_values[0].shape[-2]
            prefix_len = past_key_values[0].shape[-2]
            cache_k = past_key_values[0]
            cache_v = past_key_values[1]
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
        new_caches = (k, v) if use_cache else None
        # attention_mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                assert attention_mask.shape == (bsz, kv_seq_len), f"attention_mask shape should be {(bsz, kv_seq_len)}"
                attention_mask = repeat(attention_mask, 'b k -> b h q k', h=self.num_heads, q=q_len)
            elif attention_mask.dim() == 3:
                assert attention_mask.shape == (bsz, q_len, kv_seq_len), f"attention_mask shape should be {(bsz, q_len, kv_seq_len)}"
                attention_mask = repeat(attention_mask, 'b q k -> b h q k', h=self.num_heads)
            else:
                raise ValueError(f"attention_mask shape should be {(bsz, kv_seq_len)} or {(bsz, q_len, kv_seq_len)}")
        # compute_bias for first pass of attnetion layer
        if position_bias is None:
            position_bias = self.compute_bias(q_len, kv_seq_len, prefix_len=prefix_len)
            position_bias = repeat(position_bias, 'h q k -> b h q k', b=bsz)
        # Conduct On ori hidden states ?
        # Compute relative position bias:
        # 1) get reshape hidden_states
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)

        # 2) project hidden states
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)

        # 3) compute gate for position bias from projected hidden states
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        # 4) apply gate to position bias to compute gated position_bias
        # [bsz * num_heads, q_len, kv_len]
        gated_position_bias = gate_output.view(bsz, self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((bsz, self.num_heads, q_len, -1))

        # attention bias
        if attention_bias is not None:
            if attention_bias.shape != (bsz, q_len, kv_seq_len):
                raise ValueError(f"Expected shape of attention bias is {(bsz, q_len, kv_seq_len)}, "
                                  "but got {attention_bias.shape}")
            attention_bias = repeat(attention_bias, 'b q k -> b h q k', h=self.num_heads)
            gated_position_bias = gated_position_bias + attention_bias
        if attention_mask is not None:
            bias_for_mask = torch.zeros_like(attention_mask, dtype=q.dtype).masked_fill(~attention_mask, float("-inf"))
            gated_position_bias = gated_position_bias + bias_for_mask
        # attention forward
        scale = math.sqrt(1.0 / self.head_dim)
        if output_attentions:  # not use torch sdpa
            scores = einsum('b h i d, b h j d -> b h i j', q, k) * scale
            scores = scores + gated_position_bias
            attn_weight = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0.0:
                attn_weight = F.dropout(attn_weight, p=self.dropout)
            attn_output = einsum('b h i j, b h j d -> b h i d', attn_weight, v)
            attn_output = rearrange(attn_output, 'b h i d -> b i (h d)')
            attn_output = self.out_proj(attn_output)
        else:  # use torch sdpa
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = gated_position_bias,
                dropout_p = self.dropout if self.training else 0.0,
                scale = scale
            )
            attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
            attn_output = self.out_proj(attn_output)
            attn_weight = None
        return attn_output, attn_weight, position_bias, new_caches

    def compute_bias(
        self,
        query_length: int,
        key_length: int,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        context_position = torch.arange(prefix_len, query_length + prefix_len, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)
        # [num_heads, q_len, k_len]
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        num_buckets = self.num_buckets // 2

        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        return relative_buckets


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->WavLM
class WavLMFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

class WavLMEncoderLayer(nn.Module):
    def __init__(self, config: StreamWavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_cache = config.use_cache

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        attention_bias = None,
        position_bias = None,
        past_key_values = None,
        output_attentions = False,
    ):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias, new_caches = self.attention(
            hidden_states,
            attention_mask = attention_mask,
            attention_bias = attention_bias,
            position_bias = position_bias,
            past_key_values = past_key_values,
            output_attetions = output_attentions,
            use_cache = self.use_cache
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, position_bias, attn_weights, new_caches)
        return outputs


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: StreamWavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_cache = config.use_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias, new_caches = self.attention(
            hidden_states,
            attention_mask = attention_mask,
            attention_bias = attention_bias,
            position_bias = position_bias,
            past_key_values = past_key_values,
            output_attentions = output_attentions,
            use_cache = self.use_cache
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states, position_bias, attn_weights, new_caches)
        return outputs


class WavLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.use_cache = config.use_cache

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        attention_bias=None,
        conv_pos_cache=None,
        past_key_values=None,
        chunk_size=-1,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            tmp_mask = attention_mask
            if attention_mask.dim() == 3:
                tmp_mask = attention_mask.any(dim=-1)
            hidden_states[~tmp_mask] = 0.0
            if past_key_values is not None:
                past_len = past_key_values[0][0].shape[-2]
                attention_mask = F.pad(attention_mask.float(), (past_len, 0), value=1.0).bool()
        # with cache 
        position_embeddings, new_conv_pos_cache = self.pos_conv_embed(
                hidden_states, chunk_size=chunk_size, caches=conv_pos_cache)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None

        new_caches = [] if self.use_cache else None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # add layerdrop
            dropout_probability = torch.rand([])

            layer_cache = past_key_values[i] if past_key_values is not None else None
            # skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            skip_the_layer = False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        attention_bias,
                        position_bias,
                        layer_cache,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        attention_bias=attention_bias,
                        position_bias=position_bias,
                        past_key_values=layer_cache,
                        output_attentions=output_attentions
                    )
                hidden_states, position_bias = layer_outputs[:2]
            if skip_the_layer:
                layer_outputs = (None, None, None, None)

            if self.use_cache:
                new_caches.append(layer_outputs[-1])
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2], )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        if not return_dict:
            return hidden_states, all_hidden_states, all_self_attentions, new_conv_pos_cache, new_caches
        return StreamWavLMOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states,
            attentions = all_self_attentions,
            conv_pos_cache = new_conv_pos_cache,
            past_key_values = new_caches
        )


class WavLMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [
                WavLMEncoderLayerStableLayerNorm(config, has_relative_position_bias=(i == 0))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.use_cache = config.use_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        conv_pos_cache: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        chunk_size: int = -1,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            tmp_mask = attention_mask
            if attention_mask.dim() == 3:
                tmp_mask = attention_mask.any(dim=-1)
            hidden_states[~tmp_mask] = 0.0
            if past_key_values is not None:
                past_len = past_key_values[0][0].shape[-2]
                attention_mask = F.pad(attention_mask.float(), (past_len, 0), value=1.0).bool()
        # with cache 
        position_embeddings, new_conv_pos_cache = self.pos_conv_embed(
                hidden_states, chunk_size=chunk_size, caches=conv_pos_cache)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None

        new_caches = [] if self.use_cache else None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )
            # add layerdrop
            dropout_probability = torch.rand([])

            layer_cache = past_key_values[i] if past_key_values is not None else None
            # skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            skip_the_layer = False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        attention_bias,
                        position_bias,
                        layer_cache,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        attention_bias=attention_bias,
                        position_bias=position_bias,
                        past_key_values=layer_cache,
                        output_attentions=output_attentions
                    )
                hidden_states, position_bias = layer_outputs[:2]
            if skip_the_layer:
                layer_outputs = (None, None, None, None)
            if self.use_cache:
                new_caches.append(layer_outputs[-1])
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2], )
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        if not return_dict:
            return hidden_states, all_hidden_states, all_self_attentions, new_conv_pos_cache, new_caches
        return StreamWavLMOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states,
            attentions = all_self_attentions,
            conv_pos_cache = new_conv_pos_cache,
            past_key_values = new_caches
        )

class WavLMPreTrainedModel(PreTrainedModel):
    config_class = StreamWavLMConfig
    base_model_prefix = "wavlm"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, WavLMPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, WavLMFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)
        # TODO: avoid length < 1 to get cuda device error
        output_lengths = torch.clamp(output_lengths, min=1)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

WAVLM_START_DOCSTRING = r"""
    WavLM was proposed in [WavLM: Unified Speech Representation Learning with Labeled and Unlabeled
    Data](https://arxiv.org/abs/2110.13900) by Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo
    Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian,
    Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`WavLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


WAVLM_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.

            </Tip>

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare WavLM Model transformer outputting raw hidden-states without any specific head on top.",
    WAVLM_START_DOCSTRING,
)
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model with Wav2Vec2->WavLM, wav2vec2->wavlm, WAV_2_VEC_2->WAVLM, WavLMBaseModelOutput->Wav2Vec2BaseModelOutput
class WavLMModel(WavLMPreTrainedModel):
    def __init__(self, config: StreamWavLMConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = WavLMFeatureEncoder(config)
        self.feature_projection = WavLMFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = WavLMEncoderStableLayerNorm(config)
        else:
            self.encoder = WavLMEncoder(config)
        # disable adapter
        self.adapter = None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,  # [batch_size, src_len]
        attention_bias: Optional[torch.Tensor] = None,  # [batch_size, tgt_len, src_len]
        conv_pos_cache: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        right_values: Optional[torch.Tensor] = None,
        chunk_size: int = -1,  # <=0: full context; >0: chunk mode
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, StreamWavLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = self.config.use_cache

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask = attention_mask,
            attention_bias = attention_bias,
            conv_pos_cache = conv_pos_cache,
            past_key_values = past_key_values,
            chunk_size = chunk_size,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = True
        )
        hidden_states = encoder_outputs.last_hidden_state
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        outputs = StreamWavLMOutputWithFeat(
            last_hidden_state = hidden_states,
            hidden_states = encoder_outputs.hidden_states,
            attentions = encoder_outputs.attentions,
            conv_pos_cache = encoder_outputs.conv_pos_cache,
            past_key_values = encoder_outputs.past_key_values,
            extract_features = extract_features
        )
        return outputs
