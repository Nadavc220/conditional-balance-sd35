# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
import einops
import numpy as np
import functools


T = torch.Tensor


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True,
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.
    shared_score_shift: float = 0.
    only_self_level: float = 0.


def expand_first(feat: T, scale=1.,) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    a = feat.clone()
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat[1] = (feat[1] - feat_mean[1]) / feat_std[1]
    feat[3] = (feat[3] - feat_mean[3]) / feat_std[3]
    feat[1] = feat[1] * feat_style_std[1] + feat_style_mean[1]
    feat[3] = feat[3] * feat_style_std[3] + feat_style_mean[3]
    assert torch.all(a[0] == feat[0]) and torch.all(a[2] == feat[2])
    return feat


class DefaultAttentionProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.processor = attention_processor.JointAttnProcessor2_0()

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)
    

class SharedDynamicJointAttentionProcessor(DefaultAttentionProcessor):

    def __init__(self, style_aligned_args: StyleAlignedArgs):
        super().__init__()
        self.share_attention = style_aligned_args.share_attention
        self.adain_queries = style_aligned_args.adain_queries
        self.adain_keys = style_aligned_args.adain_keys
        self.adain_values = style_aligned_args.adain_values
        self.full_attention_share = style_aligned_args.full_attention_share
        self.shared_score_scale = style_aligned_args.shared_score_scale
        self.shared_score_shift = style_aligned_args.shared_score_shift
    

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        hidden_states = self.shared_call(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        return hidden_states
    

    def update_args(self, adain_keys=None, adain_quaries=None, adain_values=None, share_attention=None):
        if adain_keys is not None:
            self.adain_keys = adain_keys
        if adain_quaries is not None:
            self.adain_queries = adain_quaries
        if adain_quaries is not None:
            self.adain_values = adain_values
        if share_attention is not None:
            self.share_attention = share_attention


    def shifted_scaled_dot_product_attention(self, attn: attention_processor.Attention, query: T, key: T, value: T) -> T:
        logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
        logits[:, :, :, query.shape[2]:] += self.shared_score_shift
        probs = logits.softmax(-1)
        return torch.einsum('bhqk,bhkd->bhqd', probs, value)

    def shared_call(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        self.share_attention = self.adain_keys and self.adain_queries
        if self.adain_queries:
            query = adain(query)
        if self.adain_keys:
            key = adain(key)
        if self.adain_values:
            value = adain(value)
        if self.share_attention:
            key = concat_first(key, -2, scale=self.shared_score_scale)
            value = concat_first(value, -2)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = nnf.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def _get_switch_vec(total_num_layers, level):
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool)
    to_flip = level > .5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers)
    vec = torch.arange(total_num_layers)
    vec = vec % (total_num_layers // num_switch)
    vec = vec == 0
    if to_flip:
        vec = ~vec
    return vec

from diffusers.models.attention_processor import JointAttnProcessor2_0
def init_attention_processors(pipeline, style_aligned_args: StyleAlignedArgs | None = None):
    atts = {}
    attn_procs = {}
    unet = pipeline.transformer
    number_of_self, number_of_cross = 0, 0
    num_self_layers = len([name for name in unet.attn_processors.keys() if 'attn1' in name])
    if style_aligned_args is None:
        only_self_vec = _get_switch_vec(num_self_layers, 1)
    else:
        only_self_vec = _get_switch_vec(num_self_layers, style_aligned_args.only_self_level)
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attention = 'attn' in name  # TODO: Check if needs some condition
        if is_self_attention:
            number_of_self += 1
            if False:
            # if style_aligned_args is None or only_self_vec[i // 2]:
                attn_procs[name] = DefaultAttentionProcessor()
            else:
                attn_procs[name] = SharedDynamicJointAttentionProcessor(style_aligned_args)
                atts[name] = attn_procs[name]
        else:
            number_of_cross += 1
            attn_procs[name] = DefaultAttentionProcessor()

    unet.set_attn_processor(attn_procs)
    return atts


def register_shared_norm(pipeline: StableDiffusionXLPipeline,
                         share_group_norm: bool = True,
                         share_layer_norm: bool = True, ):
    def register_norm_forward(norm_layer: nn.GroupNorm | nn.LayerNorm) -> nn.GroupNorm | nn.LayerNorm:
        if not hasattr(norm_layer, 'orig_forward'):
            setattr(norm_layer, 'orig_forward', norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_
        return norm_layer

    def get_norm_layers(pipeline_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]):
        if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
            norm_layers_['layer'].append(pipeline_)
        if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
            norm_layers_['group'].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {'group': [], 'layer': []}
    get_norm_layers(pipeline.transformer, norm_layers)
    return [register_norm_forward(layer) for layer in norm_layers['group']] + [register_norm_forward(layer) for layer in
                                                                               norm_layers['layer']]


class Handler:

    def __init__(self, pipeline, layer_stats: dict, num_style_layers: int):
        self.pipeline = pipeline
        self.norm_layers = []
        self.layers_stats = layer_stats
        self.num_style_layers = num_style_layers
        
        self.sorted_layers = {int(t): {'q': [], 'k': [], 'v': []} for t in self.layers_stats['stats']}
        for t in self.layers_stats['stats']:
            current_scores = self.layers_stats['stats'][int(t)]
            self.sorted_layers[t]['q'].append(np.argsort(current_scores['q'])[:self.num_style_layers])
            self.sorted_layers[t]['k'].append(np.argsort(current_scores['k'])[:self.num_style_layers])
            self.sorted_layers[t]['v'].append(np.argsort(current_scores['v'])[:self.num_style_layers])

    def register(self, style_aligned_args: StyleAlignedArgs):
        self.norm_layers = register_shared_norm(self.pipeline, style_aligned_args.share_group_norm,
                                                style_aligned_args.share_layer_norm)
        self.self_attention_layers = init_attention_processors(self.pipeline, style_aligned_args)

    def remove(self):
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers = []
        init_attention_processors(self.pipeline, None)


    def update_adain_layers(self, t):
        if len(self.sorted_layers[int(t)]['q']) > 0 or len(self.sorted_layers[int(t)]['k']) > 0 or len(self.sorted_layers[int(t)]['v']) > 0:
            layer_names = self.layers_stats['layer_names']
            cummulative_top_q = list(set(functools.reduce(lambda l1, l2: np.concatenate([l1, l2], axis=0), self.sorted_layers[int(t)]['q'])))
            cummulative_top_k = list(set(functools.reduce(lambda l1, l2: np.concatenate([l1, l2], axis=0), self.sorted_layers[int(t)]['k'])))
            cummulative_top_v = list(set(functools.reduce(lambda l1, l2: np.concatenate([l1, l2], axis=0), self.sorted_layers[int(t)]['v'])))
            q_layers = np.array(layer_names)[cummulative_top_q] if len(cummulative_top_q) > 0 else []
            k_layers = np.array(layer_names)[cummulative_top_k] if len(cummulative_top_k) > 0 else []
            v_layers = np.array(layer_names)[cummulative_top_v] if len(cummulative_top_v) > 0 else []

            attention_layers = self.self_attention_layers
            for layer_name in layer_names:
                adain_q, adain_k, adain_v = False, False, False
                if layer_name in q_layers:
                    adain_q = True
                if layer_name in k_layers:
                    adain_k = True
                if layer_name in v_layers:
                    adain_v = True
                share_attention = adain_q or adain_k or adain_v
                attention_layers[layer_name].update_args(adain_keys=adain_k, adain_quaries=adain_q, adain_values=adain_v, share_attention=share_attention)