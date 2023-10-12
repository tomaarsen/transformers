from abc import ABC, abstractmethod
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Dict, List, Optional, Tuple, TypeVar

import torch


=======
from typing import Dict, List, Tuple, TypeVar
import torch

>>>>>>> 523380cb6 (Draft version of new KV Caching)
=======
from typing import Dict, List, Optional, Tuple, TypeVar

import torch


>>>>>>> 1129513b3 (Address numerous PR suggestions)
T = TypeVar("T")


class Cache(ABC):
    def __init__(self) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
        self.key_cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.value_cache: Dict[int, Tuple[torch.Tensor]] = {}

    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        return (
            tuple(self.key_cache[layer_idx] for layer_idx in range(len(self.key_cache))),
            tuple(self.value_cache[layer_idx] for layer_idx in range(len(self.value_cache))),
        )

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[List[torch.FloatTensor]]) -> "DynamicCache":
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
            cache.update(key_states, value_states, layer_idx)
        return cache


class DynamicCache(Cache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(
    key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.IntTensor] = None
) -> torch.Tensor:
    if position_ids:
        cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    rotated_key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return rotated_key_states
=======
        self.cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.layer_idx = 0
=======
        self.key_cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.value_cache: Dict[int, Tuple[torch.Tensor]] = {}
>>>>>>> 1129513b3 (Address numerous PR suggestions)

    @abstractmethod
    def update(self, key_states, value_states, layer_idx: int) -> None:
        pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        return (
            tuple(self.key_cache[layer_idx] for layer_idx in range(len(self.key_cache))),
            tuple(self.value_cache[layer_idx] for layer_idx in range(len(self.value_cache))),
        )

    @classmethod
<<<<<<< HEAD
    def from_past_key_values(cls, past_key_values: List[torch.FloatTensor]) -> "DynamicCache":
        raise NotImplementedError()
>>>>>>> 523380cb6 (Draft version of new KV Caching)
=======
    def from_past_key_values(cls, past_key_values: Optional[List[torch.FloatTensor]]) -> "DynamicCache":
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
            cache.update(key_states, value_states, layer_idx)
        return cache

    @classmethod
    def from_past_key_value(cls, past_key_value: Optional[torch.FloatTensor]) -> "DynamicCache":
        if past_key_value is None:
            return cls()
        cache = cls()
        cache.update(past_key_value[0], past_key_value[1], 0)
        return cache


class DynamicCache(Cache):
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
>>>>>>> 1129513b3 (Address numerous PR suggestions)


class SinkCache(Cache):
    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        super().__init__()
<<<<<<< HEAD
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_cache = {}

    def get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        return min(super().get_seq_length(layer_idx), self.window_length - 1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [bsz, num_heads, seq_len, head_dim]
        if layer_idx not in self.key_cache:
            # Empty cache
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            rotated_keys = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]
            rerotation_cos, rerotation_sin = self.get_rerotation_cos_sin(key_states, cos, sin)
            rerotated_keys = apply_rotary_pos_emb_single(rotated_keys, rerotation_cos, rerotation_sin)

            # Concatenate sink tokens, shifted & rotated tokens, and new tokens
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx][:, :, : self.num_sink_tokens], rerotated_keys, key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [
                    self.value_cache[layer_idx][:, :, : self.num_sink_tokens],
                    self.value_cache[layer_idx][
                        :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
                    ],
                    value_states,
                ],
                dim=-2,
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
=======
        self.is_prefill = False
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.index = torch.arange(num_sink_tokens, window_length)

    def update_pre_rotation(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        # idx is either 0 for key, 1 for values
        if layer_idx not in self.key_cache:
            # first in
            sink_keys = key_states[: self.num_sink_tokens]
            sink_values = value_states[: self.num_sink_tokens]

            cached_keys = torch.cat([sink_keys, key_states[:, -self.window_length :]], dim=-1)
            cached_values = torch.cat([sink_values, value_states[:, -self.window_length :]], dim=-1)

            self.key_cache[layer_idx] = torch.cat([cached_keys[None, :], cached_values[None, :]], dim=0)
        elif key_states.shape[1] < self.index.shape[-1] + self.num_sink_tokens:
            # auto-regressive
            key_len = key_states.shape[1]

            # roll cache to the left
            self.key_cache[layer_idx]._index_copy(
                0, self.index[:key_len], self.key_cache[layer_idx][0][self.num_sink_tokens + key_len :]
            )
            self.key_cache[layer_idx]._index_copy(
                1, self.index[:key_len], self.key_cache[layer_idx][1][self.num_sink_tokens + key_len :]
            )

            # add new tokens
            self.key_cache[layer_idx]._index_copy(0, self.index[-key_len:], key_states)
            self.key_cache[layer_idx]._index_copy(1, self.index[-key_len:], value_states)
        else:
            self.key_cache[layer_idx]._index_copy(
                0, self.index, key_states[:, : self.window_length - self.num_sink_tokens]
            )
            self.key_cache[layer_idx]._index_copy(
                1, self.index, value_states[:, : self.window_length - self.num_sink_tokens]
            )

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass
<<<<<<< HEAD
>>>>>>> 523380cb6 (Draft version of new KV Caching)
=======
>>>>>>> 1129513b3 (Address numerous PR suggestions)
