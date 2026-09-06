from dataclasses import dataclass
from typing import Hashable, Literal, Tuple

import torch


IterationPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class IterationItem:
    request_id: Hashable
    phase: IterationPhase
    token_ids: Tuple[int, ...]
    position_ids: Tuple[int, ...]
    start_offset: int
    end_offset: int
    produces_output: bool = True

    def __post_init__(self):
        # Describes one request participating in the current Orca iteration
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"Unsupported iteration phase: {self.phase}")
        if not self.token_ids:
            raise ValueError("An iteration item must contain at least one token")
        if len(self.token_ids) != len(self.position_ids):
            raise ValueError("token_ids and position_ids must have the same length")
        if self.start_offset < 0 or self.end_offset <= self.start_offset:
            raise ValueError("Invalid flattened-token offsets")
        if self.end_offset - self.start_offset != len(self.token_ids):
            raise ValueError("Offsets do not match the number of item tokens")
        if self.phase == "decode" and len(self.token_ids) != 1:
            raise ValueError("A decode item must contain exactly one token")
        if self.phase == "decode" and not self.produces_output:
            raise ValueError("A decode item must produce output logits")

    @property
    def token_count(self):
        return self.end_offset - self.start_offset


@dataclass(frozen=True)
class IterationBatch:
    # represents the complete work sent to the model for one Orca iteration.
    items: Tuple[IterationItem, ...]
    input_ids: torch.Tensor
    position_ids: torch.Tensor

    def __post_init__(self):
        if not self.items:
            raise ValueError("An iteration batch cannot be empty")
        if self.input_ids.ndim != 1 or self.position_ids.ndim != 1:
            raise ValueError("Flattened input_ids and position_ids must be one-dimensional")
        if self.input_ids.shape != self.position_ids.shape:
            raise ValueError("input_ids and position_ids must have the same shape")
        if self.input_ids.dtype != torch.long or self.position_ids.dtype != torch.long:
            raise ValueError("input_ids and position_ids must use torch.long")

        expected_start = 0
        request_ids = set()
        for item in self.items:
            if item.request_id in request_ids:
                raise ValueError("A request can appear only once in an iteration")
            if item.start_offset != expected_start:
                raise ValueError("Iteration item offsets must be contiguous")
            request_ids.add(item.request_id)
            expected_start = item.end_offset

        if expected_start != self.input_ids.numel():
            raise ValueError("Iteration item offsets do not cover the flattened inputs")

    @property
    def total_tokens(self):
        return self.input_ids.numel()
