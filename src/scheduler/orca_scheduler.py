import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from src.model.iteration import IterationBatch, IterationItem
from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention
from src.model.paged_decoder import PagedDecoderLM


ORCA_STRATEGY = "orca"
SARATHI_STRATEGY = "sarathi"
SUPPORTED_SCHEDULING_STRATEGIES = (ORCA_STRATEGY, SARATHI_STRATEGY)


@dataclass
class RequestState:
    request_id: int
    prompt: str
    max_new_tokens: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    pending_token_id: Optional[int] = None
    reserved_blocks: int = 0
    prefill_cursor: int = 0


class ContinuousBatchScheduler:
    def __init__(
        self,
        model_engine: PagedDecoderLM,
        max_batch_size: int,
        tokenizer,
        kv_manager: KVCacheManager,
        paged_attn_manager: PagedAttention,
        eos_token_id=None,
        scheduling_strategy=ORCA_STRATEGY,
        prefill_chunk_size=16,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if scheduling_strategy not in SUPPORTED_SCHEDULING_STRATEGIES:
            choices = ", ".join(SUPPORTED_SCHEDULING_STRATEGIES)
            raise ValueError(
                f"Unknown scheduling strategy '{scheduling_strategy}'. "
                f"Choose one of: {choices}"
            )
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")

        self.model_engine = model_engine
        self.max_batch_size = max_batch_size
        self.tokenizer = tokenizer
        self.kv_manager = kv_manager
        self.paged_attn_manager = paged_attn_manager
        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        self.scheduling_strategy = scheduling_strategy
        self.prefill_chunk_size = prefill_chunk_size

        self.waiting = deque() # Requests with prompt tokens still to prefill.
        self.active: Dict[int, RequestState] = {} # Requests that finished prefill and are generating tokens.
        self.finished: Dict[int, RequestState] = {} # Completed requests whose KV blocks have been released.
        self.next_request_id = 0 # Every arriving request an increasing ID
        self.reserved_blocks = 0 # How much KV-cache capacity has been promised to all admitted requests

    def add_request(self, prompt, max_new_tokens=200):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        prompt_token_ids = self.tokenizer.encode(prompt)
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.reshape(-1).tolist()
        else:
            prompt_token_ids = list(prompt_token_ids)

        return self.add_token_request(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
        )

    def add_token_request(
        self,
        prompt_token_ids,
        max_new_tokens=200,
        prompt="<tokenized prompt>",
    ):
        """Add an already-tokenized request for exact workload reproduction."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.reshape(-1).tolist()
        else:
            prompt_token_ids = list(prompt_token_ids)

        if not prompt_token_ids:
            raise ValueError("The tokenizer produced an empty prompt")
        if any(not isinstance(token_id, int) for token_id in prompt_token_ids):
            raise ValueError("The tokenizer must produce integer token IDs")
        if min(prompt_token_ids) < 0 or max(prompt_token_ids) >= self.model_engine.config.vocab_size:
            raise ValueError("A prompt token is outside the model vocabulary")
        if len(prompt_token_ids) + max_new_tokens > self.model_engine.config.max_sequence_length:
            raise ValueError("Prompt plus requested generation exceeds the model context limit")

        request_id = self.next_request_id
        self.next_request_id += 1
        request_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_token_ids=prompt_token_ids,
        )
        self.waiting.append(request_state)
        return request_id

    def _required_blocks(self, request):
        # Estimates the maximum KV-cache space a request may need
        maximum_tokens = len(request.prompt_token_ids) + request.max_new_tokens
        return math.ceil(maximum_tokens / self.kv_manager.block_size)

    def select_requests(self):
        if self.scheduling_strategy == SARATHI_STRATEGY:
            return self._select_sarathi_requests()

        request_pool = list(self.active.values()) + list(self.waiting)
        # FCFS scheduling
        request_pool.sort(key=lambda request: request.request_id)

        selected = []
        newly_reserved_blocks = 0
        for request in request_pool:
            if len(selected) == self.max_batch_size:
                break

            if request.reserved_blocks == 0:
                required_blocks = self._required_blocks(request)
                if self.reserved_blocks + newly_reserved_blocks + required_blocks > self.kv_manager.total_available_blocks:
                    break
                newly_reserved_blocks += required_blocks

            selected.append(request)

        return selected

    def _select_sarathi_requests(self):
        """Select one prefill chunk and fill remaining slots with decodes."""
        active_requests = sorted(
            self.active.values(),
            key=lambda request: request.request_id,
        )
        prefill_request = None

        if self.waiting:
            candidate = self.waiting[0]
            if candidate.reserved_blocks > 0:
                prefill_request = candidate
            else:
                required_blocks = self._required_blocks(candidate)
                if self.reserved_blocks + required_blocks <= self.kv_manager.total_available_blocks:
                    prefill_request = candidate

        decode_capacity = self.max_batch_size - (1 if prefill_request else 0)
        selected = active_requests[:decode_capacity]
        if prefill_request is not None:
            selected.append(prefill_request)
        return selected

    def build_iteration_batch(self, requests):
        # Converts selected requests into one flattened Orca batch
        items = []
        flat_input_ids = []
        flat_position_ids = []
        offset = 0

        for request in requests:
            if request.pending_token_id is None:
                phase = "prefill"
                prefill_start = request.prefill_cursor
                if self.scheduling_strategy == SARATHI_STRATEGY:
                    prefill_end = min(
                        prefill_start + self.prefill_chunk_size,
                        len(request.prompt_token_ids),
                    )
                else:
                    prefill_end = len(request.prompt_token_ids)
                token_ids = tuple(request.prompt_token_ids[prefill_start:prefill_end])
                position_ids = tuple(range(prefill_start, prefill_end))
                produces_output = prefill_end == len(request.prompt_token_ids)
            else:
                phase = "decode"
                token_ids = (request.pending_token_id,)
                position_ids = (self.kv_manager.requests[request.request_id].sequence_length,)
                produces_output = True

            end_offset = offset + len(token_ids)
            items.append(
                IterationItem(
                    request_id=request.request_id,
                    phase=phase,
                    token_ids=token_ids,
                    position_ids=position_ids,
                    start_offset=offset,
                    end_offset=end_offset,
                    produces_output=produces_output,
                )
            )
            flat_input_ids.extend(token_ids)
            flat_position_ids.extend(position_ids)
            offset = end_offset

        device = next(self.model_engine.parameters()).device
        return IterationBatch(
            items=tuple(items),
            input_ids=torch.tensor(flat_input_ids, dtype=torch.long, device=device),
            position_ids=torch.tensor(flat_position_ids, dtype=torch.long, device=device),
        )

    def step(self):
        requests = self.select_requests()
        if not requests:
            if self.waiting or self.active:
                raise RuntimeError("No request can be scheduled with the available KV memory")
            return {}

        # For every new request it records its block reservation
        new_requests = [request for request in requests if request.reserved_blocks == 0]
        for request in new_requests:
            request.reserved_blocks = self._required_blocks(request)
            self.reserved_blocks += request.reserved_blocks

        iteration_batch = self.build_iteration_batch(requests)
        try:
            with torch.inference_mode():
                # Only decodes and final prefill chunks produce logits rows.
                next_token_logits = self.model_engine.forward_iteration(
                    iteration_batch,
                    self.kv_manager,
                    self.paged_attn_manager,
                )
        except Exception:
            for request in new_requests:
                self.reserved_blocks -= request.reserved_blocks
                request.reserved_blocks = 0
            raise

        # The scheduler selects the next token from each row using argmax
        next_token_ids = iter(next_token_logits.argmax(dim=-1).tolist())
        emitted_tokens = {}
        completed_prefill_ids = set()
        for request, item in zip(requests, iteration_batch.items):
            if item.phase == "prefill":
                request.prefill_cursor += item.token_count
                if request.prefill_cursor < len(request.prompt_token_ids):
                    continue
                completed_prefill_ids.add(request.request_id)

            try:
                next_token_id = next(next_token_ids)
            except StopIteration as error:
                raise RuntimeError("Model returned too few logits rows") from error
            request.generated_token_ids.append(next_token_id)
            request.pending_token_id = next_token_id
            emitted_tokens[request.request_id] = next_token_id

            finished = len(request.generated_token_ids) >= request.max_new_tokens
            if self.eos_token_id is not None and next_token_id == self.eos_token_id:
                finished = True

            if finished:
                self.active.pop(request.request_id, None)
                self.kv_manager.free_request(request.request_id)
                self.reserved_blocks -= request.reserved_blocks
                request.reserved_blocks = 0
                self.finished[request.request_id] = request
            else:
                self.active[request.request_id] = request

        try:
            next(next_token_ids)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Model returned too many logits rows")

        if completed_prefill_ids:
            self.waiting = deque(
                request
                for request in self.waiting
                if request.request_id not in completed_prefill_ids
            )

        return emitted_tokens

    def run_until_complete(self):
        while self.waiting or self.active:
            self.step()
        return {
            request_id: self.tokenizer.decode(request.generated_token_ids)
            for request_id, request in self.finished.items()
        }
