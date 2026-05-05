
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utilsdfa import (
    TASK_ACCEPTNESS,
    LengthPolicy,
    TensorDFALanguage,
    _mean_var_min_max,
    _word_to_tuple,
    build_characteristic_prefix_basis,
    build_characteristic_suffix_set,
    compute_forward_path_counts,
    make_structured_tensor_dfa,
    report_dfa,
    run_valid_seq_with_states,
)


# =========================================================
# Reveal config
# =========================================================

class RevealType(str, Enum):
    ABSOLUTE = "absolute"
    RATIO = "ratio"


@dataclass
class RevealConfig:
    value: float
    type: str = RevealType.RATIO.value


# =========================================================
# Unified dataset
# =========================================================

class UnifiedSequenceDataset(Dataset):
    """
    seqs               : [N, T] long
    lengths            : [N] long
    validity           : [N] long
    acceptness         : [N] long
    valid_lengths      : [N] long
    labels             : [N] long
    final_state        : [N] long   # terminal DFA state, else -1 if absent

    compatible_token   : [N, T, A] bool
    compatible_state   : [N, T, S] bool
    valid_event_state  : [N, T, S] bool
    invalid_event_state: [N, T, S] bool
    compatible_mask    : [N, T] bool   # reveal/use mask over positions only
    """

    def __init__(
        self,
        seqs: torch.Tensor,
        lengths: torch.Tensor,
        validity: torch.Tensor,
        acceptness: torch.Tensor,
        valid_lengths: torch.Tensor,
        labels: torch.Tensor,
        final_state: torch.Tensor,
        compatible_token: torch.Tensor,
        compatible_state: torch.Tensor,
        valid_event_state: torch.Tensor,
        invalid_event_state: torch.Tensor,
        compatible_mask: torch.Tensor,
        pad_id: int,
        alphabet_size: int,
        num_states: int,
    ):
        assert seqs.dim() == 2
        N, T = seqs.shape
        for x in (lengths, validity, acceptness, valid_lengths, labels, final_state):
            assert x.dim() == 1 and x.numel() == N
        assert compatible_token.shape == (N, T, int(alphabet_size))
        assert compatible_state.shape == (N, T, int(num_states))
        assert valid_event_state.shape == (N, T, int(num_states))
        assert invalid_event_state.shape == (N, T, int(num_states))
        assert compatible_mask.shape == (N, T)

        self.seqs = seqs.long()
        self.lengths = lengths.long()
        self.validity = validity.long()
        self.acceptness = acceptness.long()
        self.valid_lengths = valid_lengths.long()
        self.labels = labels.long()
        self.final_state = final_state.long()

        self.compatible_token = compatible_token.bool()
        self.compatible_state = compatible_state.bool()
        self.valid_event_state = valid_event_state.bool()
        self.invalid_event_state = invalid_event_state.bool()
        self.compatible_mask = compatible_mask.bool()

        self.pad_id = int(pad_id)
        self.alphabet_size = int(alphabet_size)
        self.num_states = int(num_states)

    def __len__(self) -> int:
        return int(self.lengths.numel())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "seq": self.seqs[idx],
            "length": self.lengths[idx],
            "validity": self.validity[idx],
            "acceptness": self.acceptness[idx],
            "valid_length": self.valid_lengths[idx],
            "label": self.labels[idx],
            "final_state": self.final_state[idx],
            "compatible_token": self.compatible_token[idx],
            "compatible_state": self.compatible_state[idx],
            "valid_event_state": self.valid_event_state[idx],
            "invalid_event_state": self.invalid_event_state[idx],
            "compatible_mask": self.compatible_mask[idx],
        }


class UnifiedPadCollator:
    def __init__(self, bos_id: int, pad_id: int):
        self.bos_id = int(bos_id)
        self.pad_id = int(pad_id)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        lengths = torch.stack([x["length"] for x in batch], dim=0)
        T = int(lengths.max().item()) if lengths.numel() > 0 else 0
        A = int(batch[0]["compatible_token"].size(-1)) if batch else 0
        S = int(batch[0]["compatible_state"].size(-1)) if batch else 0
        B = len(batch)

        seqs = torch.full((B, T), fill_value=self.pad_id, dtype=torch.long)
        compatible_token = torch.zeros((B, T, A), dtype=torch.bool)
        compatible_state = torch.zeros((B, T, S), dtype=torch.bool)
        valid_event_state = torch.zeros((B, T, S), dtype=torch.bool)
        invalid_event_state = torch.zeros((B, T, S), dtype=torch.bool)
        compatible_mask = torch.zeros((B, T), dtype=torch.bool)

        validity = torch.stack([x["validity"] for x in batch], dim=0).long()
        acceptness = torch.stack([x["acceptness"] for x in batch], dim=0).long()
        valid_lengths = torch.stack([x["valid_length"] for x in batch], dim=0).long()
        labels = torch.stack([x["label"] for x in batch], dim=0).long()
        final_states = torch.stack([x["final_state"] for x in batch], dim=0).long()

        for i, item in enumerate(batch):
            L = int(item["length"].item())
            seqs[i, :L] = item["seq"][:L].long()
            compatible_token[i, :L] = item["compatible_token"][:L].bool()
            compatible_state[i, :L] = item["compatible_state"][:L].bool()
            valid_event_state[i, :L] = item["valid_event_state"][:L].bool()
            invalid_event_state[i, :L] = item["invalid_event_state"][:L].bool()
            compatible_mask[i, :L] = item["compatible_mask"][:L].bool()

        input_ids = torch.full((B, T + 1), fill_value=self.pad_id, dtype=torch.long)
        input_ids[:, 0] = self.bos_id
        input_ids[:, 1:] = seqs
        attention_mask = (input_ids != self.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seqs": seqs,
            "lengths": lengths.long(),
            "validity": validity,
            "acceptness": acceptness,
            "valid_lengths": valid_lengths,
            "labels": labels,
            "final_states": final_states,
            "compatible_token": compatible_token,
            "compatible_state": compatible_state,
            "valid_event_state": valid_event_state,
            "invalid_event_state": invalid_event_state,
            "compatible_mask": compatible_mask,
        }


# =========================================================
# Internal helpers
# =========================================================

_EPS = 1e-12


def _empty_supervision(n: int, T: int, A: int, S: int) -> Tuple[torch.Tensor, ...]:
    return (
        torch.zeros((n, T, A), dtype=torch.bool),
        torch.zeros((n, T, S), dtype=torch.bool),
        torch.zeros((n, T, S), dtype=torch.bool),
        torch.zeros((n, T, S), dtype=torch.bool),
        torch.zeros((n, T), dtype=torch.bool),
    )


def _build_dataset_from_seq_bank(
    seq_bank: List[torch.Tensor],
    labels_meta: List[Dict[str, int]],
    pad_id: int,
    alphabet_size: int,
    num_states: int,
) -> UnifiedSequenceDataset:
    N = len(seq_bank)
    T = max((int(x.numel()) for x in seq_bank), default=0)
    seqs = torch.full((N, T), fill_value=pad_id, dtype=torch.long)
    for i, seq in enumerate(seq_bank):
        seqs[i, : seq.numel()] = seq.long()

    lengths = torch.tensor([int(x.numel()) for x in seq_bank], dtype=torch.long)
    validity = torch.tensor([int(m["validity"]) for m in labels_meta], dtype=torch.long)
    acceptness = torch.tensor([int(m["acceptness"]) for m in labels_meta], dtype=torch.long)
    valid_lengths = torch.tensor([int(m["valid_length"]) for m in labels_meta], dtype=torch.long)
    labels = torch.tensor([int(m["label"]) for m in labels_meta], dtype=torch.long)
    final_state = torch.tensor([int(m.get("final_state", -1)) for m in labels_meta], dtype=torch.long)

    compatible_token, compatible_state, valid_event_state, invalid_event_state, compatible_mask = _empty_supervision(
        N, T, alphabet_size, num_states
    )

    return UnifiedSequenceDataset(
        seqs=seqs,
        lengths=lengths,
        validity=validity,
        acceptness=acceptness,
        valid_lengths=valid_lengths,
        labels=labels,
        final_state=final_state,
        compatible_token=compatible_token,
        compatible_state=compatible_state,
        valid_event_state=valid_event_state,
        invalid_event_state=invalid_event_state,
        compatible_mask=compatible_mask,
        pad_id=pad_id,
        alphabet_size=alphabet_size,
        num_states=num_states,
    )


def _concat_datasets(
    datasets: List[UnifiedSequenceDataset],
    pad_id: int,
    alphabet_size: int,
    num_states: int,
) -> UnifiedSequenceDataset:
    datasets = [ds for ds in datasets if len(ds) > 0]
    if not datasets:
        return UnifiedSequenceDataset(
            seqs=torch.empty((0, 0), dtype=torch.long),
            lengths=torch.empty((0,), dtype=torch.long),
            validity=torch.empty((0,), dtype=torch.long),
            acceptness=torch.empty((0,), dtype=torch.long),
            valid_lengths=torch.empty((0,), dtype=torch.long),
            labels=torch.empty((0,), dtype=torch.long),
            final_state=torch.empty((0,), dtype=torch.long),
            compatible_token=torch.empty((0, 0, alphabet_size), dtype=torch.bool),
            compatible_state=torch.empty((0, 0, num_states), dtype=torch.bool),
            valid_event_state=torch.empty((0, 0, num_states), dtype=torch.bool),
            invalid_event_state=torch.empty((0, 0, num_states), dtype=torch.bool),
            compatible_mask=torch.empty((0, 0), dtype=torch.bool),
            pad_id=pad_id,
            alphabet_size=alphabet_size,
            num_states=num_states,
        )

    total = sum(len(ds) for ds in datasets)
    T = max(ds.seqs.size(1) for ds in datasets)
    A = alphabet_size
    S = num_states

    seqs = torch.full((total, T), fill_value=pad_id, dtype=torch.long)
    lengths = torch.empty((total,), dtype=torch.long)
    validity = torch.empty((total,), dtype=torch.long)
    acceptness = torch.empty((total,), dtype=torch.long)
    valid_lengths = torch.empty((total,), dtype=torch.long)
    labels = torch.empty((total,), dtype=torch.long)
    final_state = torch.empty((total,), dtype=torch.long)
    compatible_token = torch.zeros((total, T, A), dtype=torch.bool)
    compatible_state = torch.zeros((total, T, S), dtype=torch.bool)
    valid_event_state = torch.zeros((total, T, S), dtype=torch.bool)
    invalid_event_state = torch.zeros((total, T, S), dtype=torch.bool)
    compatible_mask = torch.zeros((total, T), dtype=torch.bool)

    start = 0
    for ds in datasets:
        n = len(ds)
        end = start + n
        curT = ds.seqs.size(1)
        seqs[start:end, :curT] = ds.seqs
        lengths[start:end] = ds.lengths
        validity[start:end] = ds.validity
        acceptness[start:end] = ds.acceptness
        valid_lengths[start:end] = ds.valid_lengths
        labels[start:end] = ds.labels
        final_state[start:end] = ds.final_state
        compatible_token[start:end, :curT] = ds.compatible_token
        compatible_state[start:end, :curT] = ds.compatible_state
        valid_event_state[start:end, :curT] = ds.valid_event_state
        invalid_event_state[start:end, :curT] = ds.invalid_event_state
        compatible_mask[start:end, :curT] = ds.compatible_mask
        start = end

    return UnifiedSequenceDataset(
        seqs=seqs,
        lengths=lengths,
        validity=validity,
        acceptness=acceptness,
        valid_lengths=valid_lengths,
        labels=labels,
        final_state=final_state,
        compatible_token=compatible_token,
        compatible_state=compatible_state,
        valid_event_state=valid_event_state,
        invalid_event_state=invalid_event_state,
        compatible_mask=compatible_mask,
        pad_id=pad_id,
        alphabet_size=alphabet_size,
        num_states=num_states,
    )


# =========================================================
# Sampling helpers
# =========================================================

def _sample_valid_forward_seq(
    dfa: TensorDFALanguage,
    length: int,
    generator: Optional[torch.Generator] = None,
    max_tries: int = 2048,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(max_tries):
        s = torch.tensor(dfa.start_state, dtype=torch.long, device=dfa.device)
        seq = torch.empty((length,), dtype=torch.long, device=dfa.device)
        ok = True
        for t in range(length):
            valid_syms = torch.where(dfa.out_valid_sym_mask[s])[0]
            if valid_syms.numel() == 0:
                ok = False
                break
            k = int(torch.randint(valid_syms.numel(), (1,), generator=generator).item())
            a = valid_syms[k]
            seq[t] = a
            s = dfa.delta[s, a]
        if ok:
            return seq, s
    return None


def _sample_lengths_batch(
    length_policy: LengthPolicy,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if batch_size <= 0:
        return torch.empty((0,), dtype=torch.long)
    if length_policy.uniform:
        return torch.randint(
            int(length_policy.min_len),
            int(length_policy.max_len) + 1,
            (batch_size,),
            generator=generator,
            dtype=torch.long,
        )
    lo = max(int(length_policy.min_len), int(length_policy.avg_len) - int(length_policy.jitter))
    hi = min(int(length_policy.max_len), int(length_policy.avg_len) + int(length_policy.jitter))
    return torch.randint(
        lo,
        hi + 1,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    )


def _ensure_dfa_sampling_cache(dfa: TensorDFALanguage) -> None:
    if hasattr(dfa, 'valid_symbol_counts') and hasattr(dfa, 'valid_symbols'):
        return
    valid_mask = dfa.out_valid_sym_mask
    counts = valid_mask.sum(dim=1).long()
    max_valid = int(counts.max().item()) if counts.numel() > 0 else 0
    valid_symbols = torch.full(
        (dfa.num_states, max_valid),
        -1,
        dtype=torch.long,
        device=dfa.device,
    )
    for q in range(dfa.num_states):
        syms = torch.where(valid_mask[q])[0]
        if syms.numel() > 0:
            valid_symbols[q, : syms.numel()] = syms
    dfa.valid_symbol_counts = counts
    dfa.valid_symbols = valid_symbols


@torch.no_grad()
def _sample_valid_forward_batch_padded(
    dfa: TensorDFALanguage,
    lengths: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_dfa_sampling_cache(dfa)
    lengths = lengths.to(dfa.device).long()
    B = int(lengths.numel())
    T = int(lengths.max().item()) if B > 0 else 0
    pad_id = int(dfa.alphabet_size)

    seqs = torch.full((B, T), fill_value=pad_id, dtype=torch.long, device=dfa.device)
    cur = torch.full((B,), int(dfa.start_state), dtype=torch.long, device=dfa.device)
    success = torch.ones((B,), dtype=torch.bool, device=dfa.device)

    if B == 0 or T == 0:
        return seqs, cur, success

    for t in range(T):
        active = success & (lengths > t)
        if not bool(active.any().item()):
            continue

        active_idx = torch.where(active)[0]
        states = cur[active_idx]
        counts = dfa.valid_symbol_counts[states]
        has_edge = counts > 0

        if bool((~has_edge).any().item()):
            success[active_idx[~has_edge]] = False
        if not bool(has_edge.any().item()):
            continue

        good_idx = active_idx[has_edge]
        good_states = states[has_edge]
        good_counts = counts[has_edge]
        chooser = torch.floor(torch.rand((good_idx.numel(),), device=dfa.device) * good_counts.to(torch.float32)).long()
        chosen_sym = dfa.valid_symbols[good_states, chooser]
        seqs[good_idx, t] = chosen_sym
        cur[good_idx] = dfa.delta[good_states, chosen_sym]

    return seqs, cur, success


def _append_selected_sequences(
    dst_seq_bank: List[torch.Tensor],
    dst_metas: List[Dict[str, int]],
    *,
    seqs: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    final_states: torch.Tensor,
    selected_idx: torch.Tensor,
) -> int:
    added = 0
    for idx in selected_idx.detach().cpu().tolist():
        L = int(lengths[idx].item())
        y = int(labels[idx].item())
        qf = int(final_states[idx].item())
        dst_seq_bank.append(seqs[idx, :L].detach().cpu())
        dst_metas.append({
            "validity": 1,
            "acceptness": y,
            "valid_length": L,
            "label": y,
            "final_state": qf,
        })
        added += 1
    return added


def _sample_valid_reverse_seq_uniform(
    dfa: TensorDFALanguage,
    length: int,
    final_state: int | torch.Tensor,
    counts: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    q_final = int(torch.as_tensor(final_state).item())
    if float(counts[length, q_final].item()) <= 0.0:
        return None

    rev_mask = (
        (dfa.delta.unsqueeze(0) == torch.arange(dfa.num_states, device=dfa.device).view(dfa.num_states, 1, 1))
        & dfa.valid_mask.unsqueeze(0)
    )

    cur = torch.tensor(q_final, dtype=torch.long, device=dfa.device)
    rev_syms = torch.empty((length,), dtype=torch.long, device=dfa.device)

    for r in range(length):
        need_t = length - (r + 1)
        src_all, sym_all = torch.where(rev_mask[cur])
        if src_all.numel() == 0:
            return None
        w = counts[need_t, src_all]
        feasible = w > 0
        if not bool(feasible.any().item()):
            return None
        src_good = src_all[feasible]
        sym_good = sym_all[feasible]
        w_good = w[feasible].detach().cpu()
        k = int(torch.multinomial(w_good, 1, generator=generator).item())
        rev_syms[r] = sym_good[k]
        cur = src_good[k]

    if int(cur.item()) != dfa.start_state:
        return None
    return torch.flip(rev_syms, dims=(0,)), cur


def _sample_acceptness_label(
    dfa: TensorDFALanguage,
    length_policy: LengthPolicy,
    target_label: int,
    counts: torch.Tensor,
    sampling_mode: str,
    generator: Optional[torch.Generator] = None,
    max_tries: int = 4096,
) -> Optional[torch.Tensor]:
    for _ in range(max_tries):
        seq = None
        if sampling_mode == "forward":
            L = length_policy.sample(generator=generator)
            out = _sample_valid_forward_seq(dfa, L, generator=generator)
            seq = None if out is None else out[0]
        elif sampling_mode == "reverse":
            L = length_policy.sample(generator=generator)
            label_mask = dfa.accept_mask if target_label == 1 else dfa.reject_mask
            weights = counts[L] * label_mask.to(dtype=counts.dtype)
            pool = torch.where(weights > 0)[0]
            if pool.numel() == 0:
                continue
            pool_w = weights[pool].detach().cpu()
            k = int(torch.multinomial(pool_w, 1, generator=generator).item())
            final_state = pool[k]
            out = _sample_valid_reverse_seq_uniform(dfa, L, final_state, counts, generator=generator)
            seq = None if out is None else out[0]
        else:
            if int(torch.randint(0, 2, (1,), generator=generator).item()) == 0:
                L = length_policy.sample(generator=generator)
                out = _sample_valid_forward_seq(dfa, L, generator=generator)
                seq = None if out is None else out[0]
            else:
                L = length_policy.sample(generator=generator)
                label_mask = dfa.accept_mask if target_label == 1 else dfa.reject_mask
                weights = counts[L] * label_mask.to(dtype=counts.dtype)
                pool = torch.where(weights > 0)[0]
                if pool.numel() == 0:
                    continue
                pool_w = weights[pool].detach().cpu()
                k = int(torch.multinomial(pool_w, 1, generator=generator).item())
                final_state = pool[k]
                out = _sample_valid_reverse_seq_uniform(dfa, L, final_state, counts, generator=generator)
                seq = None if out is None else out[0]

        if seq is None:
            continue
        terminal_state, valid, _ = dfa.run_details(seq)
        if not bool(valid.item()):
            continue
        label = int(dfa.label_of_state(terminal_state).item())
        if label == target_label:
            return seq

    return None


# =========================================================
# DFA supervision primitives
# =========================================================

def compute_valid_sequence_tables_dp(
    dfa: TensorDFALanguage,
    seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      states_before: [L]
      states_after : [L]
      suffix_ok    : [L+1, S], where suffix_ok[t, s] is True iff seq[t:] is valid from state s
    """
    seq = seq.to(dfa.device).long()
    run_info = run_valid_seq_with_states(dfa, seq)
    if run_info is None:
        raise ValueError("compute_valid_sequence_tables_dp expects a valid base sequence.")

    states_before, states_after = run_info
    L = int(seq.numel())
    S = dfa.num_states
    suffix_ok = torch.zeros((L + 1, S), dtype=torch.bool, device=dfa.device)
    suffix_ok[L] = True

    for t in range(L - 1, -1, -1):
        tok = int(seq[t].item())
        nexts = dfa.delta[:, tok]
        valid = nexts >= 0
        if bool(valid.any().item()):
            suffix_ok[t, valid] = suffix_ok[t + 1, nexts[valid]]

    return states_before, states_after, suffix_ok


def compute_dfa_supervision_single(
    dfa: TensorDFALanguage,
    seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      compatible_token   : [L, A] bool
      compatible_state   : [L, S] bool
      valid_event_state  : [L, S] bool
      invalid_event_state: [L, S] bool

    Semantics:
      - compatible_token[t, a] is True iff replacing seq[t] with token a keeps the whole trace valid.
        This includes the original token.
      - compatible_state[t, s] is True iff successor state s is compatible with the remaining suffix seq[t+1:].
      - valid_event_state[t, s] aggregates over replacement tokens only (original token excluded):
            True iff there exists a != seq[t] with delta(state_before_t, a) == s and that replacement stays valid.
      - invalid_event_state[t, s] aggregates over replacement tokens only (original token excluded):
            True iff there exists a != seq[t] with delta(state_before_t, a) == s, the one-step transition is defined,
            but the remaining suffix is no longer valid.
    """
    batch = compute_dfa_supervision_batched_padded(
        dfa,
        seq.view(1, -1),
        torch.tensor([int(seq.numel())], dtype=torch.long, device=seq.device if isinstance(seq, torch.Tensor) else dfa.device),
    )
    return batch[0][0], batch[1][0], batch[2][0], batch[3][0]


def compute_dfa_supervision_batched_padded(
    dfa: TensorDFALanguage,
    seqs: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched padded computation for mixed-length valid sequences.

    Args:
      seqs   : [B, T] padded token tensor
      lengths: [B] true lengths

    Returns:
      compatible_token   : [B, T, A] bool
      compatible_state   : [B, T, S] bool
      valid_event_state  : [B, T, S] bool
      invalid_event_state: [B, T, S] bool

    Only positions t < lengths[b] are filled; padded positions remain all-zero.
    This function assumes all sequences in the batch are valid up to their respective lengths.
    """
    seqs = seqs.to(dfa.device).long()
    lengths = lengths.to(dfa.device).long()

    if seqs.dim() != 2:
        raise ValueError("seqs must have shape [B, T].")
    if lengths.dim() != 1 or lengths.numel() != seqs.size(0):
        raise ValueError("lengths must have shape [B].")

    B, T = seqs.shape
    A = dfa.alphabet_size
    S = dfa.num_states

    compatible_token = torch.zeros((B, T, A), dtype=torch.bool, device=dfa.device)
    compatible_state = torch.zeros((B, T, S), dtype=torch.bool, device=dfa.device)
    valid_event_state = torch.zeros((B, T, S), dtype=torch.bool, device=dfa.device)
    invalid_event_state = torch.zeros((B, T, S), dtype=torch.bool, device=dfa.device)

    if B == 0 or T == 0:
        return compatible_token, compatible_state, valid_event_state, invalid_event_state

    states_before = torch.zeros((B, T), dtype=torch.long, device=dfa.device)
    cur = torch.full((B,), int(dfa.start_state), dtype=torch.long, device=dfa.device)
    for t in range(T):
        active = lengths > t
        if not bool(active.any().item()):
            continue
        states_before[active, t] = cur[active]
        tok = seqs[active, t]
        ns = dfa.delta[cur[active], tok]
        if bool((ns < 0).any().item()):
            raise ValueError("compute_dfa_supervision_batched_padded expects valid sequences only.")
        cur[active] = ns

    suffix_ok = torch.zeros((B, T + 1, S), dtype=torch.bool, device=dfa.device)
    batch_idx = torch.arange(B, device=dfa.device)
    suffix_ok[batch_idx, lengths, :] = True
    for t in range(T - 1, -1, -1):
        active = lengths > t
        if not bool(active.any().item()):
            continue
        active_idx = torch.where(active)[0]
        toks = seqs[active_idx, t]
        for a in torch.unique(toks).tolist():
            a = int(a)
            sel = active_idx[toks == a]
            nexts = dfa.delta[:, a]
            defined = nexts >= 0
            if bool(defined.any().item()):
                cur_slice = suffix_ok[sel, t].clone()
                cur_slice[:, defined] = suffix_ok[sel, t + 1][:, nexts[defined]]
                suffix_ok[sel, t] = cur_slice

    token_ids = torch.arange(A, device=dfa.device).view(1, A)
    for t in range(T):
        active = lengths > t
        if not bool(active.any().item()):
            continue
        active_idx = torch.where(active)[0]
        src = states_before[active_idx, t]
        orig = seqs[active_idx, t]
        suffix_next = suffix_ok[active_idx, t + 1]

        compatible_state[active_idx, t] = suffix_next

        nexts = dfa.delta[src]
        defined = nexts >= 0
        nexts_clamped = torch.where(defined, nexts, torch.zeros_like(nexts))
        token_ok = defined & suffix_next.gather(1, nexts_clamped)
        compatible_token[active_idx, t] = token_ok

        alt_mask = token_ids != orig.unsqueeze(1)
        alt_valid = alt_mask & token_ok
        alt_invalid = alt_mask & defined & (~token_ok)

        if bool(alt_valid.any().item()):
            ves = torch.zeros((active_idx.numel(), S), dtype=torch.long, device=dfa.device)
            ves.scatter_add_(1, nexts_clamped, alt_valid.long())
            valid_event_state[active_idx, t] = ves > 0

        if bool(alt_invalid.any().item()):
            ies = torch.zeros((active_idx.numel(), S), dtype=torch.long, device=dfa.device)
            ies.scatter_add_(1, nexts_clamped, alt_invalid.long())
            invalid_event_state[active_idx, t] = ies > 0

    return compatible_token, compatible_state, valid_event_state, invalid_event_state


def build_compatible_position_mask(
    lengths: torch.Tensor,
    validity: torch.Tensor,
    max_T: int,
    reveal_config: Optional[RevealConfig],
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    compatible_mask[i, t] == 1 means: reveal/use compatible supervision at that valid position.
    Padded positions and invalid sequences always get 0.
    If reveal_config is None, reveal all valid positions.
    """
    lengths = lengths.long()
    validity = validity.long()
    N = int(lengths.numel())
    mask = torch.zeros((N, max_T), dtype=torch.bool)

    for i in range(N):
        L = int(lengths[i].item())
        if L <= 0 or int(validity[i].item()) != 1:
            continue

        if reveal_config is None:
            mask[i, :L] = True
            continue

        if reveal_config.type == RevealType.ABSOLUTE.value:
            m = int(reveal_config.value)
            m = max(0, min(L, m))
        elif reveal_config.type == RevealType.RATIO.value:
            r = float(reveal_config.value)
            r = max(0.0, min(1.0, r))
            if r == 0.0:
                m = 0
            else:
                m = min(L, max(1, int(round(r * L))))
        else:
            raise ValueError(f"Unknown reveal type: {reveal_config.type}")

        if m <= 0:
            continue

        perm = torch.randperm(L, generator=generator)[:m]
        mask[i, perm] = True

    return mask


def attach_dfa_supervision(
    dfa: TensorDFALanguage,
    ds: UnifiedSequenceDataset,
    *,
    reveal_config: Optional[RevealConfig] = None,
    reveal_seed: int = 0,
    show_progress: bool = True,
    build_batch_size: int = 8192,
) -> UnifiedSequenceDataset:
    N = len(ds)
    T = ds.seqs.size(1)
    A = dfa.alphabet_size
    S = dfa.num_states

    compatible_token = torch.zeros((N, T, A), dtype=torch.bool)
    compatible_state = torch.zeros((N, T, S), dtype=torch.bool)
    valid_event_state = torch.zeros((N, T, S), dtype=torch.bool)
    invalid_event_state = torch.zeros((N, T, S), dtype=torch.bool)

    valid_indices = torch.where((ds.validity == 1) & (ds.lengths > 0))[0]
    batch_size = max(1, int(build_batch_size))
    iterator = range(0, int(valid_indices.numel()), batch_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=(int(valid_indices.numel()) + batch_size - 1) // batch_size,
            desc="attach_dfa_supervision",
            leave=False,
            unit="batch",
        )

    for start in iterator:
        batch_idx = valid_indices[start:start + batch_size]
        if batch_idx.numel() == 0:
            continue
        seqs_batch = ds.seqs[batch_idx].to(dfa.device)
        lengths_batch = ds.lengths[batch_idx].to(dfa.device)
        ct, cs, ves, ies = compute_dfa_supervision_batched_padded(dfa, seqs_batch, lengths_batch)
        compatible_token[batch_idx] = ct.detach().cpu()
        compatible_state[batch_idx] = cs.detach().cpu()
        valid_event_state[batch_idx] = ves.detach().cpu()
        invalid_event_state[batch_idx] = ies.detach().cpu()

    g = torch.Generator(device="cpu")
    g.manual_seed(int(reveal_seed))
    compatible_mask = build_compatible_position_mask(
        ds.lengths,
        ds.validity,
        T,
        reveal_config=reveal_config,
        generator=g,
    )

    return UnifiedSequenceDataset(
        seqs=ds.seqs.clone(),
        lengths=ds.lengths.clone(),
        validity=ds.validity.clone(),
        acceptness=ds.acceptness.clone(),
        valid_lengths=ds.valid_lengths.clone(),
        labels=ds.labels.clone(),
        final_state=ds.final_state.clone(),
        compatible_token=compatible_token,
        compatible_state=compatible_state,
        valid_event_state=valid_event_state,
        invalid_event_state=invalid_event_state,
        compatible_mask=compatible_mask,
        pad_id=ds.pad_id,
        alphabet_size=ds.alphabet_size,
        num_states=ds.num_states,
    )


def apply_reveal_config(
    ds: UnifiedSequenceDataset,
    reveal_config: Optional[RevealConfig],
    *,
    seed: int = 0,
) -> UnifiedSequenceDataset:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    new_mask = build_compatible_position_mask(
        ds.lengths,
        ds.validity,
        ds.seqs.size(1),
        reveal_config=reveal_config,
        generator=g,
    )
    return UnifiedSequenceDataset(
        seqs=ds.seqs.clone(),
        lengths=ds.lengths.clone(),
        validity=ds.validity.clone(),
        acceptness=ds.acceptness.clone(),
        valid_lengths=ds.valid_lengths.clone(),
        labels=ds.labels.clone(),
        final_state=ds.final_state.clone(),
        compatible_token=ds.compatible_token.clone(),
        compatible_state=ds.compatible_state.clone(),
        valid_event_state=ds.valid_event_state.clone(),
        invalid_event_state=ds.invalid_event_state.clone(),
        compatible_mask=new_mask,
        pad_id=ds.pad_id,
        alphabet_size=ds.alphabet_size,
        num_states=ds.num_states,
    )


# =========================================================
# Builders
# =========================================================

def build_labeled_dataset(
    dfa: TensorDFALanguage,
    n_label1: int,
    n_label0: int,
    length_policy: LengthPolicy,
    sampling_mode: str = "mixed",
    pad_id: Optional[int] = None,
    seed: int = 123,
    *,
    reveal_config: Optional[RevealConfig] = None,
    reveal_seed: int = 0,
    show_progress: bool = True,
    build_batch_size: int = 8192,
) -> UnifiedSequenceDataset:
    if pad_id is None:
        pad_id = dfa.alphabet_size
    if sampling_mode not in {"forward", "mixed", "reverse"}:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    total = int(n_label1 + n_label0)
    need_pos = int(n_label1)
    need_neg = int(n_label0)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    seq_bank: List[torch.Tensor] = []
    metas: List[Dict[str, int]] = []
    batch_size = max(1, int(build_batch_size))

    progress = None
    if show_progress:
        progress = tqdm(total=total, desc="build_labeled_dataset", leave=False, unit="seq")

    if sampling_mode == "reverse":
        counts = compute_forward_path_counts(dfa, max_len=int(length_policy.max_len))
        targets = torch.empty((total,), dtype=torch.long)
        targets[:n_label1] = 1
        targets[n_label1:] = 0
        targets = targets[torch.randperm(total, generator=g)]
        for i in range(total):
            target = int(targets[i].item())
            seq = _sample_acceptness_label(dfa, length_policy, target, counts, sampling_mode, generator=g)
            if seq is None:
                raise RuntimeError(f"Failed to sample label={target} for task_type={TASK_ACCEPTNESS}")
            terminal_state, valid, valid_length = dfa.run_details(seq)
            if not bool(valid.item()):
                raise RuntimeError("reverse sampler returned invalid sequence")
            seq_cpu = seq.detach().cpu()
            seq_bank.append(seq_cpu)
            metas.append({
                "validity": 1,
                "acceptness": int(target),
                "valid_length": int(valid_length.item()),
                "label": int(target),
                "final_state": int(terminal_state.item()),
            })
            if progress is not None:
                progress.update(1)
    else:
        while need_pos > 0 or need_neg > 0:
            lengths_batch = _sample_lengths_batch(length_policy, batch_size, generator=g)
            seqs_batch, terminal_batch, success_batch = _sample_valid_forward_batch_padded(
                dfa,
                lengths_batch,
                generator=g,
            )
            good_idx = torch.where(success_batch)[0]
            if good_idx.numel() == 0:
                continue

            labels_cpu = dfa.label_of_state(terminal_batch).detach().cpu().long()
            good_labels = labels_cpu[good_idx.detach().cpu()]
            perm = torch.randperm(int(good_idx.numel()), generator=g)
            good_idx = good_idx[perm]
            good_labels = good_labels[perm]

            pos_idx = good_idx[good_labels == 1]
            neg_idx = good_idx[good_labels == 0]
            take_pos = pos_idx[: min(need_pos, int(pos_idx.numel()))]
            take_neg = neg_idx[: min(need_neg, int(neg_idx.numel()))]

            added_pos = _append_selected_sequences(
                dst_seq_bank=seq_bank,
                dst_metas=metas,
                seqs=seqs_batch,
                lengths=lengths_batch,
                labels=labels_cpu,
                final_states=terminal_batch.detach().cpu(),
                selected_idx=take_pos,
            )
            added_neg = _append_selected_sequences(
                dst_seq_bank=seq_bank,
                dst_metas=metas,
                seqs=seqs_batch,
                lengths=lengths_batch,
                labels=labels_cpu,
                final_states=terminal_batch.detach().cpu(),
                selected_idx=take_neg,
            )
            need_pos -= added_pos
            need_neg -= added_neg

            if progress is not None and (added_pos + added_neg) > 0:
                progress.update(added_pos + added_neg)
                progress.set_postfix({
                    "need_pos": int(need_pos),
                    "need_neg": int(need_neg),
                    "batch": int(batch_size),
                })

    if progress is not None:
        progress.close()

    ds = _build_dataset_from_seq_bank(
        seq_bank,
        metas,
        pad_id=pad_id,
        alphabet_size=dfa.alphabet_size,
        num_states=dfa.num_states,
    )
    return attach_dfa_supervision(
        dfa,
        ds,
        reveal_config=reveal_config,
        reveal_seed=reveal_seed,
        show_progress=show_progress,
        build_batch_size=build_batch_size,
    )

def build_classical_characteristic_dataset(
    dfa: TensorDFALanguage,
    pad_id: Optional[int] = None,
    max_suffix_len: int = 12,
    include_epsilon: bool = True,
    *,
    reveal_config: Optional[RevealConfig] = None,
    reveal_seed: int = 0,
    show_progress: bool = True,
    build_batch_size: int = 8192,
) -> UnifiedSequenceDataset:
    if pad_id is None:
        pad_id = dfa.alphabet_size

    X, _, _ = build_characteristic_prefix_basis(dfa)
    E = build_characteristic_suffix_set(dfa, max_suffix_len=max_suffix_len, include_epsilon=include_epsilon)

    seen: set[Tuple[int, ...]] = set()
    seq_bank: List[torch.Tensor] = []
    metas: List[Dict[str, int]] = []

    iterator = X
    if show_progress:
        iterator = tqdm(X, total=len(X), desc="build_classical_characteristic_dataset", leave=False)

    for x in iterator:
        for e in E:
            seq = torch.cat([x, e], dim=0)
            key = _word_to_tuple(seq)
            if key in seen:
                continue
            terminal_state, valid, valid_length = dfa.run_details(seq)
            if not bool(valid.item()):
                continue
            seq_bank.append(seq.detach().cpu())
            label = int(dfa.label_of_state(terminal_state).item())
            metas.append(
                {
                    "validity": 1,
                    "acceptness": label,
                    "valid_length": int(valid_length.item()),
                    "label": label,
                    "final_state": int(terminal_state.item()),
                }
            )
            seen.add(key)

    ds = _build_dataset_from_seq_bank(
        seq_bank,
        metas,
        pad_id=pad_id,
        alphabet_size=dfa.alphabet_size,
        num_states=dfa.num_states,
    )
    return attach_dfa_supervision(
        dfa,
        ds,
        reveal_config=reveal_config,
        reveal_seed=reveal_seed,
        show_progress=show_progress,
        build_batch_size=build_batch_size,
    )


def build_natural_characteristic_dataset(
    dfa: TensorDFALanguage,
    length_policy: LengthPolicy,
    pad_id: Optional[int] = None,
    max_suffix_len: int = 12,
    include_epsilon: bool = True,
    natural_size: Optional[int] = None,
    natural_multiplier: float = 1.0,
    sampling_mode: str = "mixed",
    seed: int = 123,
    *,
    reveal_config: Optional[RevealConfig] = None,
    reveal_seed: int = 0,
    show_progress: bool = True,
    build_batch_size: int = 8192,
) -> UnifiedSequenceDataset:
    if pad_id is None:
        pad_id = dfa.alphabet_size

    core = build_classical_characteristic_dataset(
        dfa,
        pad_id=pad_id,
        max_suffix_len=max_suffix_len,
        include_epsilon=include_epsilon,
        reveal_config=reveal_config,
        reveal_seed=reveal_seed,
        show_progress=show_progress,
        build_batch_size=build_batch_size,
    )

    if natural_size is None:
        natural_size = max(1, int(round(float(natural_multiplier) * max(1, len(core)))))

    n_pos = natural_size // 2
    n_neg = natural_size - n_pos

    natural = build_labeled_dataset(
        dfa=dfa,
        n_label1=n_pos,
        n_label0=n_neg,
        length_policy=length_policy,
        sampling_mode=sampling_mode,
        pad_id=pad_id,
        seed=seed,
        reveal_config=reveal_config,
        reveal_seed=reveal_seed,
        show_progress=show_progress,
        build_batch_size=build_batch_size,
    )

    return _concat_datasets(
        [core, natural],
        pad_id=pad_id,
        alphabet_size=dfa.alphabet_size,
        num_states=dfa.num_states,
    )


# =========================================================
# Reports / ambiguity laws (computed from stored annotations only)
# =========================================================

def _mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean((y_pred - y_true) ** 2).item()) if y_true.numel() > 0 else 0.0


def _relative_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> float:
    num = torch.sum(torch.abs(y_pred - y_true))
    den = torch.clamp(torch.sum(torch.abs(y_true)), min=eps)
    return float((num / den).item())


def _solve_lstsq(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)


def _effective_density_from_dfa(dfa: TensorDFALanguage) -> float:
    cfg = getattr(dfa, "generation_config", None)
    if isinstance(cfg, dict) and "density" in cfg:
        return float(cfg["density"])
    total = int(dfa.num_states * dfa.alphabet_size)
    defined = int((dfa.delta >= 0).sum().item())
    return 0.0 if total == 0 else float(defined / total)


class _Welford:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None

    def add(self, x: float) -> None:
        self.n += 1
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def to_dict(self) -> Dict[str, Any]:
        var = 0.0 if self.n == 0 else (self.M2 / self.n)
        return {
            "count": int(self.n),
            "mean": float(self.mean) if self.n else 0.0,
            "variance": float(var),
            "min": float(self.min) if self.min is not None else 0.0,
            "max": float(self.max) if self.max is not None else 0.0,
        }


def _fit_V_law(by_depth: List[Dict[str, Any]], *, S: int, p: float) -> Dict[str, Any]:
    ks = torch.tensor([float(int(row["k"])) for row in by_depth], dtype=torch.float64)
    y = torch.tensor([float(row["mean"]) for row in by_depth], dtype=torch.float64)
    base = float(S - 1) * (float(p) ** ks)
    X = torch.ones((len(by_depth), 1), dtype=torch.float64)
    c = _solve_lstsq(X, y - base)[0]
    pred = base + c
    return {
        "form": "V_k ≈ (S - 1) p^k + c_V, with c_V = a_V p + b_V p^2",
        "p": float(p),
        "scalar": float(c.item()),
        "mse": _mse(y, pred),
        "relative_loss": _relative_loss(y, pred),
    }


def _fit_E_law(by_depth: List[Dict[str, Any]], *, A: int, p: float) -> Dict[str, Any]:
    ks = torch.tensor([float(int(row["k"])) for row in by_depth], dtype=torch.float64)
    y = torch.tensor([float(row["mean"]) for row in by_depth], dtype=torch.float64)
    base = float(A - 1) * (float(p) ** (ks + 1.0))
    X = torch.ones((len(by_depth), 1), dtype=torch.float64)
    c = _solve_lstsq(X, y - base)[0]
    pred = base + c
    return {
        "form": "E_k ≈ (A - 1) p^(k + 1) + c_E, with c_E = a_E p + b_E p^2",
        "p": float(p),
        "scalar": float(c.item()),
        "mse": _mse(y, pred),
        "relative_loss": _relative_loss(y, pred),
    }


def _fit_Dplus_law(by_depth: List[Dict[str, Any]], *, A: int, p: float) -> Dict[str, Any]:
    ks = torch.tensor([float(int(row["k"])) for row in by_depth], dtype=torch.float64)
    y = torch.tensor([float(row["mean"]) for row in by_depth], dtype=torch.float64)
    base = float(A - 1) * (float(p) ** (ks + 1.0))
    X = torch.ones((len(by_depth), 1), dtype=torch.float64)
    c = _solve_lstsq(X, y - base)[0]
    pred = base + c
    return {
        "form": "D_k^+ ≈ (A - 1) p^(k + 1) + c_D, with c_D = a_D p + b_D p^2",
        "p": float(p),
        "scalar": float(c.item()),
        "mse": _mse(y, pred),
        "relative_loss": _relative_loss(y, pred),
    }


def _fit_Dminus_law(by_depth: List[Dict[str, Any]], *, p: float) -> Dict[str, Any]:
    ks = torch.tensor([float(int(row["k"])) for row in by_depth], dtype=torch.float64)
    y = torch.tensor([float(row["mean"]) for row in by_depth], dtype=torch.float64)
    shape = 1.0 - (float(p) ** ks)
    X = shape.unsqueeze(1)
    c = _solve_lstsq(X, y)[0]
    pred = shape * c
    return {
        "form": "D_k^- ≈ c_I (1 - p^k), with c_I = a_I p + b_I p^2",
        "p": float(p),
        "scalar": float(c.item()),
        "mse": _mse(y, pred),
        "relative_loss": _relative_loss(y, pred),
    }


def _build_position_use_mask(
    ds: UnifiedSequenceDataset,
    *,
    use_revealed_positions: bool,
) -> torch.Tensor:
    N, T = ds.seqs.shape
    pos_mask = torch.zeros((N, T), dtype=torch.bool)
    if N == 0 or T == 0:
        return pos_mask

    arange_t = torch.arange(T).view(1, T)
    valid_pos = (arange_t < ds.lengths.view(N, 1)) & (ds.validity.view(N, 1) == 1)

    if use_revealed_positions and bool(ds.compatible_mask.any().item()):
        pos_mask = valid_pos & ds.compatible_mask
    else:
        pos_mask = valid_pos

    return pos_mask


def _histogram_from_values(values: torch.Tensor, size: int) -> Dict[str, Any]:
    values = values.long()
    good = values >= 0
    if size <= 0:
        return {"num_recorded": int(good.sum().item()), "num_unique": 0, "counts": []}
    if not bool(good.any().item()):
        return {"num_recorded": 0, "num_unique": 0, "counts": [0 for _ in range(size)]}
    counts = torch.bincount(values[good], minlength=size)
    return {
        "num_recorded": int(good.sum().item()),
        "num_unique": int((counts > 0).sum().item()),
        "counts": counts.tolist(),
    }


def _stats_from_values(values: torch.Tensor) -> Dict[str, Any]:
    values = values.to(torch.float64).cpu().reshape(-1)
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    mean = float(values.mean().item())
    variance = float(values.var(unbiased=False).item())
    return {
        "count": int(values.numel()),
        "mean": mean,
        "variance": variance,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }



def _aggregate_law_stats(values: torch.Tensor, by_depth: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = values.to(torch.float64).cpu().reshape(-1)
    if values.numel() == 0:
        return {
            "pos_mean": 0.0,
            "pool_mean": 0.0,
            "depth_count": 0,
            "position_count": 0,
            "sum": 0.0,
        }

    depth_means = torch.tensor([float(row["mean"]) for row in by_depth], dtype=torch.float64)
    return {
        "pos_mean": float(depth_means.mean().item()) if depth_means.numel() > 0 else 0.0,
        "pool_mean": float(values.mean().item()),
        "depth_count": int(len(by_depth)),
        "position_count": int(values.numel()),
        "sum": float(values.sum().item()),
    }



def _stats_by_depth_from_vectors(k: torch.Tensor, values: torch.Tensor) -> List[Dict[str, Any]]:
    if k.numel() == 0:
        return []
    k = k.long().cpu()
    values = values.to(torch.float64).cpu()
    max_k = int(k.max().item())
    counts = torch.bincount(k, minlength=max_k + 1)
    sums = torch.bincount(k, weights=values, minlength=max_k + 1)
    sums_sq = torch.bincount(k, weights=values * values, minlength=max_k + 1)
    out: List[Dict[str, Any]] = []
    for kk in torch.where(counts > 0)[0].tolist():
        mask = k == int(kk)
        n = int(counts[kk].item())
        mean = float((sums[kk] / max(n, 1)).item())
        variance = float((sums_sq[kk] / max(n, 1) - mean * mean).item())
        out.append({
            "k": int(kk),
            "count": n,
            "mean": mean,
            "variance": variance,
            "min": float(values[mask].min().item()),
            "max": float(values[mask].max().item()),
        })
    return out


@torch.no_grad()
def _ambiguity_law_report_from_dataset(
    ds: UnifiedSequenceDataset,
    dfa: TensorDFALanguage,
    *,
    use_revealed_positions: bool = True,
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    pos_mask = _build_position_use_mask(ds, use_revealed_positions=use_revealed_positions)
    N, T = ds.seqs.shape
    p_fit = _effective_density_from_dfa(dfa)
    if N == 0 or T == 0:
        empty: List[Dict[str, Any]] = []
        zero_stats = {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
        empty_agg = _aggregate_law_stats(torch.empty((0,), dtype=torch.float64), empty)
        return {
            "V_k": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "E_k": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "D_k_plus": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "D_k_minus": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "meta": {
                "positions_used": 0,
                "used_revealed_positions": bool(use_revealed_positions and ds.compatible_mask.any().item()),
                "fit_density_p": float(p_fit),
                "max_depth": None if max_depth is None else int(max_depth),
            },
        }

    t_ids = torch.arange(T).view(1, T)
    k_all = ds.lengths.view(N, 1) - t_ids - 1
    use_mask = pos_mask & (k_all >= 0)
    if max_depth is not None:
        use_mask = use_mask & (k_all <= int(max_depth))

    positions_used = int(use_mask.sum().item())
    if positions_used == 0:
        empty: List[Dict[str, Any]] = []
        zero_stats = {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
        empty_agg = _aggregate_law_stats(torch.empty((0,), dtype=torch.float64), empty)
        return {
            "V_k": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "E_k": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "D_k_plus": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "D_k_minus": {"overall": dict(zero_stats), "aggregates": dict(empty_agg), "by_depth": empty, "fit": None},
            "meta": {
                "positions_used": 0,
                "used_revealed_positions": bool(use_revealed_positions and ds.compatible_mask.any().item()),
                "fit_density_p": float(p_fit),
                "max_depth": None if max_depth is None else int(max_depth),
            },
        }

    k = k_all[use_mask].long()
    V = ds.compatible_state.sum(dim=-1).to(torch.float64)[use_mask] - 1.0
    E = ds.compatible_token.sum(dim=-1).to(torch.float64)[use_mask] - 1.0
    Dp = ds.valid_event_state.sum(dim=-1).to(torch.float64)[use_mask]
    Dm = ds.invalid_event_state.sum(dim=-1).to(torch.float64)[use_mask]

    V_by_depth = _stats_by_depth_from_vectors(k, V)
    E_by_depth = _stats_by_depth_from_vectors(k, E)
    Dp_by_depth = _stats_by_depth_from_vectors(k, Dp)
    Dm_by_depth = _stats_by_depth_from_vectors(k, Dm)

    return {
        "V_k": {
            "overall": _stats_from_values(V),
            "aggregates": _aggregate_law_stats(V, V_by_depth),
            "by_depth": V_by_depth,
            "fit": None if len(V_by_depth) == 0 else _fit_V_law(V_by_depth, S=dfa.num_states, p=p_fit),
        },
        "E_k": {
            "overall": _stats_from_values(E),
            "aggregates": _aggregate_law_stats(E, E_by_depth),
            "by_depth": E_by_depth,
            "fit": None if len(E_by_depth) == 0 else _fit_E_law(E_by_depth, A=dfa.alphabet_size, p=p_fit),
        },
        "D_k_plus": {
            "overall": _stats_from_values(Dp),
            "aggregates": _aggregate_law_stats(Dp, Dp_by_depth),
            "by_depth": Dp_by_depth,
            "fit": None if len(Dp_by_depth) == 0 else _fit_Dplus_law(Dp_by_depth, A=dfa.alphabet_size, p=p_fit),
        },
        "D_k_minus": {
            "overall": _stats_from_values(Dm),
            "aggregates": _aggregate_law_stats(Dm, Dm_by_depth),
            "by_depth": Dm_by_depth,
            "fit": None if len(Dm_by_depth) == 0 else _fit_Dminus_law(Dm_by_depth, p=p_fit),
        },
        "meta": {
            "positions_used": int(positions_used),
            "used_revealed_positions": bool(use_revealed_positions and ds.compatible_mask.any().item()),
            "fit_density_p": float(p_fit),
            "max_depth": None if max_depth is None else int(max_depth),
        },
    }


@torch.no_grad()
def report_dataset(
    ds: UnifiedSequenceDataset,
    dfa: Optional[TensorDFALanguage] = None,
    *,
    use_revealed_positions: bool = True,
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    N = int(len(ds))
    lengths = ds.lengths.detach().cpu().tolist()
    valid_lengths = ds.valid_lengths.detach().cpu().tolist()
    labels = ds.labels.detach().cpu().tolist()
    validity = ds.validity.detach().cpu().tolist()
    acceptness = ds.acceptness.detach().cpu().tolist()

    positive_count = sum(1 for v in labels if int(v) == 1)
    zero_count = sum(1 for v in labels if int(v) == 0)

    out: Dict[str, Any] = {
        "dataset": {
            "N": int(N),
            "pad_id": int(ds.pad_id),
            "alphabet_size": int(ds.alphabet_size),
            "num_states": int(ds.num_states),
            "positive_count": int(positive_count),
            "zero_count": int(zero_count),
            "validity_positive_count": int(sum(int(v) for v in validity if int(v) >= 0)),
            "validity_negative_count": int(sum(1 for v in validity if int(v) == 0)),
            "acceptness_negative_one_count": int(sum(1 for v in acceptness if int(v) == -1)),
            "compatible_revealed_positions": int(ds.compatible_mask.sum().item()),
        },
        "lengths": _mean_var_min_max([float(int(L)) for L in lengths]),
        "valid_lengths": _mean_var_min_max([float(int(v)) for v in valid_lengths]),
        "final_states": {
            "all": _histogram_from_values(ds.final_state, ds.num_states),
            "label1": _histogram_from_values(ds.final_state[ds.labels == 1], ds.num_states),
            "label0": _histogram_from_values(ds.final_state[ds.labels == 0], ds.num_states),
        },
        "compatible": {},
        "oracle": {},
        "V_k": {"overall": {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}, "aggregates": {"pos_mean": 0.0, "pool_mean": 0.0, "depth_count": 0, "position_count": 0, "sum": 0.0}, "by_depth": [], "fit": None},
        "E_k": {"overall": {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}, "aggregates": {"pos_mean": 0.0, "pool_mean": 0.0, "depth_count": 0, "position_count": 0, "sum": 0.0}, "by_depth": [], "fit": None},
        "D_k_plus": {"overall": {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}, "aggregates": {"pos_mean": 0.0, "pool_mean": 0.0, "depth_count": 0, "position_count": 0, "sum": 0.0}, "by_depth": [], "fit": None},
        "D_k_minus": {"overall": {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}, "aggregates": {"pos_mean": 0.0, "pool_mean": 0.0, "depth_count": 0, "position_count": 0, "sum": 0.0}, "by_depth": [], "fit": None},
        "summary": {},
    }

    pos_mask_all = _build_position_use_mask(ds, use_revealed_positions=False)
    pos_mask_used = _build_position_use_mask(ds, use_revealed_positions=use_revealed_positions)

    if pos_mask_all.numel() > 0 and bool(pos_mask_all.any().item()):
        all_pos = pos_mask_all
        out["compatible"]["all_valid_positions"] = int(all_pos.sum().item())
        out["compatible"]["revealed_positions"] = int(ds.compatible_mask.sum().item())
        out["compatible"]["mean_true_tokens_per_valid_position_all"] = float(ds.compatible_token.sum(dim=-1)[all_pos].float().mean().item())
        out["compatible"]["mean_true_states_per_valid_position_all"] = float(ds.compatible_state.sum(dim=-1)[all_pos].float().mean().item())
        out["compatible"]["mean_valid_event_states_per_valid_position_all"] = float(ds.valid_event_state.sum(dim=-1)[all_pos].float().mean().item())
        out["compatible"]["mean_invalid_event_states_per_valid_position_all"] = float(ds.invalid_event_state.sum(dim=-1)[all_pos].float().mean().item())

    if pos_mask_used.numel() > 0 and bool(pos_mask_used.any().item()):
        used_pos = pos_mask_used
        out["compatible"]["used_positions"] = int(used_pos.sum().item())
        out["compatible"]["mean_true_tokens_per_used_position"] = float(ds.compatible_token.sum(dim=-1)[used_pos].float().mean().item())
        out["compatible"]["mean_true_states_per_used_position"] = float(ds.compatible_state.sum(dim=-1)[used_pos].float().mean().item())
        out["compatible"]["mean_valid_event_states_per_used_position"] = float(ds.valid_event_state.sum(dim=-1)[used_pos].float().mean().item())
        out["compatible"]["mean_invalid_event_states_per_used_position"] = float(ds.invalid_event_state.sum(dim=-1)[used_pos].float().mean().item())

    if dfa is not None and N > 0:
        terminal_state, valid_b, validlen_b = dfa.run_details_batch(ds.seqs.to(dfa.device), ds.lengths.to(dfa.device))
        terminal_state = terminal_state.detach().cpu()
        valid_b = valid_b.detach().cpu()
        validlen_b = validlen_b.detach().cpu()
        labels_pred = torch.full((N,), -1, dtype=torch.long)
        if bool(valid_b.any().item()):
            labels_pred[valid_b] = dfa.label_of_state(terminal_state[valid_b].to(dfa.device)).detach().cpu().long()

        validity_t = ds.validity.detach().cpu()
        acceptness_t = ds.acceptness.detach().cpu()
        valid_lengths_t = ds.valid_lengths.detach().cpu()
        labels_t = ds.labels.detach().cpu()
        final_state_t = ds.final_state.detach().cpu()

        validity_present = validity_t >= 0
        acc_present = (acceptness_t >= 0) & valid_b
        vlen_present = valid_lengths_t >= 0
        label_present = labels_t >= 0
        final_present = final_state_t >= 0

        validity_correct = int((validity_t[validity_present] == valid_b.long()[validity_present]).sum().item()) if bool(validity_present.any().item()) else 0
        acc_correct = int((acceptness_t[acc_present] == labels_pred[acc_present]).sum().item()) if bool(acc_present.any().item()) else 0
        vlen_correct = int((valid_lengths_t[vlen_present] == validlen_b[vlen_present]).sum().item()) if bool(vlen_present.any().item()) else 0
        label_correct = int((labels_t[label_present] == labels_pred[label_present]).sum().item()) if bool(label_present.any().item()) else 0
        final_correct = int((final_state_t[final_present] == terminal_state[final_present]).sum().item()) if bool(final_present.any().item()) else 0

        oracle_total = int(validity_present.sum().item() + acc_present.sum().item() + vlen_present.sum().item() + label_present.sum().item() + final_present.sum().item())
        oracle_correct = validity_correct + acc_correct + vlen_correct + label_correct + final_correct

        out["oracle"] = {
            "agreement_fraction": 0.0 if oracle_total == 0 else float(oracle_correct / oracle_total),
            "checked_field_count": int(oracle_total),
            "validity_agreement_fraction": 0.0 if int(validity_present.sum().item()) == 0 else float(validity_correct / int(validity_present.sum().item())),
            "acceptness_agreement_fraction": 0.0 if int(acc_present.sum().item()) == 0 else float(acc_correct / int(acc_present.sum().item())),
            "valid_length_agreement_fraction": 0.0 if int(vlen_present.sum().item()) == 0 else float(vlen_correct / int(vlen_present.sum().item())),
            "label_agreement_fraction": 0.0 if int(label_present.sum().item()) == 0 else float(label_correct / int(label_present.sum().item())),
            "final_state_agreement_fraction": 0.0 if int(final_present.sum().item()) == 0 else float(final_correct / int(final_present.sum().item())),
        }
        out["final_states"]["oracle"] = _histogram_from_values(terminal_state, ds.num_states)

    if dfa is not None:
        rep = _ambiguity_law_report_from_dataset(ds, dfa, use_revealed_positions=use_revealed_positions, max_depth=max_depth)
        out["V_k"] = rep["V_k"]
        out["E_k"] = rep["E_k"]
        out["D_k_plus"] = rep["D_k_plus"]
        out["D_k_minus"] = rep["D_k_minus"]
        out["summary"] = {
            "mean_length": float(out["lengths"]["mean"]),
            "mean_valid_length": float(out["valid_lengths"]["mean"]),
            "positions_used_for_laws": int(rep["meta"]["positions_used"]),
            "used_revealed_positions": bool(rep["meta"]["used_revealed_positions"]),
            "fit_density_p": float(rep["meta"]["fit_density_p"]),
            "V_k_overall_mean": float(out["V_k"]["overall"]["mean"]),
            "E_k_overall_mean": float(out["E_k"]["overall"]["mean"]),
            "D_k_plus_overall_mean": float(out["D_k_plus"]["overall"]["mean"]),
            "D_k_minus_overall_mean": float(out["D_k_minus"]["overall"]["mean"]),
            "V_k_pos_mean": float(out["V_k"]["aggregates"]["pos_mean"]),
            "V_k_pool_mean": float(out["V_k"]["aggregates"]["pool_mean"]),
            "E_k_pos_mean": float(out["E_k"]["aggregates"]["pos_mean"]),
            "E_k_pool_mean": float(out["E_k"]["aggregates"]["pool_mean"]),
            "D_k_plus_pos_mean": float(out["D_k_plus"]["aggregates"]["pos_mean"]),
            "D_k_plus_pool_mean": float(out["D_k_plus"]["aggregates"]["pool_mean"]),
            "D_k_minus_pos_mean": float(out["D_k_minus"]["aggregates"]["pos_mean"]),
            "D_k_minus_pool_mean": float(out["D_k_minus"]["aggregates"]["pool_mean"]),
            "V_k_k0_mean": float(out["V_k"]["by_depth"][0]["mean"]) if len(out["V_k"]["by_depth"]) > 0 else 0.0,
            "E_k_k0_mean": float(out["E_k"]["by_depth"][0]["mean"]) if len(out["E_k"]["by_depth"]) > 0 else 0.0,
            "D_k_plus_k0_mean": float(out["D_k_plus"]["by_depth"][0]["mean"]) if len(out["D_k_plus"]["by_depth"]) > 0 else 0.0,
            "D_k_minus_k0_mean": float(out["D_k_minus"]["by_depth"][0]["mean"]) if len(out["D_k_minus"]["by_depth"]) > 0 else 0.0,
            "stored_final_states": int((ds.final_state >= 0).sum().item()),
        }
    else:
        out["summary"] = {
            "mean_length": float(out["lengths"]["mean"]),
            "mean_valid_length": float(out["valid_lengths"]["mean"]),
            "stored_final_states": int((ds.final_state >= 0).sum().item()),
        }

    return out


dataset_diagnostics = report_dataset


# =========================================================
# CLI / main
# =========================================================

def _parse_reveal_config(args: argparse.Namespace) -> Optional[RevealConfig]:
    if args.reveal_type is None:
        return None
    return RevealConfig(value=float(args.reveal_value), type=str(args.reveal_type))


def _build_length_policy_from_args(args: argparse.Namespace) -> LengthPolicy:
    return LengthPolicy(
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        avg_len=int(args.avg_len),
        jitter=int(args.jitter),
        uniform=bool(args.uniform_lengths),
    )


def _build_dataset_from_args(args: argparse.Namespace, dfa: TensorDFALanguage) -> UnifiedSequenceDataset:
    length_policy = _build_length_policy_from_args(args)
    reveal_config = _parse_reveal_config(args)

    if args.dataset_kind == "labeled":
        return build_labeled_dataset(
            dfa=dfa,
            n_label1=int(args.n_label1),
            n_label0=int(args.n_label0),
            length_policy=length_policy,
            sampling_mode=str(args.sampling_mode),
            pad_id=None,
            seed=int(args.dataset_seed),
            reveal_config=reveal_config,
            reveal_seed=int(args.reveal_seed),
            show_progress=bool(args.show_progress),
        )

    if args.dataset_kind == "classical":
        return build_classical_characteristic_dataset(
            dfa=dfa,
            pad_id=None,
            max_suffix_len=int(args.max_suffix_len),
            include_epsilon=bool(args.include_epsilon),
            reveal_config=reveal_config,
            reveal_seed=int(args.reveal_seed),
            show_progress=bool(args.show_progress),
        )

    if args.dataset_kind == "natural":
        return build_natural_characteristic_dataset(
            dfa=dfa,
            length_policy=length_policy,
            pad_id=None,
            max_suffix_len=int(args.max_suffix_len),
            include_epsilon=bool(args.include_epsilon),
            natural_size=None if args.natural_size is None else int(args.natural_size),
            natural_multiplier=float(args.natural_multiplier),
            sampling_mode=str(args.sampling_mode),
            seed=int(args.dataset_seed),
            reveal_config=reveal_config,
            reveal_seed=int(args.reveal_seed),
            show_progress=bool(args.show_progress),
        )

    raise ValueError(f"Unknown dataset_kind: {args.dataset_kind}")


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DFA dataset annotations and run ambiguity-law diagnostics.")

    parser.add_argument("--num-states", type=int, default=100)
    parser.add_argument("--alphabet-size", type=int, default=100)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--accept-prob", type=float, default=0.5)
    parser.add_argument("--dfa-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=_default_device())

    parser.add_argument("--dataset-kind", type=str, choices=["labeled", "classical", "natural"], default="labeled")
    parser.add_argument("--n-label1", type=int, default=51200)
    parser.add_argument("--n-label0", type=int, default=51200)
    parser.add_argument("--dataset-seed", type=int, default=123)
    parser.add_argument("--sampling-mode", type=str, choices=["forward", "reverse", "mixed"], default="mixed")

    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--avg-len", type=int, default=20)
    parser.add_argument("--jitter", type=int, default=15)
    parser.add_argument("--uniform-lengths", action="store_true")

    parser.add_argument("--max-suffix-len", type=int, default=12)
    parser.add_argument("--include-epsilon", action="store_true")
    parser.add_argument("--natural-size", type=int, default=None)
    parser.add_argument("--natural-multiplier", type=float, default=1.0)

    parser.add_argument("--reveal-type", type=str, choices=[RevealType.ABSOLUTE.value, RevealType.RATIO.value], default=None)
    parser.add_argument("--reveal-value", type=float, default=1.0)
    parser.add_argument("--reveal-seed", type=int, default=0)

    parser.add_argument("--law-max-depth", type=int, default=None)
    parser.add_argument("--use-all-positions", action="store_true", help="Ignore compatible_mask and use all valid positions in diagnostics.")
    parser.add_argument("--show-progress", dest="show_progress", action="store_true", default=True)
    parser.add_argument("--no-show-progress", dest="show_progress", action="store_false")
    parser.add_argument("--indent", type=int, default=2)

    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_argparser()
    args = parser.parse_args(argv)

    dfa = make_structured_tensor_dfa(
        num_states=int(args.num_states),
        alphabet_size=int(args.alphabet_size),
        density=float(args.density),
        accept_prob=float(args.accept_prob),
        seed=int(args.dfa_seed),
        device=str(args.device),
    )

    ds = _build_dataset_from_args(args, dfa)
    diag = report_dataset(
        ds,
        dfa=dfa,
        use_revealed_positions=not bool(args.use_all_positions),
        max_depth=args.law_max_depth,
    )

    payload = {
        "dfa": report_dfa(dfa),
        "dataset_report": diag,
    }
    print(json.dumps(payload, indent=int(args.indent), sort_keys=False))
    return payload


if __name__ == "__main__":
    main()
