from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

# =========================================================
# Constants / policies
# =========================================================

class TaskName(str, Enum):
    ACCEPTNESS = "acceptness"


class FamilyName(str, Enum):
    RANDOM = "random"
    CLUSTERED = "clustered"
    MODULAR = "modular"
    MERGE = "merge"


TASK_ACCEPTNESS = TaskName.ACCEPTNESS.value

FAMILY_RANDOM = 0
FAMILY_CLUSTERED = 1
FAMILY_MODULAR = 2
FAMILY_MERGE = 3

FAMILY_NAME_TO_ID = {
    FamilyName.RANDOM.value: FAMILY_RANDOM,
    FamilyName.CLUSTERED.value: FAMILY_CLUSTERED,
    FamilyName.MODULAR.value: FAMILY_MODULAR,
    FamilyName.MERGE.value: FAMILY_MERGE,
}
FAMILY_ID_TO_NAME = {v: k for k, v in FAMILY_NAME_TO_ID.items()}

_EPS = 1e-12


@dataclass
class LengthPolicy:
    min_len: int = 5
    max_len: int = 40
    avg_len: int = 20
    jitter: int = 15
    uniform: bool = False

    def sample(self, generator: Optional[torch.Generator] = None) -> int:
        if self.uniform:
            return int(torch.randint(self.min_len, self.max_len + 1, (1,), generator=generator).item())
        lo = max(self.min_len, self.avg_len - self.jitter)
        hi = min(self.max_len, self.avg_len + self.jitter)
        return int(torch.randint(lo, hi + 1, (1,), generator=generator).item())


# =========================================================
# Helpers
# =========================================================

def _plain(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            return x.item()
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _mean_var_min_max(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
    x = torch.tensor(values, dtype=torch.float64)
    mu = x.mean()
    return {
        "count": int(x.numel()),
        "mean": float(mu.item()),
        "variance": float(((x - mu) ** 2).mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _word_to_tuple(x: torch.Tensor) -> Tuple[int, ...]:
    if x.numel() == 0:
        return ()
    return tuple(int(v) for v in x.detach().cpu().tolist())


def _tuple_to_word(x: Tuple[int, ...], device: str | torch.device = "cpu") -> torch.Tensor:
    if len(x) == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.tensor(list(x), dtype=torch.long, device=device)


def _normalize_family_weights(family_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if family_weights is None:
        family_weights = {"random": 1.0}
    clean = {k: 0.0 for k in FAMILY_NAME_TO_ID}
    for k, v in family_weights.items():
        if k not in clean:
            raise ValueError(f"Unknown family key: {k}")
        clean[k] = float(v)
    total = sum(max(0.0, v) for v in clean.values())
    if total <= 0.0:
        raise ValueError("At least one family weight must be positive.")
    return {k: float(max(0.0, v) / total) for k, v in clean.items()}


def _partition_ids_evenly(num_items: int, num_groups: int) -> torch.Tensor:
    ids = torch.empty((num_items,), dtype=torch.long)
    base = num_items // num_groups
    rem = num_items % num_groups
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        ids[start:start + size] = g
        start += size
    return ids


def _members_from_ids(ids: torch.Tensor, num_groups: int) -> List[List[int]]:
    ids_cpu = ids.detach().cpu()
    return [torch.where(ids_cpu == g)[0].tolist() for g in range(num_groups)]


def _torch_multinomial_from_probs(probs: List[float], generator: torch.Generator) -> int:
    p = torch.tensor(probs, dtype=torch.float64)
    return int(torch.multinomial(p, 1, generator=generator).item())


def _randint_exclusive(high: int, generator: torch.Generator) -> int:
    return int(torch.randint(0, high, (1,), generator=generator).item())


def _choose_uniform_from_list(items: List[int], generator: torch.Generator) -> int:
    return int(items[_randint_exclusive(len(items), generator)])


# =========================================================
# Tensor DFA
# =========================================================

class TensorDFALanguage:
    def __init__(
        self,
        num_states: int,
        alphabet_size: int,
        delta: Optional[torch.Tensor] = None,
        accept_mask: Optional[torch.Tensor] = None,
        reject_mask: Optional[torch.Tensor] = None,
        start_state: int = 0,
        device: str | torch.device = "cpu",
        generation_config: Optional[Dict[str, Any]] = None,
        transition_family: Optional[torch.Tensor] = None,
        cluster_id: Optional[torch.Tensor] = None,
        residue_id: Optional[torch.Tensor] = None,
        symbol_increments: Optional[torch.Tensor] = None,
        merge_symbols: Optional[torch.Tensor] = None,
        merge_targets: Optional[List[List[int]]] = None,
    ):
        assert num_states >= 2
        assert alphabet_size >= 1
        self.num_states = int(num_states)
        self.alphabet_size = int(alphabet_size)
        self.start_state = int(start_state)
        self.device = torch.device(device)

        if delta is None:
            self.delta = torch.full((self.num_states, self.alphabet_size), -1, dtype=torch.long, device=self.device)
        else:
            self.delta = delta.to(device=self.device, dtype=torch.long).clone()

        if accept_mask is None:
            accept_mask = torch.zeros(self.num_states, dtype=torch.bool, device=self.device)
            accept_mask[: max(1, self.num_states // 2)] = True
        else:
            accept_mask = accept_mask.to(device=self.device, dtype=torch.bool).clone()
        if reject_mask is None:
            reject_mask = ~accept_mask
        else:
            reject_mask = reject_mask.to(device=self.device, dtype=torch.bool).clone()

        self.accept_mask = accept_mask
        self.reject_mask = reject_mask
        if transition_family is None:
            transition_family = torch.full((self.num_states, self.alphabet_size), -1, dtype=torch.long)
        self.transition_family = transition_family.to(device=self.device, dtype=torch.long).clone()
        self.cluster_id = None if cluster_id is None else cluster_id.to(device=self.device, dtype=torch.long).clone()
        self.residue_id = None if residue_id is None else residue_id.to(device=self.device, dtype=torch.long).clone()
        self.symbol_increments = None if symbol_increments is None else symbol_increments.to(device=self.device, dtype=torch.long).clone()
        self.merge_symbols = None if merge_symbols is None else merge_symbols.to(device=self.device, dtype=torch.bool).clone()
        self.merge_targets = merge_targets
        self.generation_config = generation_config if generation_config is not None else {}
        self.valid_mask = self.delta >= 0
        self.out_valid_sym_mask = self.valid_mask.clone()

        self.refresh_sampling_cache()

    def refresh_sampling_cache(self) -> None:
        # Fast sampling cache: for each state, compact list of defined symbols.
        self.valid_symbol_counts = self.out_valid_sym_mask.sum(dim=1).long()
        max_valid = int(self.valid_symbol_counts.max().item()) if self.num_states > 0 else 0
        self.valid_symbols = torch.full(
            (self.num_states, max_valid),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for q in range(self.num_states):
            syms = torch.where(self.out_valid_sym_mask[q])[0]
            if syms.numel() > 0:
                self.valid_symbols[q, : syms.numel()] = syms

    @torch.no_grad()
    def run_details(self, seq: torch.Tensor, start_state: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = seq.to(self.device)
        s = torch.tensor(self.start_state if start_state is None else int(start_state), dtype=torch.long, device=self.device)
        valid_len = 0
        for t in range(seq.numel()):
            ns = self.delta[s, seq[t]]
            if ns < 0:
                return s, torch.tensor(False, dtype=torch.bool, device=self.device), torch.tensor(valid_len, dtype=torch.long, device=self.device)
            s = ns
            valid_len += 1
        return s, torch.tensor(True, dtype=torch.bool, device=self.device), torch.tensor(valid_len, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def run(self, seq: torch.Tensor, start_state: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        terminal_state, valid, _ = self.run_details(seq, start_state=start_state)
        return terminal_state, valid

    def label_of_state(self, state: int | torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.long, device=self.device)
        return self.accept_mask[s].long()

    @torch.no_grad()
    def run_details_batch(
        self,
        seqs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = seqs.to(self.device)
        lengths = lengths.to(self.device).long()
        B, T = seqs.shape
        cur = torch.full((B,), int(self.start_state), dtype=torch.long, device=self.device)
        valid = torch.ones((B,), dtype=torch.bool, device=self.device)
        valid_lengths = torch.zeros((B,), dtype=torch.long, device=self.device)
        for t in range(T):
            active = valid & (lengths > t)
            if not bool(active.any().item()):
                continue
            tok = seqs[active, t]
            ns = self.delta[cur[active], tok]
            ok = ns >= 0
            active_idx = torch.where(active)[0]
            if bool(ok.any().item()):
                good_idx = active_idx[ok]
                cur[good_idx] = ns[ok]
                valid_lengths[good_idx] += 1
            if bool((~ok).any().item()):
                bad_idx = active_idx[~ok]
                valid[bad_idx] = False
        return cur, valid, valid_lengths

    @torch.no_grad()
    def run_details_batch_from_states(
        self,
        seqs: torch.Tensor,
        lengths: torch.Tensor,
        start_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = seqs.to(self.device)
        lengths = lengths.to(self.device).long()
        start_states = start_states.to(self.device).long()
        B, T = seqs.shape
        cur = start_states.clone()
        valid = torch.ones((B,), dtype=torch.bool, device=self.device)
        valid_lengths = torch.zeros((B,), dtype=torch.long, device=self.device)
        for t in range(T):
            active = valid & (lengths > t)
            if not bool(active.any().item()):
                continue
            tok = seqs[active, t]
            ns = self.delta[cur[active], tok]
            ok = ns >= 0
            active_idx = torch.where(active)[0]
            if bool(ok.any().item()):
                good_idx = active_idx[ok]
                cur[good_idx] = ns[ok]
                valid_lengths[good_idx] += 1
            if bool((~ok).any().item()):
                bad_idx = active_idx[~ok]
                valid[bad_idx] = False
        return cur, valid, valid_lengths


# =========================================================
# Constructors
# =========================================================

def make_structured_tensor_dfa(
    num_states: int,
    alphabet_size: int,
    density: float = 0.15,
    accept_prob: float = 0.5,
    family_weights: Optional[Dict[str, float]] = None,
    clustered_params: Optional[Dict[str, Any]] = None,
    modular_params: Optional[Dict[str, Any]] = None,
    merge_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> TensorDFALanguage:
    device = torch.device(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    family_weights = _normalize_family_weights(family_weights)
    clustered_params = dict(clustered_params or {})
    modular_params = dict(modular_params or {})
    merge_params = dict(merge_params or {})

    cluster_count = int(clustered_params.get("cluster_count", max(2, min(num_states, max(2, num_states // 8)))))
    cluster_count = max(1, min(cluster_count, num_states))
    cluster_bias = float(clustered_params.get("cluster_bias", 0.8))
    self_loop_bias = float(clustered_params.get("self_loop_bias", 0.1))
    cluster_id = _partition_ids_evenly(num_states, cluster_count)
    cluster_members = _members_from_ids(cluster_id, cluster_count)

    modulus_k = int(modular_params.get("modulus_k", max(2, min(8, num_states))))
    modulus_k = max(1, min(modulus_k, num_states))
    modular_bias = float(modular_params.get("modular_bias", 0.9))
    residue_id = _partition_ids_evenly(num_states, modulus_k)
    residue_members = _members_from_ids(residue_id, modulus_k)

    symbol_increments_in = modular_params.get("symbol_increments", None)
    if symbol_increments_in is None:
        symbol_increments = torch.randint(0, modulus_k, (alphabet_size,), generator=g, dtype=torch.long)
    else:
        symbol_increments = torch.tensor(symbol_increments_in, dtype=torch.long) % modulus_k

    merge_size = int(merge_params.get("merge_size", max(1, min(4, num_states))))
    merge_size = max(1, min(merge_size, num_states))
    merge_bias = float(merge_params.get("merge_bias", 0.9))
    merge_symbol_fraction = float(merge_params.get("merge_symbol_fraction", 1.0))
    merge_symbol_count = int(round(max(0.0, min(1.0, merge_symbol_fraction)) * alphabet_size))
    merge_symbols = torch.zeros(alphabet_size, dtype=torch.bool)
    if merge_symbol_count > 0:
        perm = torch.randperm(alphabet_size, generator=g)
        merge_symbols[perm[:merge_symbol_count]] = True

    merge_targets: List[List[int]] = [[] for _ in range(alphabet_size)]
    for a in range(alphabet_size):
        if bool(merge_symbols[a].item()):
            perm = torch.randperm(num_states, generator=g)
            merge_targets[a] = perm[:merge_size].tolist()

    delta = torch.full((num_states, alphabet_size), -1, dtype=torch.long)
    transition_family = torch.full((num_states, alphabet_size), -1, dtype=torch.long)
    family_order = ["random", "clustered", "modular", "merge"]
    family_probs = [family_weights[name] for name in family_order]

    for q in range(num_states):
        q_cluster = int(cluster_id[q].item())
        q_res = int(residue_id[q].item())
        for a in range(alphabet_size):
            if float(torch.rand((1,), generator=g).item()) >= density:
                continue
            fam_idx = _torch_multinomial_from_probs(family_probs, generator=g)
            transition_family[q, a] = fam_idx
            if fam_idx == FAMILY_RANDOM:
                dst = _randint_exclusive(num_states, g)
            elif fam_idx == FAMILY_CLUSTERED:
                u = float(torch.rand((1,), generator=g).item())
                if u < self_loop_bias:
                    dst = q
                elif u < self_loop_bias + cluster_bias:
                    dst = _choose_uniform_from_list(cluster_members[q_cluster], g)
                else:
                    dst = _randint_exclusive(num_states, g)
            elif fam_idx == FAMILY_MODULAR:
                u = float(torch.rand((1,), generator=g).item())
                if u < modular_bias:
                    tgt_res = int((q_res + int(symbol_increments[a].item())) % modulus_k)
                    dst = _choose_uniform_from_list(residue_members[tgt_res], g)
                else:
                    dst = _randint_exclusive(num_states, g)
            elif fam_idx == FAMILY_MERGE:
                use_merge = bool(merge_symbols[a].item()) and (float(torch.rand((1,), generator=g).item()) < merge_bias)
                if use_merge and len(merge_targets[a]) > 0:
                    dst = _choose_uniform_from_list(merge_targets[a], g)
                else:
                    dst = _randint_exclusive(num_states, g)
            else:
                raise RuntimeError(f"Unknown family id: {fam_idx}")
            delta[q, a] = int(dst)

    accept_mask = torch.rand((num_states,), generator=g) < accept_prob
    if bool(accept_mask.all().item()):
        accept_mask[-1] = False
    if bool((~accept_mask).all().item()):
        accept_mask[0] = True
    reject_mask = ~accept_mask

    generation_config = {
        "density": float(density),
        "accept_prob": float(accept_prob),
        "family_weights": dict(family_weights),
        "modular_params": {"modulus_k": int(modulus_k)},
    }

    return TensorDFALanguage(
        num_states=num_states,
        alphabet_size=alphabet_size,
        delta=delta.to(device),
        accept_mask=accept_mask.to(device),
        reject_mask=reject_mask.to(device),
        start_state=0,
        device=device,
        generation_config=generation_config,
        transition_family=transition_family.to(device),
        cluster_id=cluster_id.to(device),
        residue_id=residue_id.to(device),
        symbol_increments=symbol_increments.to(device),
        merge_symbols=merge_symbols.to(device),
        merge_targets=merge_targets,
    )


# =========================================================
# DFA utilities
# =========================================================

def compute_forward_path_counts(dfa: TensorDFALanguage, max_len: int) -> torch.Tensor:
    counts = torch.zeros((max_len + 1, dfa.num_states), dtype=torch.float64, device=dfa.device)
    counts[0, dfa.start_state] = 1.0
    for t in range(1, max_len + 1):
        prev = counts[t - 1]
        active_src = torch.where(prev > 0)[0]
        if active_src.numel() == 0:
            continue
        dst = dfa.delta[active_src]
        valid = dst >= 0
        if not bool(valid.any().item()):
            continue
        src_weights = prev[active_src].unsqueeze(1).expand_as(dst)
        counts[t].scatter_add_(0, dst[valid], src_weights[valid])
    return counts


def compute_forward_reachability(dfa: TensorDFALanguage, max_len: int) -> torch.Tensor:
    reach = torch.zeros((max_len + 1, dfa.num_states), dtype=torch.bool, device=dfa.device)
    reach[0, dfa.start_state] = True
    for t in range(1, max_len + 1):
        prev = reach[t - 1]
        active_src = torch.where(prev)[0]
        if active_src.numel() == 0:
            continue
        dst = dfa.delta[active_src]
        valid = dst >= 0
        if not bool(valid.any().item()):
            continue
        reach[t, dst[valid]] = True
    return reach


def run_valid_seq_with_states(dfa: TensorDFALanguage, seq: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    seq = seq.to(dfa.device)
    L = int(seq.numel())
    states_before = torch.empty((L,), dtype=torch.long, device=dfa.device)
    states_after = torch.empty((L,), dtype=torch.long, device=dfa.device)
    s = torch.tensor(dfa.start_state, dtype=torch.long, device=dfa.device)
    for t in range(L):
        states_before[t] = s
        ns = dfa.delta[s, seq[t]]
        if ns < 0:
            return None
        states_after[t] = ns
        s = ns
    return states_before, states_after


def compatible_start_states_for_suffix_prefixes(dfa: TensorDFALanguage, suffix_tokens: torch.Tensor) -> List[torch.Tensor]:
    suffix_tokens = suffix_tokens.to(dfa.device)
    S = dfa.num_states
    cur = torch.arange(S, dtype=torch.long, device=dfa.device)
    alive = torch.ones((S,), dtype=torch.bool, device=dfa.device)
    comps: List[torch.Tensor] = []
    for token in suffix_tokens:
        ns = torch.full((S,), -1, dtype=torch.long, device=dfa.device)
        if bool(alive.any().item()):
            ns[alive] = dfa.delta[cur[alive], token]
        alive = alive & (ns >= 0)
        cur = torch.where(alive, ns, cur)
        comps.append(alive.clone())
    return comps


# =========================================================
# Characteristic sample helpers
# =========================================================

def shortest_access_strings(dfa: TensorDFALanguage) -> Dict[int, torch.Tensor]:
    access: Dict[int, torch.Tensor] = {int(dfa.start_state): torch.empty((0,), dtype=torch.long, device=dfa.device)}
    q = deque([int(dfa.start_state)])
    while q:
        s = q.popleft()
        prefix = access[s]
        for a in range(dfa.alphabet_size):
            ns = int(dfa.delta[s, a].item())
            if ns < 0 or ns in access:
                continue
            access[ns] = torch.cat([prefix, torch.tensor([a], dtype=torch.long, device=dfa.device)], dim=0)
            q.append(ns)
    return access


def characteristic_outcome_from_state(dfa: TensorDFALanguage, state: int, suffix: torch.Tensor, task_type: str) -> int:
    terminal_state, valid, _ = dfa.run_details(suffix, start_state=state)
    if task_type == TASK_ACCEPTNESS:
        if not bool(valid.item()):
            return 2
        return int(dfa.label_of_state(terminal_state).item())
    raise ValueError(f"Unsupported characteristic task_type: {task_type}")


def _completed_step_with_sink(dfa: TensorDFALanguage, state: int, symbol: int, sink_state: int) -> int:
    if state == sink_state:
        return sink_state
    ns = int(dfa.delta[state, symbol].item())
    return sink_state if ns < 0 else ns


def _characteristic_state_output_with_sink(dfa: TensorDFALanguage, state: int, sink_state: int) -> int:
    if state == sink_state:
        return 2
    return int(dfa.accept_mask[state].long().item())


def shortest_distinguishing_suffix_bounded(dfa: TensorDFALanguage, state_a: int, state_b: int, max_suffix_len: int) -> Optional[torch.Tensor]:
    sink = dfa.num_states

    def norm_pair(x: int, y: int) -> Tuple[int, int]:
        return (x, y) if x <= y else (y, x)

    start = norm_pair(int(state_a), int(state_b))
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    parent_sym: Dict[Tuple[int, int], Optional[int]] = {start: None}
    depth: Dict[Tuple[int, int], int] = {start: 0}
    q = deque([start])
    while q:
        u, v = q.popleft()
        d = depth[(u, v)]
        if _characteristic_state_output_with_sink(dfa, u, sink) != _characteristic_state_output_with_sink(dfa, v, sink):
            rev: List[int] = []
            cur = (u, v)
            while parent[cur] is not None:
                rev.append(int(parent_sym[cur]))
                cur = parent[cur]
            rev.reverse()
            return torch.tensor(rev, dtype=torch.long, device=dfa.device)
        if d == max_suffix_len:
            continue
        for a in range(dfa.alphabet_size):
            nu = _completed_step_with_sink(dfa, u, a, sink)
            nv = _completed_step_with_sink(dfa, v, a, sink)
            nxt = norm_pair(nu, nv)
            if nxt in parent:
                continue
            parent[nxt] = (u, v)
            parent_sym[nxt] = int(a)
            depth[nxt] = d + 1
            q.append(nxt)
    return None


def pair_distinguished_by_suffix(dfa: TensorDFALanguage, pair: Tuple[int, int], suffix_tuple: Tuple[int, ...]) -> bool:
    s1, s2 = int(pair[0]), int(pair[1])
    suffix = _tuple_to_word(suffix_tuple, device=dfa.device)
    return characteristic_outcome_from_state(dfa, s1, suffix, TASK_ACCEPTNESS) != characteristic_outcome_from_state(dfa, s2, suffix, TASK_ACCEPTNESS)


def _solve_suffix_cover_branch_and_bound(candidate_suffixes: List[Tuple[int, ...]], cover_sets: List[set[int]], universe_size: int) -> List[int]:
    if universe_size == 0:
        return []
    candidate_ids = [j for j in range(len(candidate_suffixes)) if len(cover_sets[j]) > 0]
    if not candidate_ids:
        raise RuntimeError("No candidate suffixes cover the distinguishable pairs.")
    candidate_ids.sort(key=lambda j: (-len(cover_sets[j]), len(candidate_suffixes[j]), candidate_suffixes[j]))
    suffixes = [candidate_suffixes[j] for j in candidate_ids]
    covers = [set(cover_sets[j]) for j in candidate_ids]
    full_universe = set(range(universe_size))

    def greedy_upper_bound(remaining: set[int]) -> Optional[List[int]]:
        rem = set(remaining)
        chosen: List[int] = []
        available = set(range(len(suffixes)))
        while rem:
            best_j = None
            best_gain = 0
            for j in available:
                gain = len(covers[j] & rem)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j is None or best_gain == 0:
                return None
            chosen.append(best_j)
            rem -= covers[best_j]
            available.remove(best_j)
        return chosen

    best = greedy_upper_bound(full_universe)
    if best is None:
        raise RuntimeError("Failed to find an initial suffix cover.")

    def lower_bound(remaining: set[int], available: List[int]) -> int:
        if not remaining:
            return 0
        max_gain = max((len(covers[j] & remaining) for j in available), default=0)
        if max_gain <= 0:
            return 10**9
        return int(math.ceil(len(remaining) / max_gain))

    def dfs(remaining: set[int], available: List[int], chosen: List[int]) -> None:
        nonlocal best
        if not remaining:
            if best is None or len(chosen) < len(best):
                best = list(chosen)
            return
        if best is not None and len(chosen) >= len(best):
            return
        if best is not None and len(chosen) + lower_bound(remaining, available) >= len(best):
            return
        hardest_opts = None
        for u in sorted(remaining):
            opts = [j for j in available if u in covers[j]]
            if hardest_opts is None or len(opts) < len(hardest_opts):
                hardest_opts = opts
        if not hardest_opts:
            return
        hardest_opts.sort(key=lambda j: (-len(covers[j] & remaining), len(suffixes[j]), suffixes[j]))
        for j in hardest_opts:
            chosen.append(j)
            dfs(remaining - covers[j], [k for k in available if k > j], chosen)
            chosen.pop()

    dfs(full_universe, list(range(len(suffixes))), [])
    assert best is not None
    return [candidate_ids[j] for j in best]


def build_characteristic_suffix_set(dfa: TensorDFALanguage, max_suffix_len: int = 12, include_epsilon: bool = True) -> List[torch.Tensor]:
    access = shortest_access_strings(dfa)
    reachable_states = sorted(int(s) for s in access.keys())
    pairs: List[Tuple[int, int]] = []
    shortest: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    for i in range(len(reachable_states)):
        for j in range(i + 1, len(reachable_states)):
            p = int(reachable_states[i])
            q = int(reachable_states[j])
            w = shortest_distinguishing_suffix_bounded(dfa, p, q, max_suffix_len)
            if w is None:
                continue
            pair = (p, q)
            pairs.append(pair)
            shortest[pair] = _word_to_tuple(w)
    suffixes: List[Tuple[int, ...]] = []
    epsilon: Tuple[int, ...] = ()
    remaining_pairs: List[Tuple[int, int]] = []
    if include_epsilon:
        suffixes.append(epsilon)
    for pair in pairs:
        if include_epsilon and pair_distinguished_by_suffix(dfa, pair, epsilon):
            continue
        remaining_pairs.append(pair)
    if remaining_pairs:
        candidate_pool = sorted(set(shortest[pair] for pair in remaining_pairs), key=lambda x: (len(x), x))
        pair_index = {pair: i for i, pair in enumerate(remaining_pairs)}
        cover_sets: List[set[int]] = []
        for suf in candidate_pool:
            covered = set()
            for pair in remaining_pairs:
                if pair_distinguished_by_suffix(dfa, pair, suf):
                    covered.add(pair_index[pair])
            cover_sets.append(covered)
        chosen_ids = _solve_suffix_cover_branch_and_bound(candidate_pool, cover_sets, len(remaining_pairs))
        suffixes.extend(sorted([candidate_pool[j] for j in chosen_ids], key=lambda x: (len(x), x)))
    return [_tuple_to_word(x, device=dfa.device) for x in suffixes]


def build_characteristic_prefix_basis(dfa: TensorDFALanguage) -> Tuple[List[torch.Tensor], Dict[int, torch.Tensor], List[int]]:
    access = shortest_access_strings(dfa)
    reachable_states = sorted(int(s) for s in access.keys())
    seen: set[Tuple[int, ...]] = set()
    prefixes: List[torch.Tensor] = []

    def add_word(w: torch.Tensor) -> None:
        key = _word_to_tuple(w)
        if key in seen:
            return
        seen.add(key)
        prefixes.append(w.detach().clone())

    for s in reachable_states:
        add_word(access[s])
    for s in reachable_states:
        p = access[s]
        for a in range(dfa.alphabet_size):
            add_word(torch.cat([p, torch.tensor([a], dtype=torch.long, device=dfa.device)], dim=0))
    prefixes = sorted(prefixes, key=lambda x: (int(x.numel()), _word_to_tuple(x)))
    return prefixes, access, reachable_states


# =========================================================
# Reports
# =========================================================

def report_dfa(dfa: TensorDFALanguage, max_len: int = 40) -> Dict[str, Any]:
    S = dfa.num_states
    A = dfa.alphabet_size
    valid = dfa.valid_mask
    defined_transition_count = int(valid.sum().item())
    total_transition_slots = int(S * A)
    undefined_transition_count = total_transition_slots - defined_transition_count
    out_deg = valid.sum(dim=1).to(torch.long).detach().cpu().tolist()
    indeg = [0] * S
    src_idx, sym_idx = torch.where(valid)
    dst_idx = dfa.delta[src_idx, sym_idx]
    for d in dst_idx.detach().cpu().tolist():
        indeg[int(d)] += 1
    reach = compute_forward_reachability(dfa, max_len=max_len)
    reachable_any = reach.any(dim=0)
    family_counts = {name: 0 for name in FAMILY_NAME_TO_ID}
    tf_cpu = dfa.transition_family.detach().cpu()
    valid_cpu = dfa.valid_mask.detach().cpu()
    for fam_name, fam_id in FAMILY_NAME_TO_ID.items():
        family_counts[fam_name] = int(((tf_cpu == fam_id) & valid_cpu).sum().item())
    return {
        "identity": {
            "num_states": int(S),
            "alphabet_size": int(A),
            "start_state": int(dfa.start_state),
            "accept_state_count": int(dfa.accept_mask.sum().item()),
            "reject_state_count": int(dfa.reject_mask.sum().item()),
        },
        "generation": _plain(dfa.generation_config),
        "assignment": {
            "defined_transition_count": int(defined_transition_count),
            "undefined_transition_count": int(undefined_transition_count),
            "total_transition_slots": int(total_transition_slots),
            "family_counts": family_counts,
        },
        "graph": {
            "transition_density": 0.0 if total_transition_slots == 0 else float(defined_transition_count / total_transition_slots),
            "out_degree": _mean_var_min_max([float(x) for x in out_deg]),
            "in_degree": _mean_var_min_max([float(x) for x in indeg]),
        },
        "reachability": {
            "max_len_used": int(max_len),
            "reachable_states_any_length_le_max_count": int(reachable_any.sum().item()),
        },
    }
