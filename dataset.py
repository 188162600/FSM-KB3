# dataset.py
#
# New dataset.py that:
# - Builds a TOTAL DFA with an absorbing sink.
# - Defines language: in-language iff NEVER hits sink (equivalently final != sink because sink absorbing).
# - Provides datasets returning List[(seq, tag_set)].
#
# Datasets included:
#   1) AcceptDataset: sequences that avoid sink completely (accept)
#   2) RandomGeneratedRejectDataset: random sequences filtered to be reject (hits sink somewhere)
#   3) RejectAvgSinkHitDataset: rejects where sink is forced to happen at a target PERCENT of length
#       pct in {0.25,0.5,0.75,1.0} typically.
#       Construction:
#         - Choose total length L from LengthPolicy
#         - Choose sink hit step k = clamp(round(pct*L), 1..L)
#         - Build a prefix of length k-1 that stays non-sink from start_state
#         - Choose one symbol that sends current state -> sink
#         - Append random tail to reach length L (still sink from step k onwards)
#
# IMPORTANT:
# - This RejectAvgSinkHitDataset GUARANTEES samples are runs from start_state (prefix is built from start_state).
# - It also tags the exact sink hit step: "sink_hit_step_<k>".
# - main() performs strong verification:
#     * final state is sink
#     * computed sink-hit index equals the tag k
#     * prefix up to k-1 never hit sink
#   And prints average hit/len per pct config.

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

Sample = Tuple[List[int], Set[str]]


# =========================================================
# Total DFA with absorbing sink
# =========================================================

class TotalDFASinkLanguage:
    def __init__(self, num_states: int, alphabet_size: int, sink_state: int):
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.sink_state = sink_state

        # Total transition function; default -> sink
        self.delta: List[List[int]] = [
            [sink_state for _ in range(alphabet_size)] for _ in range(num_states)
        ]

        # indexes for generation
        self.out_syms_nonsink: List[List[int]] = [[] for _ in range(num_states)]  # state -> symbols not going to sink
        self.sink_syms: List[List[int]] = [[] for _ in range(num_states)]         # state -> symbols going to sink

    def rebuild_indexes(self) -> None:
        self.out_syms_nonsink = [[] for _ in range(self.num_states)]
        self.sink_syms = [[] for _ in range(self.num_states)]

        for s in range(self.num_states):
            if s == self.sink_state:
                continue
            row = self.delta[s]
            for a in range(self.alphabet_size):
                ns = row[a]
                if ns == self.sink_state:
                    self.sink_syms[s].append(a)
                else:
                    self.out_syms_nonsink[s].append(a)

    def step(self, state: int, sym: int) -> int:
        return self.delta[state][sym]

    def run(self, start_state: int, seq: List[int]) -> int:
        s = start_state
        for a in seq:
            s = self.delta[s][a]
        return s

    def in_language(self, start_state: int, seq: List[int]) -> bool:
        return self.run(start_state, seq) != self.sink_state

    def run_until_sink(self, start_state: int, seq: List[int]) -> Tuple[int, int]:
        """
        Returns (final_state, sink_hit_index).
        sink_hit_index is 1-based first step entering sink;
        if never hits sink => len(seq);
        if seq empty => 0.
        """
        s = start_state
        for i, a in enumerate(seq, start=1):
            s = self.delta[s][a]
            if s == self.sink_state:
                return s, i
        return s, len(seq)

    def random_word(self, length: int) -> List[int]:
        return [random.randrange(self.alphabet_size) for _ in range(length)]

    def walk_avoid_sink_exact(self, start_state: int, length: int, max_tries: int = 500) -> Optional[List[int]]:
        """
        Produce a non-sink-only path of exact length from start_state.
        Returns None if stuck.
        """
        for _ in range(max_tries):
            s = start_state
            out: List[int] = []
            ok = True
            for _t in range(length):
                choices = self.out_syms_nonsink[s]
                if not choices:
                    ok = False
                    break
                a = random.choice(choices)
                out.append(a)
                s = self.delta[s][a]
                if s == self.sink_state:
                    ok = False
                    break
            if ok and len(out) == length:
                return out
        return None


def make_sparse_total_dfa_with_sink(
    num_states: int = 100,
    alphabet_size: int = 100,
    non_sink_transitions: int = 1000,
    seed: int = 42,
) -> TotalDFASinkLanguage:
    """
    Total DFA:
      - sink = num_states - 1
      - default goes to sink
      - add `non_sink_transitions` transitions (src!=sink) to dst!=sink
      - sink loops to sink
    """
    random.seed(seed)
    sink = num_states - 1
    dfa = TotalDFASinkLanguage(num_states, alphabet_size, sink_state=sink)

    # Ensure sink loops
    for a in range(alphabet_size):
        dfa.delta[sink][a] = sink

    used: Set[Tuple[int, int]] = set()
    while len(used) < non_sink_transitions:
        src = random.randrange(num_states - 1)
        sym = random.randrange(alphabet_size)
        if (src, sym) in used:
            continue
        dst = random.randrange(num_states - 1)  # non-sink
        dfa.delta[src][sym] = dst
        used.add((src, sym))

    dfa.rebuild_indexes()
    return dfa


# =========================================================
# Length controller
# =========================================================

@dataclass
class LengthPolicy:
    min_len: int = 5
    max_len: int = 40
    avg_len: int = 20
    jitter: int = 15
    uniform: bool = False

    def sample(self) -> int:
        if self.uniform:
            return random.randint(self.min_len, self.max_len)
        lo = max(self.min_len, self.avg_len - self.jitter)
        hi = min(self.max_len, self.avg_len + self.jitter)
        return random.randint(lo, hi)


# =========================================================
# Dataset base
# =========================================================

class Dataset:
    def __init__(self, labels: Optional[Set[str]] = None):
        self.labels = set(labels) if labels is not None else None

    def _labels(self, default: Set[str]) -> Set[str]:
        return set(self.labels) if self.labels is not None else set(default)

    def generate(self, n: int, length_policy: LengthPolicy) -> List[Sample]:
        raise NotImplementedError


# =========================================================
# Accept dataset
# =========================================================

class AcceptDataset(Dataset):
    """accept: avoid sink entirely."""
    def __init__(self, dfa: TotalDFASinkLanguage, start_state: int = 0, labels: Optional[Set[str]] = None):
        super().__init__(labels)
        self.dfa = dfa
        self.start_state = start_state

    def generate(self, n: int, length_policy: LengthPolicy) -> List[Sample]:
        lab = self._labels({"accept"})
        out: List[Sample] = []
        while len(out) < n:
            L = length_policy.sample()
            seq = self.dfa.walk_avoid_sink_exact(self.start_state, L)
            if seq is None:
                continue
            if len(seq) >= length_policy.min_len and self.dfa.in_language(self.start_state, seq):
                out.append((seq, lab.copy()))
        return out


# =========================================================
# Random reject dataset
# =========================================================

class RandomGeneratedRejectDataset(Dataset):
    """
    reject: random words filtered to sink (sink-anywhere).
    """
    def __init__(self, dfa: TotalDFASinkLanguage, start_state: int = 0, labels: Optional[Set[str]] = None):
        super().__init__(labels)
        self.dfa = dfa
        self.start_state = start_state

    def generate(self, n: int, length_policy: LengthPolicy) -> List[Sample]:
        lab = self._labels({"reject", "sink_anywhere", "random_generated"})
        out: List[Sample] = []
        while len(out) < n:
            L = length_policy.sample()
            seq = self.dfa.random_word(L)
            if not self.dfa.in_language(self.start_state, seq):
                final, hit = self.dfa.run_until_sink(self.start_state, seq)
                tags = lab.copy()
                tags.add(f"sink_hit_step_{hit}")
                out.append((seq, tags))
        return out


# =========================================================
# Reject dataset with target sink-hit percent
# =========================================================

class RejectAvgSinkHitDataset(Dataset):
    """
    reject: force sink at approximately pct of length.

    pct=1.0 means sink happens at the last token (k=L)
    pct=0.25 means sink happens early (k≈L/4)

    Tags:
      - "reject"
      - "sink_anywhere"
      - f"sink_hit_pct_{pct}"
      - f"sink_hit_step_{k}"
    """
    def __init__(
        self,
        dfa: TotalDFASinkLanguage,
        start_state: int = 0,
        pct: float = 0.5,
        labels: Optional[Set[str]] = None,
        max_tries: int = 2000,
        verbose_stats: bool = False,
    ):
        super().__init__(labels)
        self.dfa = dfa
        self.start_state = start_state
        self.pct = float(pct)
        self.max_tries = max_tries
        self.verbose_stats = verbose_stats

    def _target_k(self, L: int) -> int:
        k = int(round(self.pct * L))
        if k < 1:
            k = 1
        if k > L:
            k = L
        return k

    def _make_one(self, L: int) -> Optional[Tuple[List[int], int]]:
        """
        Construct seq of length L that hits sink exactly at step k=target_k(L).
        Returns (seq, k) or None.
        """
        k = self._target_k(L)
        prefix_len = k - 1

        # Build prefix of exact length prefix_len that avoids sink from start_state
        prefix = self.dfa.walk_avoid_sink_exact(self.start_state, prefix_len, max_tries=300)
        if prefix is None:
            return None

        # compute state after prefix (must be non-sink)
        s = self.dfa.run(self.start_state, prefix)
        if s == self.dfa.sink_state:
            return None

        # Need a symbol that goes to sink at this state
        sink_choices = self.dfa.sink_syms[s]
        if not sink_choices:
            return None
        a_sink = random.choice(sink_choices)

        # Tail after sink can be anything (sink absorbing)
        tail_len = L - k
        tail = self.dfa.random_word(tail_len) if tail_len > 0 else []

        seq = prefix + [a_sink] + tail

        # Verify it hits sink exactly at k from start_state
        final, hit = self.dfa.run_until_sink(self.start_state, seq)
        if final != self.dfa.sink_state or hit != k:
            return None

        # Also verify prefix never hit sink (redundant, but strong)
        if prefix_len > 0:
            p_final, p_hit = self.dfa.run_until_sink(self.start_state, prefix)
            if p_final == self.dfa.sink_state:
                return None

        return seq, k

    def generate(self, n: int, length_policy: LengthPolicy) -> List[Sample]:
        lab = self._labels({"reject", "sink_anywhere", f"sink_hit_pct_{self.pct}"})
        out: List[Sample] = []
        attempts = 0

        while len(out) < n:
            attempts += 1
            if attempts > self.max_tries * max(1, n):
                # If DFA is too sparse and we can't build enough, stop hard to avoid infinite loop
                break

            L = length_policy.sample()
            made = self._make_one(L)
            if made is None:
                continue
            seq, k = made
            tags = lab.copy()
            tags.add(f"sink_hit_step_{k}")
            out.append((seq, tags))

        if self.verbose_stats and out:
            avg_hit = sum(int(next(int(t.split("_")[-1]) for t in tags if t.startswith("sink_hit_step_")))
                          for _, tags in out) / len(out)
            avg_len = sum(len(s) for s, _ in out) / len(out)
            print(f"[RejectAvgSinkHitDataset pct={self.pct}] n={len(out)} avg_len={avg_len:.2f} avg_hit={avg_hit:.2f}")

        return out


# =========================================================
# Main: verification + stats
# =========================================================

def _extract_k(tags: Set[str]) -> Optional[int]:
    for t in tags:
        if t.startswith("sink_hit_step_"):
            try:
                return int(t.split("_")[-1])
            except Exception:
                return None
    return None


def verify_samples_start_state(dfa: TotalDFASinkLanguage, start_state: int, samples: List[Sample]) -> Dict[str, int]:
    """
    Verify (for rejects):
      - running from start_state hits sink
      - computed hit == tag k
      - prefix up to k-1 never hits sink
    """
    ok = 0
    bad_final = 0
    bad_hit = 0
    bad_prefix = 0
    no_k = 0

    for seq, tags in samples:
        if "accept" in tags:
            # accepts should be in-language from start_state
            if dfa.in_language(start_state, seq):
                ok += 1
            else:
                bad_final += 1
            continue

        # rejects
        k = _extract_k(tags)
        if k is None:
            no_k += 1
            continue

        final, hit = dfa.run_until_sink(start_state, seq)
        if final != dfa.sink_state:
            bad_final += 1
            continue
        if hit != k:
            bad_hit += 1
            continue

        # prefix check
        if k > 1:
            prefix = seq[:k-1]
            pf, _ph = dfa.run_until_sink(start_state, prefix)
            if pf == dfa.sink_state:
                bad_prefix += 1
                continue

        ok += 1

    return {"ok": ok, "bad_final": bad_final, "bad_hit": bad_hit, "bad_prefix": bad_prefix, "no_k": no_k}


def summarize_hit_by_dataset(samples: List[Sample]) -> Dict[str, Dict[str, float]]:
    """
    Returns per sink_hit_pct_* dataset:
      avg_len, avg_hit, avg_hit_frac
    """
    buckets: Dict[str, List[Tuple[int, int]]] = {}
    for seq, tags in samples:
        name = None
        for t in tags:
            if t.startswith("sink_hit_pct_") or t == "random_generated" or t == "accept":
                name = t
                break
        if name is None:
            name = "other"

        k = _extract_k(tags)
        if k is None:
            continue
        buckets.setdefault(name, []).append((len(seq), k))

    out: Dict[str, Dict[str, float]] = {}
    for name, pairs in buckets.items():
        avg_len = sum(L for L, _k in pairs) / len(pairs)
        avg_hit = sum(_k for _L, _k in pairs) / len(pairs)
        out[name] = {
            "n": float(len(pairs)),
            "avg_len": avg_len,
            "avg_hit": avg_hit,
            "avg_hit_frac": avg_hit / max(1e-9, avg_len),
        }
    return out


def main():
    SEED = 42
    random.seed(SEED)

    dfa = make_sparse_total_dfa_with_sink(
        num_states=100,
        alphabet_size=100,
        non_sink_transitions=1000,
        seed=SEED,
    )
    start = 0

    lp = LengthPolicy(min_len=5, max_len=40, avg_len=20, jitter=15, uniform=False)

    n_each = 500
    pcts = [0.25, 0.5, 0.75, 1.0]

    acc_ds = AcceptDataset(dfa, start)
    rg_ds = RandomGeneratedRejectDataset(dfa, start)

    samples: List[Sample] = []
    samples += acc_ds.generate(n_each, lp)
    samples += rg_ds.generate(n_each, lp)

    for pct in pcts:
        ds = RejectAvgSinkHitDataset(dfa, start, pct=pct, verbose_stats=False)
        samples += ds.generate(n_each, lp)

    # Verify
    stats = verify_samples_start_state(dfa, start, samples)
    print("==== Verification (start_state correctness) ====")
    print(stats)

    # Print avg hit length for different hit config
    print("\n==== Avg hit/len by dataset tag ====")
    summ = summarize_hit_by_dataset(samples)
    for name in sorted(summ.keys()):
        s = summ[name]
        print(f"{name:>18} | n={int(s['n']):4d} | avg_len={s['avg_len']:.2f} | avg_hit={s['avg_hit']:.2f} | avg_hit_frac={s['avg_hit_frac']:.3f}")

    # For pct buckets, print how close to target pct
    print("\n==== Avg hit frac vs target pct (pct buckets) ====")
    for pct in pcts:
        key = f"sink_hit_pct_{pct}"
        if key not in summ:
            print(f"{key:>18} | missing")
            continue
        avg_frac = summ[key]["avg_hit_frac"]
        print(f"{key:>18} | target={pct:.2f} | observed_avg_hit_frac={avg_frac:.3f} | diff={avg_frac - pct:+.3f}")


if __name__ == "__main__":
    main()
