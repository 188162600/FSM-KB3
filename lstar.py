# lstar_learn_dfa_from_graph_eq_modes_with_val_metrics.py
# ------------------------------------------------------------
# Angluin L* to learn language of VALID traces in a partial DFA graph.
#
# Changes requested:
#  1) Move counterexample finding into the Graph (graph returns CE).
#  2) Print validation true/false accuracy during learning.
#  3) Config: choose EQ mode:
#        - "exact": shortest CE by BFS on product automaton
#        - "random": randomized CE search (faster, not guaranteed)
#
# Notes:
#  - We treat the partial graph as a TOTAL DFA by adding a rejecting sink state.
#  - Language accepts strings that never enter sink (i.e., valid traces).
# ------------------------------------------------------------

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter, deque


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    seed: int = 123

    # Target graph (partial DFA)
    n_states: int = 100
    n_events: int = 100
    n_transitions: int = 9000
    start_state: int = 0

    # L* iteration controls
    max_lstar_iters: int = 10_000
    print_every_iter: bool = True

    # Equivalence query mode: "exact" or "random"
    eq_mode: str = "exact"

    # Random CE finder params (used if eq_mode="random")
    random_ce_trials: int = 50_000     # how many random strings to try per EQ
    random_ce_max_len: int = 80        # max length of random strings
    random_ce_mix_valid: float = 0.70  # prob sample a VALID-walk prefix, else uniform random string
    random_ce_corrupt_p: float = 0.25  # if sampling valid-walk, corrupt each token with this prob

    # Validation monitor (fixed set generated once)
    val_monitor_n_pos: int = 2000
    val_monitor_n_neg: int = 2000
    val_monitor_avg_len: int = 21
    val_monitor_min_len: int = 5
    val_monitor_max_len: int = 80
    val_monitor_corrupt_until_invalid: bool = True  # negatives: corrupt-until-invalid (harder)


# -----------------------------
# Hypothesis DFA
# -----------------------------
class DFA:
    def __init__(self, n_states: int, sigma_size: int):
        self.n_states = n_states
        self.sigma_size = sigma_size
        self.start = 0
        self.acc = [False] * n_states
        self.trans = [[0] * sigma_size for _ in range(n_states)]

    def delta(self, s: int, a: int) -> int:
        return self.trans[s][a]

    def accept(self, s: int) -> bool:
        return self.acc[s]


# -----------------------------
# Random labeled graph (partial DFA) + CE finder
# -----------------------------
class LabeledGraph:
    """
    Partial DFA graph:
      - delta[state][event] -> next_state (if exists), else missing.
    For learning, we add an implicit rejecting sink state (id = n_states).
    """
    def __init__(self, n_states: int, n_events: int, n_transitions: int, seed: int, start_state: int = 0):
        self.n_states = n_states
        self.n_events = n_events
        self.n_transitions = n_transitions
        self.start_state = start_state

        rng = random.Random(seed)
        self.delta: List[Dict[int, int]] = [dict() for _ in range(n_states)]

        # ensure at least 1 outgoing per state
        for s in range(n_states):
            e = rng.randrange(n_events)
            t = rng.randrange(n_states)
            self.delta[s][e] = t

        added = sum(len(m) for m in self.delta)
        while added < n_transitions:
            s = rng.randrange(n_states)
            e = rng.randrange(n_events)
            t = rng.randrange(n_states)
            if e in self.delta[s]:
                continue
            self.delta[s][e] = t
            added += 1

    @property
    def sink(self) -> int:
        return self.n_states

    @property
    def total_states(self) -> int:
        return self.n_states + 1

    @property
    def sigma(self) -> List[int]:
        return list(range(self.n_events))

    @property
    def avg_out(self) -> float:
        return self.n_transitions / float(self.n_states)

    def step(self, state: int, event: int) -> Optional[int]:
        return self.delta[state].get(event, None)

    def delta_total(self, state: int, event: int) -> int:
        if state == self.sink:
            return self.sink
        ns = self.step(state, event)
        return self.sink if ns is None else ns

    def accept_total(self, state: int) -> bool:
        return state != self.sink

    def is_valid(self, trace: List[int]) -> bool:
        s = self.start_state
        for ev in trace:
            ns = self.step(s, ev)
            if ns is None:
                return False
            s = ns
        return True

    def generate_valid_trace(self, rng: random.Random, length: int) -> List[int]:
        s = self.start_state
        out: List[int] = []
        for _ in range(length):
            keys = list(self.delta[s].keys())
            ev = keys[rng.randrange(len(keys))]
            out.append(ev)
            s = self.delta[s][ev]
        return out

    # ---------- Counterexample finders ----------
    def find_counterexample_exact(self, hyp: DFA) -> Optional[List[int]]:
        """
        Exact EQ via BFS on product automaton between:
          - target total DFA (graph + sink)
          - hypothesis DFA
        Returns shortest w s.t. target(w) != hyp(w), or None if equivalent.
        """
        start_pair = (self.start_state, hyp.start)
        q = deque([start_pair])
        prev: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
        visited = {start_pair}

        def rebuild(end_pair: Tuple[int, int]) -> List[int]:
            out = []
            cur = end_pair
            while cur in prev:
                p, a = prev[cur]
                out.append(a)
                cur = p
            out.reverse()
            return out

        # epsilon check
        if self.accept_total(start_pair[0]) != hyp.accept(start_pair[1]):
            return []

        while q:
            ts, hs = q.popleft()
            for a in self.sigma:
                ts2 = self.delta_total(ts, a)
                hs2 = hyp.delta(hs, a)
                pair2 = (ts2, hs2)
                if pair2 in visited:
                    continue
                visited.add(pair2)
                prev[pair2] = ((ts, hs), a)

                if self.accept_total(ts2) != hyp.accept(hs2):
                    return rebuild(pair2)
                q.append(pair2)

        return None

    def find_counterexample_random(
        self,
        hyp: DFA,
        rng: random.Random,
        trials: int,
        max_len: int,
        mix_valid: float,
        corrupt_p: float,
    ) -> Optional[List[int]]:
        """
        Random EQ: try random strings until disagreement found.
        Not guaranteed, but can be much faster than exact BFS in some settings.

        Strategy:
          - with prob mix_valid: sample a valid walk of length L, then corrupt tokens
          - else: sample uniform random length-L string
        """
        # epsilon check
        if self.accept_total(self.start_state) != hyp.accept(hyp.start):
            return []

        for _ in range(trials):
            L = rng.randrange(1, max_len + 1)
            if rng.random() < mix_valid:
                w = self.generate_valid_trace(rng, L)
                # corrupt
                for i in range(len(w)):
                    if rng.random() < corrupt_p:
                        old = w[i]
                        new = rng.randrange(self.n_events)
                        if new == old:
                            new = (new + 1) % self.n_events
                        w[i] = new
            else:
                w = [rng.randrange(self.n_events) for _ in range(L)]

            # evaluate target vs hyp
            ts = self.start_state
            hs = hyp.start
            # check prefix-by-prefix; return the earliest disagreeing prefix (helps L*)
            if self.accept_total(ts) != hyp.accept(hs):
                return []
            for i, a in enumerate(w):
                ts = self.delta_total(ts, a)
                hs = hyp.delta(hs, a)
                if self.accept_total(ts) != hyp.accept(hs):
                    return w[: i + 1]
        return None

    def find_counterexample(self, hyp: DFA, cfg: Config, rng: random.Random) -> Optional[List[int]]:
        if cfg.eq_mode == "exact":
            return self.find_counterexample_exact(hyp)
        if cfg.eq_mode == "random":
            return self.find_counterexample_random(
                hyp=hyp,
                rng=rng,
                trials=cfg.random_ce_trials,
                max_len=cfg.random_ce_max_len,
                mix_valid=cfg.random_ce_mix_valid,
                corrupt_p=cfg.random_ce_corrupt_p,
            )
        raise ValueError(f"Unknown eq_mode: {cfg.eq_mode}")


# -----------------------------
# Oracle wrapper (MQ cache + EQ count)
# -----------------------------
class Oracle:
    def __init__(self, graph: LabeledGraph):
        self.g = graph
        self._mq_cache: Dict[Tuple[int, ...], bool] = {}
        self.mq_calls = 0
        self.eq_calls = 0

    def mq(self, word: Tuple[int, ...]) -> bool:
        v = self._mq_cache.get(word)
        if v is not None:
            return v
        self.mq_calls += 1
        v = self.g.is_valid(list(word))
        self._mq_cache[word] = v
        return v

    def eq(self, hyp: DFA, cfg: Config, rng: random.Random) -> Optional[List[int]]:
        self.eq_calls += 1
        return self.g.find_counterexample(hyp, cfg, rng)


# -----------------------------
# L* Observation Table
# -----------------------------
def concat(u: Tuple[int, ...], v: Tuple[int, ...]) -> Tuple[int, ...]:
    return u + v


class LStar:
    def __init__(self, oracle: Oracle):
        self.O = oracle
        self.S: List[Tuple[int, ...]] = [tuple()]
        self.E: List[Tuple[int, ...]] = [tuple()]
        self.S_set: Set[Tuple[int, ...]] = {tuple()}
        self.E_set: Set[Tuple[int, ...]] = {tuple()}
        self.T: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], bool] = {}

    def _ensure_entry(self, s: Tuple[int, ...], e: Tuple[int, ...]) -> None:
        key = (s, e)
        if key not in self.T:
            self.T[key] = self.O.mq(concat(s, e))

    def fill_table(self, sigma: List[int]) -> None:
        # Fill rows for S and S·Σ over E
        for s in self.S:
            for e in self.E:
                self._ensure_entry(s, e)
        for s in self.S:
            for a in sigma:
                sa = s + (a,)
                for e in self.E:
                    self._ensure_entry(sa, e)

    def row(self, s: Tuple[int, ...]) -> Tuple[bool, ...]:
        return tuple(self.T[(s, e)] for e in self.E)

    def is_closed(self, sigma: List[int]) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        rows_S = {self.row(s) for s in self.S}
        for s in self.S:
            for a in sigma:
                sa = s + (a,)
                if self.row(sa) not in rows_S:
                    return False, sa
        return True, None

    def is_consistent(self, sigma: List[int]) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        reps: Dict[Tuple[bool, ...], List[Tuple[int, ...]]] = {}
        for s in self.S:
            reps.setdefault(self.row(s), []).append(s)

        for ss in reps.values():
            if len(ss) < 2:
                continue
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    s1, s2 = ss[i], ss[j]
                    for a in sigma:
                        r1 = self.row(s1 + (a,))
                        r2 = self.row(s2 + (a,))
                        if r1 != r2:
                            # find e where T differs => add a+e
                            for e in self.E:
                                v1 = self.T[(s1 + (a,), e)]
                                v2 = self.T[(s2 + (a,), e)]
                                if v1 != v2:
                                    return False, (a,) + e
                            return False, (a,)
        return True, None

    def add_counterexample_prefixes_to_S(self, ce: List[int]) -> None:
        t = tuple(ce)
        for i in range(len(t) + 1):
            p = t[:i]
            if p not in self.S_set:
                self.S_set.add(p)
                self.S.append(p)

    def add_to_E(self, e: Tuple[int, ...]) -> None:
        if e not in self.E_set:
            self.E_set.add(e)
            self.E.append(e)

    def build_hypothesis(self, sigma_size: int) -> DFA:
        row_to_state: Dict[Tuple[bool, ...], int] = {}
        rep_for_row: Dict[Tuple[bool, ...], Tuple[int, ...]] = {}

        for s in self.S:
            r = self.row(s)
            if r not in row_to_state:
                row_to_state[r] = len(row_to_state)
                rep_for_row[r] = s

        hyp = DFA(n_states=len(row_to_state), sigma_size=sigma_size)
        hyp.start = row_to_state[self.row(tuple())]

        for r, sid in row_to_state.items():
            srep = rep_for_row[r]
            hyp.acc[sid] = self.T[(srep, tuple())]

        sigma = list(range(sigma_size))
        for r, sid in row_to_state.items():
            srep = rep_for_row[r]
            for a in sigma:
                ra = self.row(srep + (a,))
                hyp.trans[sid][a] = row_to_state[ra]  # closedness ensures present
        return hyp


# -----------------------------
# Validation monitor set (fixed)
# -----------------------------
def sample_len_poissonish(rng: random.Random, avg: int, lo: int, hi: int) -> int:
    lam = max(1.0, float(avg))
    L = 0
    p = 1.0
    thresh = math.exp(-lam)
    while p > thresh and L < hi:
        L += 1
        p *= rng.random()
    return max(lo, min(hi, L))


def corrupt_uniform(trace: List[int], rng: random.Random, p: float, n_events: int) -> List[int]:
    out = trace[:]
    for i in range(len(out)):
        if rng.random() < p:
            old = out[i]
            new = rng.randrange(n_events)
            if new == old:
                new = (new + 1) % n_events
            out[i] = new
    return out


def build_val_monitor(graph: LabeledGraph, cfg: Config) -> Tuple[List[List[int]], List[List[int]]]:
    rng = random.Random(cfg.seed + 424242)
    pos: List[List[int]] = []
    for _ in range(cfg.val_monitor_n_pos):
        L = sample_len_poissonish(rng, cfg.val_monitor_avg_len, cfg.val_monitor_min_len, cfg.val_monitor_max_len)
        pos.append(graph.generate_valid_trace(rng, L))

    neg: List[List[int]] = []
    for tr in pos[: cfg.val_monitor_n_neg]:
        # pick corruption rate like your earlier eval
        p = (0.10, 0.20, 0.30, 0.45)[rng.randrange(4)]
        cand = corrupt_uniform(tr, rng, p, graph.n_events)
        if cfg.val_monitor_corrupt_until_invalid:
            bump = 0
            while graph.is_valid(cand) and bump < 10:
                cand = corrupt_uniform(tr, rng, min(0.95, p + 0.10 * (bump + 1)), graph.n_events)
                bump += 1
        neg.append(cand)

    return pos, neg


def eval_hyp_on_monitor(graph: LabeledGraph, hyp: DFA, pos: List[List[int]], neg: List[List[int]]) -> Dict[str, float]:
    # Evaluate hypothesis acceptance vs true validity
    tp = fp = tn = fn = 0

    # helper: run hyp
    def hyp_accept(word: List[int]) -> bool:
        s = hyp.start
        for a in word:
            s = hyp.delta(s, a)
        return hyp.accept(s)

    for w in pos:
        pred = hyp_accept(w)
        if pred: tp += 1
        else: fn += 1
    for w in neg:
        pred = hyp_accept(w)
        if pred: fp += 1
        else: tn += 1

    tpr = tp / max(1, tp + fn)  # true accept rate
    tnr = tn / max(1, tn + fp)  # true reject rate
    bal = 0.5 * (tpr + tnr)
    return {"TPR": tpr, "TNR": tnr, "balanced": bal}


# -----------------------------
# Run L*
# -----------------------------
def run_lstar(cfg: Config) -> None:
    random.seed(cfg.seed)

    graph = LabeledGraph(cfg.n_states, cfg.n_events, cfg.n_transitions, seed=cfg.seed, start_state=cfg.start_state)
    oracle = Oracle(graph)
    lstar = LStar(oracle)

    # Fixed validation monitor set
    val_pos, val_neg = build_val_monitor(graph, cfg)

    eq_rng = random.Random(cfg.seed + 99991)

    print(
        f"Target: |Q|={cfg.n_states} (+sink), |Σ|={cfg.n_events}, |δ|={cfg.n_transitions}, avg_out={graph.avg_out:.2f}"
    )
    print(f"EQ mode: {cfg.eq_mode}")
    if cfg.eq_mode == "random":
        print(
            f"  random_ce_trials={cfg.random_ce_trials} max_len={cfg.random_ce_max_len} "
            f"mix_valid={cfg.random_ce_mix_valid} corrupt_p={cfg.random_ce_corrupt_p}"
        )
    print(
        f"Val monitor: pos={len(val_pos)} neg={len(val_neg)} "
        f"(corrupt_until_invalid={cfg.val_monitor_corrupt_until_invalid})\n"
    )

    sigma = graph.sigma

    for it in range(1, cfg.max_lstar_iters + 1):
        # ensure closed+consistent
        changed = True
        while changed:
            changed = False
            lstar.fill_table(sigma)

            closed, witness = lstar.is_closed(sigma)
            if not closed:
                # witness is a prefix in S·Σ; add it to S
                lstar.add_counterexample_prefixes_to_S(list(witness))
                changed = True
                continue

            consistent, new_e = lstar.is_consistent(sigma)
            if not consistent:
                lstar.add_to_E(new_e)
                changed = True
                continue

        hyp = lstar.build_hypothesis(sigma_size=graph.n_events)

        # validation monitor metrics
        vm = eval_hyp_on_monitor(graph, hyp, val_pos, val_neg)

        # ask EQ
        ce = oracle.eq(hyp, cfg, eq_rng)

        if cfg.print_every_iter:
            ce_len = 0 if ce is None else len(ce)
            print(
                f"iter={it:4d} |S|={len(lstar.S):6d} |E|={len(lstar.E):5d} |Q_h|={hyp.n_states:5d} "
                f"MQ={oracle.mq_calls:9d} EQ={oracle.eq_calls:5d} "
                f"val_TPR={vm['TPR']:.4f} val_TNR={vm['TNR']:.4f} val_bal={vm['balanced']:.4f} "
                f"ce_len={ce_len:3d}"
                + ("  (EQ=OK)" if ce is None else "")
            )

        if ce is None:
            print("\nDONE: hypothesis is equivalent to target (under chosen EQ mode).")
            print(f"Final: |Q_h|={hyp.n_states}, MQ={oracle.mq_calls}, EQ={oracle.eq_calls}")
            return

        # incorporate CE
        lstar.add_counterexample_prefixes_to_S(ce)

    print("\nStopped: reached max_lstar_iters without convergence.")
    print(f"MQ={oracle.mq_calls}, EQ={oracle.eq_calls}")


def main():
    cfg = Config()
    run_lstar(cfg)


if __name__ == "__main__":
    main()
