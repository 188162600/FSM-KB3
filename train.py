from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dataset import (
    RevealConfig,
    UnifiedSequenceDataset,
    build_classical_characteristic_dataset,
    build_labeled_dataset,
    build_natural_characteristic_dataset,
    report_dataset,
)
from model import (
    Model,
    ModelConfig,
    aux_forward_next_state,
    gather_by_index,
    kl_stopgrad_p_to_q,
    last_nonpad_index,
    masked_label_smoothing_ce,
    multi_start_rollout_loss,
    rollout_from_bos,
)
from utilsdfa import (
    TASK_ACCEPTNESS,
    FamilyName,
    LengthPolicy,
    TensorDFALanguage,
    make_structured_tensor_dfa,
    report_dfa,
)


class GroupName(str, Enum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    J = "J"
    D = "D"
    E = "E"
    K = "K"


class VariantName(str, Enum):
    NAIVE = "naive"
    DENOISE = "denoise"
    FORWARD = "forward"
    ROLLOUT = "rollout"
    COMPATIBLE_NAIVE = "compatible_naive"
    COMPATIBLE_ROLLOUT = "compatible_rollout"


class SourceName(str, Enum):
    RANDOM = "random"
    CLASSICAL_CHARACTERISTIC = "classical_characteristic"
    NATURAL_CHARACTERISTIC = "natural_characteristic"


# =========================================================
# Schedules / objective config
# =========================================================

@dataclass(frozen=True)
class SchedulePoint:
    frac: float
    value: float


class PiecewiseLinearSchedule:
    def __init__(self, points: Tuple[SchedulePoint, ...]):
        pts = tuple(sorted(points, key=lambda p: p.frac))
        if len(pts) < 2:
            raise ValueError("Need at least 2 schedule points.")
        self.points = pts

    def __call__(self, progress: float) -> float:
        p = float(max(0.0, min(1.0, progress)))
        for a, b in zip(self.points[:-1], self.points[1:]):
            if p <= b.frac + 1e-12:
                if b.frac <= a.frac + 1e-12:
                    return float(b.value)
                t = (p - a.frac) / (b.frac - a.frac)
                return float(a.value + t * (b.value - a.value))
        return float(self.points[-1].value)


def const_sched(v: float) -> Tuple[SchedulePoint, ...]:
    return (SchedulePoint(0.0, float(v)), SchedulePoint(1.0, float(v)))


@dataclass
class LossScheduleConfig:
    cls: Tuple[SchedulePoint, ...] = field(default_factory=lambda: const_sched(1.0))
    rtd: Tuple[SchedulePoint, ...] = field(default_factory=lambda: const_sched(1.0))
    rec: Tuple[SchedulePoint, ...] = field(default_factory=lambda: const_sched(1.0))
    rec1: Tuple[SchedulePoint, ...] = field(default_factory=lambda: const_sched(1.0))
    auxf: Tuple[SchedulePoint, ...] = field(default_factory=lambda: (SchedulePoint(0.0, 0.0), SchedulePoint(0.1, 0.2), SchedulePoint(1.0, 0.2)))
    rollout: Tuple[SchedulePoint, ...] = field(default_factory=lambda: (SchedulePoint(0.0, 0.0), SchedulePoint(0.1, 0.1), SchedulePoint(1.0, 0.1)))
    roll_stab: Tuple[SchedulePoint, ...] = field(default_factory=lambda: (SchedulePoint(0.0, 0.0), SchedulePoint(0.1, 0.1), SchedulePoint(1.0, 0.1)))
    cls_roll: Tuple[SchedulePoint, ...] = field(default_factory=lambda: (SchedulePoint(0.0, 0.0), SchedulePoint(0.1, 0.02), SchedulePoint(1.0, 0.05)))
    cls_roll_kd: Tuple[SchedulePoint, ...] = field(default_factory=lambda: (SchedulePoint(0.0, 0.0), SchedulePoint(0.1, 0.02), SchedulePoint(1.0, 0.05)))
    compatible: Tuple[SchedulePoint, ...] = field(default_factory=lambda: (SchedulePoint(0.0, 0.0), SchedulePoint(0.1, 1.0), SchedulePoint(1.0, 1.0)))

    def build(self) -> Dict[str, PiecewiseLinearSchedule]:
        return {k: PiecewiseLinearSchedule(getattr(self, k)) for k in self.__dataclass_fields__}


@dataclass
class ObjectiveConfig:
    enabled: Dict[str, bool]
    schedules: LossScheduleConfig = field(default_factory=LossScheduleConfig)


def objective_for_variant(variant: VariantName | str) -> ObjectiveConfig:
    variant_name = variant.value if isinstance(variant, Enum) else str(variant)
    enabled = {k: False for k in ["cls", "rtd", "rec", "rec1", "auxf", "rollout", "roll_stab", "cls_roll", "cls_roll_kd", "compatible"]}
    schedules = LossScheduleConfig()
    if variant_name == VariantName.NAIVE.value:
        enabled["cls"] = True
        schedules = LossScheduleConfig(rtd=const_sched(0.0), rec=const_sched(0.0), rec1=const_sched(0.0), auxf=const_sched(0.0), rollout=const_sched(0.0), roll_stab=const_sched(0.0), cls_roll=const_sched(0.0), cls_roll_kd=const_sched(0.0), compatible=const_sched(0.0))
    elif variant_name == VariantName.DENOISE.value:
        enabled.update({"cls": True, "rtd": True, "rec": True, "rec1": True})
        schedules = LossScheduleConfig(auxf=const_sched(0.0), rollout=const_sched(0.0), roll_stab=const_sched(0.0), cls_roll=const_sched(0.0), cls_roll_kd=const_sched(0.0), compatible=const_sched(0.0))
    elif variant_name == VariantName.FORWARD.value:
        enabled.update({"cls": True, "rtd": True, "rec": True, "rec1": True, "auxf": True})
        schedules = LossScheduleConfig(rollout=const_sched(0.0), roll_stab=const_sched(0.0), cls_roll=const_sched(0.0), cls_roll_kd=const_sched(0.0), compatible=const_sched(0.0))
    elif variant_name == VariantName.ROLLOUT.value:
        enabled.update({"cls": True, "rtd": True, "rec": True, "rec1": True, "auxf": True, "rollout": True, "roll_stab": True, "cls_roll": True, "cls_roll_kd": True})
        schedules = LossScheduleConfig(compatible=const_sched(0.0))
    elif variant_name == VariantName.COMPATIBLE_NAIVE.value:
        enabled.update({"cls": True, "compatible": True})
        schedules = LossScheduleConfig(rtd=const_sched(0.0), rec=const_sched(0.0), rec1=const_sched(0.0), auxf=const_sched(0.0), rollout=const_sched(0.0), roll_stab=const_sched(0.0), cls_roll=const_sched(0.0), cls_roll_kd=const_sched(0.0))
    elif variant_name == VariantName.COMPATIBLE_ROLLOUT.value:
        enabled.update({"cls": True, "compatible": True, "rollout": True, "roll_stab": True, "cls_roll": True, "cls_roll_kd": True})
        schedules = LossScheduleConfig(rtd=const_sched(0.0), rec=const_sched(0.0), rec1=const_sched(0.0), auxf=const_sched(0.0))
    else:
        raise ValueError(f"Unknown variant: {variant_name}")
    return ObjectiveConfig(enabled=enabled, schedules=schedules)


# =========================================================
# Experiment configs
# =========================================================

@dataclass
class DfaConfig:
    num_states: int = 50
    alphabet_size: int = 50
    density: float = 0.2
    accept_prob: float = 0.5
    family: FamilyName | str = FamilyName.RANDOM
    device: str = "cpu"


@dataclass
class DataConfig:
    task_type: str = TASK_ACCEPTNESS
    train_source: SourceName | str = SourceName.RANDOM
    test_source: SourceName | str = SourceName.RANDOM
    train_size: int = 50000
    test_size: int = 50000
    sampling_mode: str = "mixed"
    length_policy: LengthPolicy = field(default_factory=lambda: LengthPolicy(min_len=5, max_len=40, avg_len=20, jitter=15, uniform=False))
    max_suffix_len: int = 12
    natural_multiplier: float = 1.0
    reveal_config: RevealConfig = field(default_factory=lambda: RevealConfig(value=1.0, type="ratio"))


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123
    batch_size: int = 512
    eval_batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = True
    total_train_samples: int = 2_000_000
    eval_every_num_samples: int = 100_000
    lr: float = 6e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    warmup_frac: float = 0.06
    min_lr_frac: float = 0.05
    replace_p_start: float = 0.05
    replace_p_mid: float = 0.20
    replace_p_end: float = 0.30
    replace_warmup_frac: float = 0.10
    replace_mid_frac: float = 0.33
    label_smoothing_rec: float = 0.05
    state_softmax_temp: float = 1.0
    t_softmax_temp: float = 1.0


@dataclass
class ExperimentJob:
    group: GroupName | str
    name: str
    dfa: DfaConfig
    data: DataConfig
    variant: VariantName | str
    seed: Optional[int] = None
    latent_states: Optional[int] = None


def default_group_seed_counts() -> Dict[str, int]:
    return {
        GroupName.S.value: 1,
        GroupName.A.value: 5,
        GroupName.B.value: 5,
        GroupName.C.value: 5,
        GroupName.D.value: 3,
        GroupName.E.value: 3,
        GroupName.J.value: 3,
        GroupName.K.value: 3,
    }


@dataclass
class SuiteConfig:
    output_dir: str = "experiment_runs"
    groups: Tuple[GroupName | str, ...] = (GroupName.S, GroupName.A, GroupName.B, GroupName.C, GroupName.J, GroupName.D, GroupName.E, GroupName.K)
    skip_existing: bool = True
    dry_run: bool = False
    max_experiments: Optional[int] = None
    dataset_tqdm: bool = True
    report_tqdm: bool = False
    group_seed_counts: Dict[str, int] = field(default_factory=default_group_seed_counts)
    shard_rank: int = 0
    num_shards: int = 1
    assign_device_from_shard: bool = True
    train: TrainConfig = field(default_factory=TrainConfig)


# =========================================================
# Utilities
# =========================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _enum_value(x: Any) -> Any:
    return x.value if isinstance(x, Enum) else x


def variant_uses_compatible(variant: VariantName | str) -> bool:
    v = _enum_value(variant)
    return v in {VariantName.COMPATIBLE_NAIVE.value, VariantName.COMPATIBLE_ROLLOUT.value}


def seed_values_for_group(group: GroupName | str, suite_cfg: SuiteConfig) -> List[int]:
    group_name = str(_enum_value(group))
    count = int(suite_cfg.group_seed_counts.get(group_name, 1))
    if count <= 1:
        return [int(suite_cfg.train.seed)]
    base_seed = int(suite_cfg.train.seed)
    return [base_seed + 1009 * i for i in range(count)]


def expand_seed_sweep(jobs: List[ExperimentJob], suite_cfg: SuiteConfig) -> List[ExperimentJob]:
    expanded: List[ExperimentJob] = []
    for job in jobs:
        seed_values = seed_values_for_group(job.group, suite_cfg)
        for seed in seed_values:
            seed_suffix = f"_seed{seed}" if len(seed_values) > 1 else ""
            expanded.append(
                replace(
                    job,
                    name=f"{job.name}{seed_suffix}",
                    seed=int(seed),
                )
            )
    return expanded


def lr_at_num_samples(samples_seen: int, total_train_samples: int, base_lr: float, warmup_frac: float, min_lr_frac: float) -> float:
    warmup_samples = int(round(warmup_frac * total_train_samples))
    if samples_seen < warmup_samples:
        return base_lr * (samples_seen + 1) / max(1, warmup_samples)
    t = (samples_seen - warmup_samples) / max(1, total_train_samples - warmup_samples)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)


def replace_prob_at_num_samples(samples_seen: int, total_train_samples: int, cfg: TrainConfig) -> float:
    p = samples_seen / max(1, total_train_samples - 1)
    if p <= cfg.replace_warmup_frac:
        return float(cfg.replace_p_start)
    if p <= cfg.replace_mid_frac:
        t = (p - cfg.replace_warmup_frac) / max(1e-12, cfg.replace_mid_frac - cfg.replace_warmup_frac)
        return float(cfg.replace_p_start + t * (cfg.replace_p_mid - cfg.replace_p_start))
    t = (p - cfg.replace_mid_frac) / max(1e-12, 1.0 - cfg.replace_mid_frac)
    return float(cfg.replace_p_mid + t * (cfg.replace_p_end - cfg.replace_p_mid))


def infinite_loader(loader: DataLoader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def cls_stats(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    total = int(y.numel())
    acc = float((pred == y).float().mean().item()) if total > 0 else 0.0
    mask0 = y == 0
    mask1 = y == 1
    y0 = float((pred[mask0] == y[mask0]).float().mean().item()) if mask0.any() else 0.0
    y1 = float((pred[mask1] == y[mask1]).float().mean().item()) if mask1.any() else 0.0
    return {"cls": acc, "cls_balance": 0.5 * (y0 + y1), "cls_label0": y0, "cls_label1": y1}


def mean_ce_from_logits(logits_list: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    if not logits_list:
        return y.float().sum() * 0.0
    losses = [F.cross_entropy(logits.float(), y) for logits in logits_list]
    return torch.stack(losses).mean()


def mean_kd_from_logits(logits_list: List[torch.Tensor], teacher_probs: torch.Tensor) -> torch.Tensor:
    if not logits_list:
        return teacher_probs.sum() * 0.0
    losses = [
        F.kl_div(
            F.log_softmax(logits.float(), dim=-1),
            teacher_probs,
            reduction="batchmean",
            log_target=False,
        )
        for logits in logits_list
    ]
    return torch.stack(losses).mean()


def compute_readout_logits(
    model: Model,
    out: Dict[str, torch.Tensor],
    cls_pos_ix: torch.Tensor,
    p_state: torch.Tensor,
    p_roll: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    final_token = gather_by_index(out["h_ca"].float(), cls_pos_ix)
    final_pred_state = gather_by_index(p_state.float(), cls_pos_ix)

    logits: Dict[str, torch.Tensor] = {
        "state_linear": model.cls_logits_from_state(final_pred_state),
        "state_mlp": model.cls_mlp_logits_from_state(final_pred_state),
        "token_linear": model.cls_logits_from_token(final_token),
        "token_mlp": model.cls_mlp_logits_from_token(final_token),
    }

    if p_roll is not None:
        final_roll_state = gather_by_index(p_roll.float(), cls_pos_ix)
        delta_state = (final_pred_state - final_roll_state).detach()
        logits.update({
            "roll_linear": model.cls_roll_logits(final_roll_state.detach()),
            "roll_mlp": model.cls_roll_mlp_logits(final_roll_state.detach()),
            "delta_linear": model.cls_delta_logits(delta_state),
            "delta_mlp": model.cls_delta_mlp_logits(delta_state),
        })

    return logits


def summarize_readout_metrics(logits_by_name: Dict[str, torch.Tensor], y: torch.Tensor) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if "state_linear" in logits_by_name:
        metrics.update(cls_stats(logits_by_name["state_linear"].detach(), y))
    for name, logits in logits_by_name.items():
        metrics[f"cls_{name}"] = cls_stats(logits.detach(), y)["cls"]
    return metrics


def accumulate_weighted_metrics(dst: Dict[str, float], src: Dict[str, float], weight: int) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + float(v) * float(weight)


def finalize_weighted_metrics(sums: Dict[str, float], denom: int) -> Dict[str, float]:
    return {k: float(v / max(1, denom)) for k, v in sums.items()}


ROLLOUT_DIAG_HORIZONS: Tuple[int, ...] = (1, 2, 3, 5)


def entropy_of_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = probs.float().clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=dim)


def symmetric_kl_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(1e-12)
    q = q.float().clamp_min(1e-12)
    log_p = p.log()
    log_q = q.log()
    kl_pq = F.kl_div(log_p, q, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(log_q, p, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pq + kl_qp)


def mean_cosine_similarity(p: torch.Tensor, q: torch.Tensor) -> float:
    if p.numel() == 0 or q.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(p.float(), q.float(), dim=-1).mean().item())


def compute_state_rollout_metrics(
    p_state: torch.Tensor,
    p_roll: torch.Tensor,
    T_probs: torch.Tensor,
    tok_idx_all: torch.Tensor,
    valid_tok_all: torch.Tensor,
    valid_state_all: torch.Tensor,
    cls_pos_ix: torch.Tensor,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    valid_state = valid_state_all.bool()
    if valid_state.any():
        flat_state = p_state.float()[valid_state]
        pH = entropy_of_probs(flat_state, dim=-1).mean()
        marginal_state = flat_state.mean(dim=0)
        mH = entropy_of_probs(marginal_state, dim=0)
        metrics["mH"] = float(mH.item())
        metrics["pH"] = float(pH.item())
        metrics["effS"] = float(torch.exp(mH).item())
    else:
        metrics["mH"] = 0.0
        metrics["pH"] = 0.0
        metrics["effS"] = 0.0

    final_state = gather_by_index(p_state.float(), cls_pos_ix)
    metrics["state_entropy"] = float(entropy_of_probs(final_state, dim=-1).mean().item()) if final_state.numel() > 0 else 0.0

    final_valid = valid_state_all.gather(1, cls_pos_ix.unsqueeze(1)).squeeze(1).bool()
    final_roll = gather_by_index(p_roll.float(), cls_pos_ix)
    if final_valid.any():
        metrics["rollout_cos_final"] = mean_cosine_similarity(final_roll[final_valid], final_state[final_valid])
        metrics["rollout_kl_final"] = float(symmetric_kl_probs(final_roll[final_valid], final_state[final_valid]).item())
    else:
        metrics["rollout_cos_final"] = 0.0
        metrics["rollout_kl_final"] = 0.0

    num_state_steps = int(p_state.size(1))
    for k in ROLLOUT_DIAG_HORIZONS:
        cos_key = f"rollout_cos_k{k}"
        kl_key = f"rollout_kl_k{k}"
        if num_state_steps <= k:
            metrics[cos_key] = 0.0
            metrics[kl_key] = 0.0
            continue

        num_start_pos = num_state_steps - k
        roll_k = p_state[:, :num_start_pos, :].float()
        valid_k = valid_state_all[:, :num_start_pos].bool() & valid_state_all[:, k:].bool()
        for step in range(k):
            tok_step = tok_idx_all[:, step:step + num_start_pos]
            valid_k = valid_k & valid_tok_all[:, step:step + num_start_pos].bool()
            roll_k = aux_forward_next_state(T_probs, roll_k, tok_step)

        target_k = p_state[:, k:, :].float()
        if valid_k.any():
            pred = roll_k[valid_k]
            tgt = target_k[valid_k]
            metrics[cos_key] = mean_cosine_similarity(pred, tgt)
            metrics[kl_key] = float(symmetric_kl_probs(pred, tgt).item())
        else:
            metrics[cos_key] = 0.0
            metrics[kl_key] = 0.0

    return metrics


def jaccard_metrics(logits: torch.Tensor, target: torch.Tensor, pos_mask: torch.Tensor) -> Dict[str, float]:
    if pos_mask.sum().item() == 0:
        return {"J_micro": 0.0, "J_macro": 0.0, "exact": 0.0, "precision": 0.0, "recall": 0.0}
    pred = torch.sigmoid(logits) > 0.5
    tgt = target.bool()
    inter = (pred & tgt).sum(dim=-1).float()
    union = (pred | tgt).sum(dim=-1).float()
    j_row = torch.where(union > 0, inter / union, torch.ones_like(union))
    valid_rows = pos_mask
    inter_tot = inter[valid_rows].sum().item()
    union_tot = union[valid_rows].sum().item()
    pred_tot = pred[valid_rows].sum().item()
    tgt_tot = tgt[valid_rows].sum().item()
    return {
        "J_micro": float(inter_tot / max(1.0, union_tot)),
        "J_macro": float(j_row[valid_rows].mean().item()),
        "exact": float((((pred == tgt).all(dim=-1) & valid_rows).float().sum() / valid_rows.float().sum().clamp_min(1.0)).item()),
        "precision": float(inter_tot / max(1.0, pred_tot)),
        "recall": float(inter_tot / max(1.0, tgt_tot)),
    }


def family_weights_for_family(name: FamilyName | str) -> Dict[str, float]:
    name = _enum_value(name)
    if name == FamilyName.RANDOM.value:
        return {"random": 1.0}
    if name == FamilyName.MODULAR.value:
        return {"modular": 1.0}
    if name == FamilyName.CLUSTERED.value:
        return {"clustered": 1.0}
    if name == FamilyName.MERGE.value:
        return {"merge": 1.0}
    raise ValueError(f"Unknown family: {name}")


# =========================================================
# Lightweight prepared pools / collators
# =========================================================

@dataclass
class PreparedSample:
    seq: List[int]
    label: int
    first_invalid_pos: int


@dataclass
class PreparedCompatibleSample:
    seq: List[int]
    label: int
    compatible_token: torch.Tensor  # [L, A]
    compatible_mask: torch.Tensor   # [L]


class PreparedSequencePool(Dataset):
    def __init__(self, samples: List[PreparedSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.seq, s.label, s.first_invalid_pos


class PreparedCompatiblePool(Dataset):
    def __init__(self, samples: List[PreparedCompatibleSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.seq, s.label, s.compatible_token, s.compatible_mask


class RawPadCollator:
    def __init__(self, bos_id: int, pad_id: int):
        self.bos_id = int(bos_id)
        self.pad_id = int(pad_id)

    def __call__(self, batch):
        seqs_real = [seq for (seq, _y, _sp) in batch]
        ys = [y for (_seq, y, _sp) in batch]
        first_invalid_pos = [_sp for (_seq, _y, _sp) in batch]
        seqs = [[self.bos_id] + seq for seq in seqs_real]
        T = max(len(s) for s in seqs) if seqs else 0
        input_ids = []
        attn = []
        lengths = []
        for s in seqs:
            pad = T - len(s)
            input_ids.append(s + [self.pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
            lengths.append(len(s) - 1)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            torch.tensor(first_invalid_pos, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )


class CompatiblePadCollator:
    def __init__(self, bos_id: int, pad_id: int, alphabet_size: int):
        self.bos_id = int(bos_id)
        self.pad_id = int(pad_id)
        self.alphabet_size = int(alphabet_size)

    def __call__(self, batch):
        seqs_real = [seq for (seq, _y, _ct, _cm) in batch]
        ys = [y for (_seq, y, _ct, _cm) in batch]
        ct_real = [ct for (_seq, _y, ct, _cm) in batch]
        cm_real = [cm for (_seq, _y, _ct, cm) in batch]
        seqs = [[self.bos_id] + seq for seq in seqs_real]
        T = max(len(s) for s in seqs) if seqs else 0
        input_ids = []
        attn = []
        lengths = []
        compatible_token = torch.zeros((len(batch), max(0, T - 1), self.alphabet_size), dtype=torch.bool)
        compatible_mask = torch.zeros((len(batch), max(0, T - 1)), dtype=torch.bool)
        for i, s in enumerate(seqs):
            pad = T - len(s)
            input_ids.append(s + [self.pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
            L = len(s) - 1
            lengths.append(L)
            if L > 0:
                compatible_token[i, :L] = ct_real[i][:L].bool()
                compatible_mask[i, :L] = cm_real[i][:L].bool()
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            compatible_token,
            compatible_mask,
            torch.tensor(lengths, dtype=torch.long),
        )


def prepare_pool_from_tensor_dataset(ds: UnifiedSequenceDataset) -> PreparedSequencePool:
    samples: List[PreparedSample] = []
    for i in range(len(ds)):
        L = int(ds.lengths[i].item())
        validity = int(ds.validity[i].item())
        valid_length = int(ds.valid_lengths[i].item())
        first_invalid_pos = -1 if validity == 1 else valid_length + 1
        samples.append(
            PreparedSample(
                seq=[int(x) for x in ds.seqs[i, :L].tolist()],
                label=int(ds.labels[i].item()),
                first_invalid_pos=int(first_invalid_pos),
            )
        )
    return PreparedSequencePool(samples)


def prepare_pool_from_compatible_dataset(ds: UnifiedSequenceDataset) -> PreparedCompatiblePool:
    samples: List[PreparedCompatibleSample] = []
    for i in range(len(ds)):
        L = int(ds.lengths[i].item())
        samples.append(
            PreparedCompatibleSample(
                seq=[int(x) for x in ds.seqs[i, :L].tolist()],
                label=int(ds.labels[i].item()),
                compatible_token=ds.compatible_token[i, :L].detach().cpu().clone(),
                compatible_mask=ds.compatible_mask[i, :L].detach().cpu().clone(),
            )
        )
    return PreparedCompatiblePool(samples)


# =========================================================
# Prefix-valid views / corruption helpers
# =========================================================

def prefix_state_and_token_views(raw_ids: torch.Tensor, attn: torch.Tensor, first_invalid_pos: Optional[torch.Tensor], bos_id: int, pad_id: int):
    tokens = raw_ids[:, 1:]
    tok_idx = tokens.clamp(0, bos_id - 1)
    if first_invalid_pos is None:
        valid_len = attn.long().sum(dim=1)
    else:
        valid_len = attn.long().sum(dim=1)
        hit = first_invalid_pos >= 0
        if hit.any():
            valid_len[hit] = torch.minimum(first_invalid_pos[hit].clamp_min(1), valid_len[hit])
    state_pos = torch.arange(raw_ids.size(1), device=raw_ids.device).unsqueeze(0)
    valid_state = (state_pos < valid_len.unsqueeze(1)) & (attn == 1)
    tok_pos = torch.arange(tokens.size(1), device=raw_ids.device).unsqueeze(0)
    max_valid_tok = (valid_len - 1).clamp_min(0)
    valid_tok = (tok_pos < max_valid_tok.unsqueeze(1)) & (tokens != pad_id) & (tokens != bos_id) & (attn[:, 1:] == 1) & (attn[:, :-1] == 1)
    return tokens, tok_idx, valid_tok, valid_state


def make_prefix_denoise_view_multi(raw_ids_cpu: torch.Tensor, valid_tok_cpu: torch.Tensor, rng: random.Random, replace_p: float, alphabet_size: int, bos_id: int, pad_id: int):
    B, _ = raw_ids_cpu.shape
    inp = raw_ids_cpu.clone()
    rec_labels = torch.full_like(raw_ids_cpu, -100)
    rtd_labels = torch.full_like(raw_ids_cpu, -100)
    for b in range(B):
        eligible = torch.nonzero(valid_tok_cpu[b], as_tuple=False).squeeze(-1).tolist()
        if not eligible:
            continue
        for j in eligible:
            rtd_labels[b, j + 1] = 0
        chosen = [j for j in eligible if rng.random() < replace_p]
        if not chosen:
            chosen = [rng.choice(eligible)]
        for j in chosen:
            raw_pos = j + 1
            true = int(raw_ids_cpu[b, raw_pos].item())
            if true in {bos_id, pad_id}:
                continue
            rep = rng.randrange(alphabet_size)
            if alphabet_size > 1:
                while rep == true:
                    rep = rng.randrange(alphabet_size)
            inp[b, raw_pos] = rep
            rec_labels[b, raw_pos] = true
            rtd_labels[b, raw_pos] = 1
    return inp, rec_labels, rtd_labels


def sample_rec1_positions(valid_tok_cpu: torch.Tensor, rng: random.Random) -> torch.Tensor:
    B, _ = valid_tok_cpu.shape
    pos = torch.full((B,), -1, dtype=torch.long)
    for b in range(B):
        eligible = torch.nonzero(valid_tok_cpu[b], as_tuple=False).squeeze(-1).tolist()
        if eligible:
            pos[b] = rng.choice(eligible)
    return pos


def make_prefix_denoise_view_rec1(raw_ids_cpu: torch.Tensor, rec1_pos: torch.Tensor, rng: random.Random, alphabet_size: int, bos_id: int, pad_id: int):
    B, _ = raw_ids_cpu.shape
    inp = raw_ids_cpu.clone()
    rec_labels = torch.full_like(raw_ids_cpu, -100)
    for b in range(B):
        pos = int(rec1_pos[b].item())
        if pos < 0:
            continue
        raw_pos = pos + 1
        true = int(raw_ids_cpu[b, raw_pos].item())
        if true in {bos_id, pad_id}:
            continue
        rep = rng.randrange(alphabet_size)
        if alphabet_size > 1:
            while rep == true:
                rep = rng.randrange(alphabet_size)
        inp[b, raw_pos] = rep
        rec_labels[b, raw_pos] = true
    return inp, rec_labels


# =========================================================
# Dataset routing
# =========================================================

def build_dataset_from_source(dfa: TensorDFALanguage, data_cfg: DataConfig, source: SourceName | str, split: str, seed: int, *, show_progress: bool = False) -> UnifiedSequenceDataset:
    source = _enum_value(source)
    pad_id = dfa.alphabet_size
    size = data_cfg.train_size if split == "train" else data_cfg.test_size
    n_pos = size // 2
    n_neg = size - n_pos
    common = dict(pad_id=pad_id, reveal_config=data_cfg.reveal_config, reveal_seed=seed + 17, show_progress=show_progress, build_batch_size=8192)
    if source == SourceName.RANDOM.value:
        return build_labeled_dataset(
            dfa=dfa,
            n_label1=n_pos,
            n_label0=n_neg,
            length_policy=data_cfg.length_policy,
            sampling_mode=data_cfg.sampling_mode,
            seed=seed,
            **common,
        )
    if source == SourceName.CLASSICAL_CHARACTERISTIC.value:
        return build_classical_characteristic_dataset(
            dfa=dfa,
            max_suffix_len=data_cfg.max_suffix_len,
            include_epsilon=True,
            **common,
        )
    if source == SourceName.NATURAL_CHARACTERISTIC.value:
        return build_natural_characteristic_dataset(
            dfa=dfa,
            length_policy=data_cfg.length_policy,
            max_suffix_len=data_cfg.max_suffix_len,
            natural_size=size,
            natural_multiplier=data_cfg.natural_multiplier,
            sampling_mode=data_cfg.sampling_mode,
            seed=seed,
            **common,
        )
    raise ValueError(f"Unknown source: {source}")


def construct_dfa_and_datasets(job: ExperimentJob, train_cfg: TrainConfig, *, show_progress: bool = False):
    dfa = make_structured_tensor_dfa(
        num_states=job.dfa.num_states,
        alphabet_size=job.dfa.alphabet_size,
        density=job.dfa.density,
        accept_prob=job.dfa.accept_prob,
        family_weights=family_weights_for_family(job.dfa.family),
        seed=train_cfg.seed,
        device=job.dfa.device,
    )
    train_ds = build_dataset_from_source(dfa, job.data, job.data.train_source, "train", train_cfg.seed + 1, show_progress=show_progress)
    test_ds = build_dataset_from_source(dfa, job.data, job.data.test_source, "test", train_cfg.seed + 2, show_progress=show_progress)
    return dfa, train_ds, test_ds


# =========================================================
# Training / eval
# =========================================================

def build_model_config(job: ExperimentJob, train_ds: UnifiedSequenceDataset, test_ds: UnifiedSequenceDataset) -> ModelConfig:
    observed_max_len = 0
    if len(train_ds) > 0:
        observed_max_len = max(observed_max_len, int(train_ds.lengths.max().item()))
    if len(test_ds) > 0:
        observed_max_len = max(observed_max_len, int(test_ds.lengths.max().item()))
    return ModelConfig(
        alphabet_size=job.dfa.alphabet_size,
        max_len=observed_max_len + 1,
        num_latent_states=int(job.latent_states) if job.latent_states is not None else max(16, min(256, job.dfa.num_states + 8)),
    )


def _run_clean_forward(model: Model, raw_ids: torch.Tensor, attn: torch.Tensor):
    return model(raw_ids, attn)


def train_epoch_acceptness(model: Model, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, model_cfg: ModelConfig, train_cfg: TrainConfig, objective: ObjectiveConfig, samples_seen_start: int, total_train_samples: int, chunk_budget: int) -> Dict[str, float]:
    model.train()
    schedules = objective.schedules.build()
    batch_iter = infinite_loader(loader)
    amp_enabled = train_cfg.use_amp and train_cfg.device.startswith("cuda")
    autocast_device = "cuda" if amp_enabled else "cpu"
    total_n = 0
    metric_sums: Dict[str, float] = {}

    while total_n < chunk_budget:
        raw_ids_cpu, attn_cpu, y_cpu, first_invalid_pos_cpu, _lengths_cpu = next(batch_iter)
        B = int(y_cpu.size(0))
        if total_n + B > chunk_budget:
            keep = chunk_budget - total_n
            raw_ids_cpu = raw_ids_cpu[:keep]
            attn_cpu = attn_cpu[:keep]
            y_cpu = y_cpu[:keep]
            first_invalid_pos_cpu = first_invalid_pos_cpu[:keep]
            B = keep

        progress = (samples_seen_start + total_n) / max(1, total_train_samples - 1)
        replace_p = replace_prob_at_num_samples(samples_seen_start + total_n, total_train_samples, train_cfg)
        lr = lr_at_num_samples(samples_seen_start + total_n, total_train_samples, train_cfg.lr, train_cfg.warmup_frac, train_cfg.min_lr_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        valid_tok_cpu = prefix_state_and_token_views(raw_ids_cpu, attn_cpu, first_invalid_pos_cpu, model_cfg.bos_id, model_cfg.pad_id)[2]
        rng = random.Random(train_cfg.seed + 1000003 * (samples_seen_start + total_n + 1))
        inp_cpu, rec_lab_cpu, rtd_lab_cpu = make_prefix_denoise_view_multi(raw_ids_cpu, valid_tok_cpu, rng, replace_p, model_cfg.alphabet_size, model_cfg.bos_id, model_cfg.pad_id)
        rec1_pos = sample_rec1_positions(valid_tok_cpu, rng)
        inp1_cpu, rec1_lab_cpu = make_prefix_denoise_view_rec1(raw_ids_cpu, rec1_pos, rng, model_cfg.alphabet_size, model_cfg.bos_id, model_cfg.pad_id)

        raw_ids = raw_ids_cpu.to(train_cfg.device, non_blocking=True)
        attn = attn_cpu.to(train_cfg.device, non_blocking=True)
        y = y_cpu.to(train_cfg.device, non_blocking=True)
        first_invalid_pos = first_invalid_pos_cpu.to(train_cfg.device, non_blocking=True)
        inp = inp_cpu.to(train_cfg.device, non_blocking=True)
        rec_labels = rec_lab_cpu.to(train_cfg.device, non_blocking=True)
        rtd_labels = rtd_lab_cpu.to(train_cfg.device, non_blocking=True)
        inp1 = inp1_cpu.to(train_cfg.device, non_blocking=True)
        rec1_labels = rec1_lab_cpu.to(train_cfg.device, non_blocking=True)

        tokens_all, tok_idx_all, valid_tok_all, valid_state_all = prefix_state_and_token_views(raw_ids, attn, first_invalid_pos, model_cfg.bos_id, model_cfg.pad_id)
        cls_pos_ix = last_nonpad_index(attn)

        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=amp_enabled):
            out = _run_clean_forward(model, raw_ids, attn)

        p_state = F.softmax(out["state_logits"].float() / train_cfg.state_softmax_temp, dim=-1)
        T_probs = model.T_probs_float(temp=train_cfg.t_softmax_temp)

        p0 = p_state[:, 0, :].float()
        p_roll = rollout_from_bos(T_probs, p0, tok_idx_all, valid_tok_all, model_cfg.alphabet_size)

        readout_logits = compute_readout_logits(model, out, cls_pos_ix, p_state, p_roll=p_roll)
        state_loss = mean_ce_from_logits([readout_logits["state_linear"], readout_logits["state_mlp"]], y)
        token_loss = mean_ce_from_logits([readout_logits["token_linear"], readout_logits["token_mlp"]], y)
        loss_cls = 0.75 * state_loss + 0.25 * token_loss

        loss = schedules["cls"](progress) * loss_cls

        if objective.enabled["auxf"] and valid_tok_all.any():
            cur_state = p_state[:, :-1, :]
            next_state = p_state[:, 1:, :]
            p_hat_next = aux_forward_next_state(T_probs, cur_state.float(), tok_idx_all)
            loss = loss + schedules["auxf"](progress) * 0.5 * (kl_stopgrad_p_to_q(next_state[valid_tok_all], p_hat_next[valid_tok_all]) + kl_stopgrad_p_to_q(p_hat_next[valid_tok_all], next_state[valid_tok_all]))

        if objective.enabled["roll_stab"]:
            m = valid_state_all[:, 1:]
            if m.any():
                enc = p_state[:, 1:, :]
                roll = p_roll[:, 1:, :]
                loss = loss + schedules["roll_stab"](progress) * 0.5 * (kl_stopgrad_p_to_q(enc[m], roll[m]) + kl_stopgrad_p_to_q(roll[m], enc[m]))
        if objective.enabled["rollout"]:
            loss = loss + schedules["rollout"](progress) * multi_start_rollout_loss(T_probs=T_probs, enc_states=p_state.float(), tokens=tok_idx_all, valid_tok=valid_tok_all, max_k=5, horizon_decay="inv")
        if objective.enabled["cls_roll"]:
            roll_loss = mean_ce_from_logits([readout_logits["roll_linear"], readout_logits["roll_mlp"]], y)
            delta_loss = mean_ce_from_logits([readout_logits["delta_linear"], readout_logits["delta_mlp"]], y)
            loss = loss + schedules["cls_roll"](progress) * 0.5 * (roll_loss + delta_loss)
        if objective.enabled["cls_roll_kd"]:
            with torch.no_grad():
                teacher_probs = 0.5 * (
                    F.softmax(readout_logits["state_linear"].float(), dim=-1)
                    + F.softmax(readout_logits["state_mlp"].float(), dim=-1)
                )
            roll_kd = mean_kd_from_logits([readout_logits["roll_linear"], readout_logits["roll_mlp"]], teacher_probs)
            delta_kd = mean_kd_from_logits([readout_logits["delta_linear"], readout_logits["delta_mlp"]], teacher_probs)
            loss = loss + schedules["cls_roll_kd"](progress) * 0.5 * (roll_kd + delta_kd)

        if objective.enabled["rtd"] or objective.enabled["rec"] or objective.enabled["rec1"]:
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=amp_enabled):
                out_den = model(inp, attn)
                if objective.enabled["rtd"]:
                    m_rtd = rtd_labels != -100
                    if m_rtd.any():
                        loss = loss + schedules["rtd"](progress) * F.binary_cross_entropy_with_logits(out_den["rtd_logits"][m_rtd], rtd_labels[m_rtd].float())
                if objective.enabled["rec"]:
                    loss = loss + schedules["rec"](progress) * masked_label_smoothing_ce(out_den["rec_logits"], rec_labels, train_cfg.label_smoothing_rec)
                if objective.enabled["rec1"]:
                    out_rec1 = model(inp1, attn)
                    loss = loss + schedules["rec1"](progress) * masked_label_smoothing_ce(out_rec1["rec_logits"], rec1_labels, train_cfg.label_smoothing_rec)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_n += B
        batch_metrics = summarize_readout_metrics(readout_logits, y)
        batch_metrics.update(compute_state_rollout_metrics(p_state, p_roll, T_probs, tok_idx_all, valid_tok_all, valid_state_all, cls_pos_ix))
        batch_metrics["loss"] = float(loss.item())
        accumulate_weighted_metrics(metric_sums, batch_metrics, B)

    result = {"n": float(total_n)}
    result.update(finalize_weighted_metrics(metric_sums, total_n))
    return result


@torch.no_grad()
def eval_epoch_acceptness(model: Model, loader: DataLoader, model_cfg: ModelConfig, train_cfg: TrainConfig, objective: ObjectiveConfig) -> Dict[str, float]:
    model.eval()
    metric_sums: Dict[str, float] = {}
    total_n = 0
    T_probs = model.T_probs_float(temp=train_cfg.t_softmax_temp)

    for raw_ids_cpu, attn_cpu, y_cpu, first_invalid_pos_cpu, _lengths_cpu in loader:
        raw_ids = raw_ids_cpu.to(train_cfg.device, non_blocking=True)
        attn = attn_cpu.to(train_cfg.device, non_blocking=True)
        y = y_cpu.to(train_cfg.device, non_blocking=True)
        first_invalid_pos = first_invalid_pos_cpu.to(train_cfg.device, non_blocking=True)
        _tokens_all, tok_idx_all, valid_tok_all, valid_state_all = prefix_state_and_token_views(raw_ids, attn, first_invalid_pos, model_cfg.bos_id, model_cfg.pad_id)
        cls_pos_ix = last_nonpad_index(attn)

        out = _run_clean_forward(model, raw_ids, attn)
        p_state = F.softmax(out["state_logits"].float() / train_cfg.state_softmax_temp, dim=-1)
        p0 = p_state[:, 0, :].float()
        p_roll = rollout_from_bos(T_probs, p0, tok_idx_all, valid_tok_all, model_cfg.alphabet_size)
        readout_logits = compute_readout_logits(model, out, cls_pos_ix, p_state, p_roll=p_roll)

        B = int(y.size(0))
        total_n += B
        batch_metrics = summarize_readout_metrics(readout_logits, y)
        batch_metrics.update(compute_state_rollout_metrics(p_state, p_roll, T_probs, tok_idx_all, valid_tok_all, valid_state_all, cls_pos_ix))
        accumulate_weighted_metrics(metric_sums, batch_metrics, B)

    return finalize_weighted_metrics(metric_sums, total_n)


def train_epoch_compatible(model: Model, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, model_cfg: ModelConfig, train_cfg: TrainConfig, objective: ObjectiveConfig, samples_seen_start: int, total_train_samples: int, chunk_budget: int) -> Dict[str, float]:
    model.train()
    schedules = objective.schedules.build()
    batch_iter = infinite_loader(loader)
    amp_enabled = train_cfg.use_amp and train_cfg.device.startswith("cuda")
    autocast_device = "cuda" if amp_enabled else "cpu"
    total_n = 0
    metric_sums: Dict[str, float] = {}

    while total_n < chunk_budget:
        raw_ids_cpu, attn_cpu, y_cpu, compatible_token_cpu, compatible_mask_cpu, _lengths_cpu = next(batch_iter)
        B = int(y_cpu.size(0))
        if total_n + B > chunk_budget:
            keep = chunk_budget - total_n
            raw_ids_cpu = raw_ids_cpu[:keep]
            attn_cpu = attn_cpu[:keep]
            y_cpu = y_cpu[:keep]
            compatible_token_cpu = compatible_token_cpu[:keep]
            compatible_mask_cpu = compatible_mask_cpu[:keep]
            B = keep

        progress = (samples_seen_start + total_n) / max(1, total_train_samples - 1)
        lr = lr_at_num_samples(samples_seen_start + total_n, total_train_samples, train_cfg.lr, train_cfg.warmup_frac, train_cfg.min_lr_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        raw_ids = raw_ids_cpu.to(train_cfg.device, non_blocking=True)
        attn = attn_cpu.to(train_cfg.device, non_blocking=True)
        y = y_cpu.to(train_cfg.device, non_blocking=True)
        compatible_token = compatible_token_cpu.to(train_cfg.device, non_blocking=True)
        compatible_mask = compatible_mask_cpu.to(train_cfg.device, non_blocking=True)
        first_invalid_pos = torch.full((B,), -1, dtype=torch.long, device=train_cfg.device)

        _tokens_all, tok_idx_all, valid_tok_all, valid_state_all = prefix_state_and_token_views(raw_ids, attn, first_invalid_pos, model_cfg.bos_id, model_cfg.pad_id)
        cls_pos_ix = last_nonpad_index(attn)

        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=amp_enabled):
            out = _run_clean_forward(model, raw_ids, attn)

        p_state = F.softmax(out["state_logits"].float() / train_cfg.state_softmax_temp, dim=-1)
        T_probs = model.T_probs_float(temp=train_cfg.t_softmax_temp)
        p0 = p_state[:, 0, :].float()
        p_roll = rollout_from_bos(T_probs, p0, tok_idx_all, valid_tok_all, model_cfg.alphabet_size)

        readout_logits = compute_readout_logits(model, out, cls_pos_ix, p_state, p_roll=p_roll)

        state_loss = mean_ce_from_logits([readout_logits["state_linear"], readout_logits["state_mlp"]], y)
        token_loss = mean_ce_from_logits([readout_logits["token_linear"], readout_logits["token_mlp"]], y)
        loss = readout_logits["state_linear"].sum() * 0.0
        if objective.enabled["cls"]:
            loss = loss + schedules["cls"](progress) * (0.75 * state_loss + 0.25 * token_loss)
        if objective.enabled["compatible"]:
            pos_mask = compatible_mask
            elem_mask = pos_mask.unsqueeze(-1).expand_as(compatible_token)
            if elem_mask.any():
                loss = loss + schedules["compatible"](progress) * F.binary_cross_entropy_with_logits(out["compat_logits"][elem_mask], compatible_token[elem_mask].float())

        if objective.enabled["roll_stab"]:
            m = valid_state_all[:, 1:]
            if m.any():
                enc = p_state[:, 1:, :]
                roll = p_roll[:, 1:, :]
                loss = loss + schedules["roll_stab"](progress) * 0.5 * (kl_stopgrad_p_to_q(enc[m], roll[m]) + kl_stopgrad_p_to_q(roll[m], enc[m]))
        if objective.enabled["rollout"]:
            loss = loss + schedules["rollout"](progress) * multi_start_rollout_loss(T_probs=T_probs, enc_states=p_state.float(), tokens=tok_idx_all, valid_tok=valid_tok_all, max_k=5, horizon_decay="inv")
        if objective.enabled["cls_roll"]:
            roll_loss = mean_ce_from_logits([readout_logits["roll_linear"], readout_logits["roll_mlp"]], y)
            delta_loss = mean_ce_from_logits([readout_logits["delta_linear"], readout_logits["delta_mlp"]], y)
            loss = loss + schedules["cls_roll"](progress) * 0.5 * (roll_loss + delta_loss)
        if objective.enabled["cls_roll_kd"]:
            with torch.no_grad():
                teacher_probs = 0.5 * (
                    F.softmax(readout_logits["state_linear"].float(), dim=-1)
                    + F.softmax(readout_logits["state_mlp"].float(), dim=-1)
                )
            roll_kd = mean_kd_from_logits([readout_logits["roll_linear"], readout_logits["roll_mlp"]], teacher_probs)
            delta_kd = mean_kd_from_logits([readout_logits["delta_linear"], readout_logits["delta_mlp"]], teacher_probs)
            loss = loss + schedules["cls_roll_kd"](progress) * 0.5 * (roll_kd + delta_kd)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_n += B
        batch_metrics = summarize_readout_metrics(readout_logits, y)
        batch_metrics.update(compute_state_rollout_metrics(p_state, p_roll, T_probs, tok_idx_all, valid_tok_all, valid_state_all, cls_pos_ix))
        if objective.enabled["compatible"]:
            batch_metrics["J_micro"] = jaccard_metrics(out["compat_logits"].detach().cpu(), compatible_token.cpu(), compatible_mask.cpu())["J_micro"]
        batch_metrics["loss"] = float(loss.item())
        accumulate_weighted_metrics(metric_sums, batch_metrics, B)

    result = {"n": float(total_n)}
    result.update(finalize_weighted_metrics(metric_sums, total_n))
    return result


@torch.no_grad()
def eval_epoch_compatible(model: Model, loader: DataLoader, model_cfg: ModelConfig, train_cfg: TrainConfig, objective: ObjectiveConfig) -> Dict[str, float]:
    model.eval()
    metric_sums: Dict[str, float] = {}
    jm = jM = exact = prec = rec = 0.0
    total_n = 0
    T_probs = model.T_probs_float(temp=train_cfg.t_softmax_temp)

    for raw_ids_cpu, attn_cpu, y_cpu, compatible_token_cpu, compatible_mask_cpu, _lengths_cpu in loader:
        raw_ids = raw_ids_cpu.to(train_cfg.device, non_blocking=True)
        attn = attn_cpu.to(train_cfg.device, non_blocking=True)
        y = y_cpu.to(train_cfg.device, non_blocking=True)
        compatible_token = compatible_token_cpu.to(train_cfg.device, non_blocking=True)
        compatible_mask = compatible_mask_cpu.to(train_cfg.device, non_blocking=True)
        first_invalid_pos = torch.full((raw_ids.size(0),), -1, dtype=torch.long, device=train_cfg.device)
        _tokens_all, tok_idx_all, valid_tok_all, valid_state_all = prefix_state_and_token_views(raw_ids, attn, first_invalid_pos, model_cfg.bos_id, model_cfg.pad_id)
        cls_pos_ix = last_nonpad_index(attn)

        out = _run_clean_forward(model, raw_ids, attn)
        p_state = F.softmax(out["state_logits"].float() / train_cfg.state_softmax_temp, dim=-1)
        p0 = p_state[:, 0, :].float()
        p_roll = rollout_from_bos(T_probs, p0, tok_idx_all, valid_tok_all, model_cfg.alphabet_size)
        readout_logits = compute_readout_logits(model, out, cls_pos_ix, p_state, p_roll=p_roll)

        stats = jaccard_metrics(out["compat_logits"].detach().cpu(), compatible_token.cpu(), compatible_mask.cpu())
        B = raw_ids.size(0)
        jm += stats["J_micro"] * B
        jM += stats["J_macro"] * B
        exact += stats["exact"] * B
        prec += stats["precision"] * B
        rec += stats["recall"] * B
        total_n += B
        batch_metrics = summarize_readout_metrics(readout_logits, y)
        batch_metrics.update(compute_state_rollout_metrics(p_state, p_roll, T_probs, tok_idx_all, valid_tok_all, valid_state_all, cls_pos_ix))
        accumulate_weighted_metrics(metric_sums, batch_metrics, B)

    outm = finalize_weighted_metrics(metric_sums, total_n)
    outm.update({"J_micro": jm / max(1, total_n), "J_macro": jM / max(1, total_n), "exact": exact / max(1, total_n), "precision": prec / max(1, total_n), "recall": rec / max(1, total_n)})
    return outm


# =========================================================
# Rollout diagnostic / experiment runner
# =========================================================

@torch.no_grad()
def collect_rollout_features(model: Model, loader: DataLoader, model_cfg: ModelConfig, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    T_probs = model.T_probs_float()
    for raw_ids_cpu, attn_cpu, y_cpu, first_invalid_pos_cpu, _lengths_cpu in loader:
        raw_ids = raw_ids_cpu.to(device, non_blocking=True)
        attn = attn_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        first_invalid_pos = first_invalid_pos_cpu.to(device, non_blocking=True)
        tokens_all, tok_idx_all, valid_tok_all, _valid_state_all = prefix_state_and_token_views(raw_ids, attn, first_invalid_pos, model_cfg.bos_id, model_cfg.pad_id)
        cls_pos_ix = last_nonpad_index(attn)
        out = model(raw_ids, attn)
        p_state = F.softmax(out["state_logits"].float(), dim=-1)
        p0 = p_state[:, 0, :].float()
        p_roll = rollout_from_bos(T_probs, p0, tok_idx_all, valid_tok_all, model_cfg.alphabet_size)
        roll_final = gather_by_index(p_roll, cls_pos_ix).detach().cpu()
        xs.append(roll_final)
        ys.append(y.detach().cpu())
    if not xs:
        return torch.empty((0, model_cfg.num_latent_states)), torch.empty((0,), dtype=torch.long)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def fit_linear_probe_lstsq(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor) -> torch.Tensor:
    X = torch.cat([x_train, torch.ones((x_train.size(0), 1), dtype=x_train.dtype)], dim=1)
    Xte = torch.cat([x_test, torch.ones((x_test.size(0), 1), dtype=x_test.dtype)], dim=1)
    Y = F.one_hot(y_train.long(), num_classes=2).float()
    W = torch.linalg.lstsq(X, Y).solution
    return Xte @ W


# =========================================================
# Experiment runner
# =========================================================

def _jsonable(x: Any) -> Any:
    if isinstance(x, Enum):
        return x.value
    if hasattr(x, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.dim() > 0 else x.item()
    return x


def fingerprint_job(job: ExperimentJob) -> str:
    payload = {"group": _jsonable(job.group), "name": job.name, "variant": _jsonable(job.variant), "dfa": _jsonable(job.dfa), "data": _jsonable(job.data)}
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def run_experiment(job: ExperimentJob, suite_cfg: SuiteConfig) -> Dict[str, Any]:
    train_cfg = replace(suite_cfg.train, seed=int(job.seed) if job.seed is not None else int(suite_cfg.train.seed))
    set_all_seeds(train_cfg.seed)
    fp = fingerprint_job(job)
    out_dir = Path(suite_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job.name}_{fp}.json"
    if suite_cfg.skip_existing and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    objective = objective_for_variant(job.variant)
    attach_compatible = bool(objective.enabled["compatible"])
    t0 = time.time()
    dfa, train_ds, test_ds = construct_dfa_and_datasets(job, train_cfg, show_progress=suite_cfg.dataset_tqdm)
    model_cfg = build_model_config(job, train_ds, test_ds)

    if attach_compatible:
        train_pool = prepare_pool_from_compatible_dataset(train_ds)
        test_pool = prepare_pool_from_compatible_dataset(test_ds)
        collator = CompatiblePadCollator(model_cfg.bos_id, model_cfg.pad_id, model_cfg.alphabet_size)
    else:
        train_pool = prepare_pool_from_tensor_dataset(train_ds)
        test_pool = prepare_pool_from_tensor_dataset(test_ds)
        collator = RawPadCollator(model_cfg.bos_id, model_cfg.pad_id)

    pin = bool(train_cfg.pin_memory and train_cfg.device.startswith("cuda"))
    train_loader = DataLoader(train_pool, batch_size=train_cfg.batch_size, shuffle=True, drop_last=False, num_workers=train_cfg.num_workers, pin_memory=pin, persistent_workers=(pin and train_cfg.num_workers > 0), collate_fn=collator)
    test_loader = DataLoader(test_pool, batch_size=train_cfg.eval_batch_size, shuffle=False, drop_last=False, num_workers=train_cfg.num_workers, pin_memory=pin, persistent_workers=(pin and train_cfg.num_workers > 0), collate_fn=collator)

    model = Model(model_cfg).to(train_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_cfg.use_amp and train_cfg.device.startswith("cuda")))

    print("DATASET", json.dumps({"job": job.name, "train_N": int(len(train_ds)), "test_N": int(len(test_ds)), "train_max_len": int(train_ds.lengths.max().item()) if len(train_ds) > 0 else 0, "test_max_len": int(test_ds.lengths.max().item()) if len(test_ds) > 0 else 0, "train_revealed_positions": int(train_ds.compatible_mask.sum().item()), "test_revealed_positions": int(test_ds.compatible_mask.sum().item())}, sort_keys=True))
    print("EXEC", json.dumps({"job": job.name, "group": _enum_value(job.group), "variant": _enum_value(job.variant), "seed": int(train_cfg.seed), "latent_states": int(job.latent_states) if job.latent_states is not None else None, "family": _enum_value(job.dfa.family), "num_states": int(job.dfa.num_states), "density": float(job.dfa.density), "train_source": _enum_value(job.data.train_source), "test_source": _enum_value(job.data.test_source), "train_size": int(job.data.train_size), "test_size": int(job.data.test_size), "reveal_config": _jsonable(job.data.reveal_config), "compatible_enabled": bool(attach_compatible), "enabled_losses": {k: bool(v) for k, v in objective.enabled.items() if v}}, sort_keys=True))

    samples_trained = 0
    history: List[Dict[str, Any]] = []
    train_epoch_fn = train_epoch_compatible if attach_compatible else train_epoch_acceptness
    eval_epoch_fn = eval_epoch_compatible if attach_compatible else eval_epoch_acceptness
    total_budget = int(train_cfg.total_train_samples)
    active_eval_every = int(train_cfg.eval_every_num_samples)
    exp_bar = tqdm(total=total_budget, desc=f"{_enum_value(job.group)}:{_enum_value(job.variant)}:{fp[:8]}", unit="sample", leave=False, position=1)

    while samples_trained < total_budget:
        chunk_budget = min(active_eval_every, total_budget - samples_trained)
        tr = train_epoch_fn(model, train_loader, optimizer, scaler, model_cfg, train_cfg, objective, samples_trained, total_budget, chunk_budget)
        samples_trained += int(tr.get("n", 0))
        exp_bar.update(int(tr.get("n", 0)))
        te = eval_epoch_fn(model, test_loader, model_cfg, train_cfg, objective)
        history.append({"samples": int(samples_trained), "train": tr, "test": te})
        if attach_compatible:
            exp_bar.set_postfix(train_J=f"{100 * tr.get('J_micro', 0):.1f}%", test_J=f"{100 * te.get('J_micro', 0):.1f}%")
        else:
            exp_bar.set_postfix(train_cls=f"{100 * tr.get('cls', 0):.1f}%", test_cls=f"{100 * te.get('cls', 0):.1f}%")
    exp_bar.close()

    final_train = history[-1]["train"] if history else {}
    final_test = history[-1]["test"] if history else {}

    def _extract_rollout_diag(metrics: Dict[str, Any]) -> Dict[str, float]:
        explicit = {"mH", "pH", "effS", "state_entropy"}
        return {
            k: float(v)
            for k, v in metrics.items()
            if k.startswith("cls_") or k.startswith("rollout_") or k in explicit
        }

    roll_diag = {
        "train": _extract_rollout_diag(final_train),
        "test": _extract_rollout_diag(final_test),
    }

    runtime = time.time() - t0
    result = {
        "fingerprint": fp,
        "job": {"group": _jsonable(job.group), "name": job.name, "variant": _jsonable(job.variant), "seed": int(train_cfg.seed), "latent_states": int(job.latent_states) if job.latent_states is not None else None, "dfa": _jsonable(job.dfa), "data": _jsonable(job.data), "sample_budget": int(total_budget), "eval_every_num_samples": int(active_eval_every)},
        "dfa_info": report_dfa(dfa, max_len=max(12, int(model_cfg.max_len))),
        "train_dataset_info": report_dataset(train_ds, dfa, use_revealed_positions=True),
        "test_dataset_info": report_dataset(test_ds, dfa, use_revealed_positions=True),
        "history": history,
        "rollout_diagnostic": roll_diag,
        "runtime_seconds": runtime,
    }
    if not suite_cfg.dry_run:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    if attach_compatible:
        print(
            f"DONE {job.name} [{fp}] | "
            f"train cls {100*final_train.get('cls',0):6.2f}% J {100*final_train.get('J_micro',0):6.2f}% | "
            f"test cls {100*final_test.get('cls',0):6.2f}% J {100*final_test.get('J_micro',0):6.2f}% "
            f"Jm {100*final_test.get('J_macro',0):6.2f}% exact {100*final_test.get('exact',0):6.2f}% "
            f"| stateMLP {100*final_test.get('cls_state_mlp',0):6.2f}% "
            f"roll {100*final_test.get('cls_roll_linear',0):6.2f}% "
            f"delta {100*final_test.get('cls_delta_linear',0):6.2f}% "
            f"token {100*final_test.get('cls_token_linear',0):6.2f}% runtime {runtime:.1f}s"
        )
    else:
        rd = roll_diag.get("test", {}) if isinstance(roll_diag, dict) else {}
        print(
            f"DONE {job.name} [{fp}] | "
            f"train cls {100*final_train.get('cls',0):6.2f}% | "
            f"test cls {100*final_test.get('cls',0):6.2f}% bal {100*final_test.get('cls_balance',0):6.2f}% "
            f"y0 {100*final_test.get('cls_label0',0):6.2f}% y1 {100*final_test.get('cls_label1',0):6.2f}% | "
            f"stateMLP {100*rd.get('cls_state_mlp',0):6.2f}% "
            f"roll {100*rd.get('cls_roll_linear',0):6.2f}% "
            f"rollMLP {100*rd.get('cls_roll_mlp',0):6.2f}% "
            f"delta {100*rd.get('cls_delta_linear',0):6.2f}% "
            f"deltaMLP {100*rd.get('cls_delta_mlp',0):6.2f}% "
            f"token {100*rd.get('cls_token_linear',0):6.2f}% "
            f"tokenMLP {100*rd.get('cls_token_mlp',0):6.2f}% runtime {runtime:.1f}s"
        )
    return result


# =========================================================
# Suite groups
# =========================================================

def make_job(
    group: GroupName | str,
    name: str,
    size: int,
    density: float,
    variant: VariantName | str,
    source: SourceName | str,
    family: FamilyName | str = FamilyName.RANDOM,
    train_size: int = 100000,
    test_size: int = 100000,
    reveal_cfg: Optional[RevealConfig] = None,
    alphabet_size: Optional[int] = None,
    latent_states: Optional[int] = None,
) -> ExperimentJob:
    if reveal_cfg is None:
        reveal_cfg = RevealConfig(value=1.0, type="ratio")
    if alphabet_size is None:
        alphabet_size = size
    data_cfg = DataConfig(task_type=TASK_ACCEPTNESS, train_source=source, test_source=SourceName.RANDOM, train_size=train_size, test_size=test_size, reveal_config=reveal_cfg)
    dfa_cfg = DfaConfig(num_states=size, alphabet_size=alphabet_size, density=density, family=family)
    return ExperimentJob(group=group, name=name, dfa=dfa_cfg, data=data_cfg, variant=variant, latent_states=latent_states)


def build_group_S(train_cfg: TrainConfig) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    base = [VariantName.NAIVE, VariantName.DENOISE, VariantName.FORWARD, VariantName.ROLLOUT, VariantName.COMPATIBLE_NAIVE, VariantName.COMPATIBLE_ROLLOUT]
    for variant in base:
        jobs.append(make_job(GroupName.S, f"smoke_S50_d0.20_task-acceptness_model-{variant.value}", 50, 0.20, variant, SourceName.RANDOM, train_size=2000, test_size=2000, reveal_cfg=RevealConfig(value=0.5, type="ratio")))
    return jobs


def build_group_A() -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for size in [50, 100]:
        for density in [0.10, 0.20, 0.30]:
            for variant in [VariantName.NAIVE, VariantName.DENOISE, VariantName.FORWARD, VariantName.ROLLOUT]:
                jobs.append(make_job(GroupName.A, f"mainAccept_S{size}_d{density:.2f}_task-acceptness_model-{variant.value}", size, density, variant, SourceName.RANDOM))
    return jobs


def build_group_B() -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for density in [0.40, 0.50]:
        for variant in [VariantName.NAIVE, VariantName.ROLLOUT]:
            jobs.append(make_job(GroupName.B, f"highAmbiguity_S50_d{density:.2f}_task-acceptness_model-{variant.value}", 50, density, variant, SourceName.RANDOM))
    return jobs


def build_group_C() -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for density in [0.10, 0.20, 0.30]:
        for source in [SourceName.CLASSICAL_CHARACTERISTIC, SourceName.NATURAL_CHARACTERISTIC]:
            for variant in [VariantName.NAIVE, VariantName.ROLLOUT]:
                jobs.append(make_job(GroupName.C, f"charDiagnostic_S50_d{density:.2f}_src-{source.value}_task-acceptness_model-{variant.value}", 50, density, variant, source))
    return jobs


def build_group_J() -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    reveal_cfgs = [("1pos", RevealConfig(value=1, type="absolute")), ("25p", RevealConfig(value=0.25, type="ratio")), ("50p", RevealConfig(value=0.5, type="ratio")), ("100p", RevealConfig(value=1.0, type="ratio"))]
    for density in [0.10, 0.20, 0.30]:
        for tag, rcfg in reveal_cfgs:
            for variant in [VariantName.COMPATIBLE_NAIVE, VariantName.COMPATIBLE_ROLLOUT]:
                jobs.append(make_job(GroupName.J, f"compatibleMask_S50_d{density:.2f}_{tag}_model-{variant.value}", 50, density, variant, SourceName.RANDOM, reveal_cfg=rcfg))
    return jobs


def build_group_D() -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for family in [FamilyName.MODULAR, FamilyName.CLUSTERED, FamilyName.MERGE]:
        for density in [0.10, 0.20, 0.30, 0.40, 0.50]:
            for variant in [VariantName.NAIVE, VariantName.ROLLOUT]:
                jobs.append(make_job(GroupName.D, f"familyRobust_{family.value}_S50_d{density:.2f}_task-acceptness_model-{variant.value}", 50, density, variant, SourceName.RANDOM, family=family))
    return jobs


def build_group_E() -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    configs = [
        (50, 50, 24),
        (50, 50, 20),
        (50, 50, 28),
        (100, 100, 10),
        (100, 100, 12),
        (100, 100, 15),
        (25, 25, 48),
        (25, 25, 56),
        (25, 25, 40),
        (50, 100, 10),
        (50, 100, 12),
        (50, 100, 15),
        (50, 100, 20),
        (50, 100, 24),
        (50, 100, 28),
        (100, 50, 10),
        (100, 50, 12),
        (100, 50, 15),
        (100, 50, 20),
        (100, 50, 24),
        (100, 50, 28),
    ]
    for variant in [VariantName.ROLLOUT, VariantName.DENOISE]:
        for alphabet_size, num_states, density_pct in configs:
            density = density_pct / 100.0
            jobs.append(
                make_job(
                    GroupName.E,
                    f"randomAccept_A{alphabet_size}_S{num_states}_d{density:.2f}_task-acceptness_model-{variant.value}",
                    num_states,
                    density,
                    variant,
                    SourceName.RANDOM,
                    alphabet_size=alphabet_size,
                )
            )
    return jobs

def build_group_K() -> List[ExperimentJob]:
    """Latent Markov-capacity sweep for the rollout comparison.

    This keeps the DFA family, density, data source, and training budget fixed
    while varying only the latent channel size, matching the manuscript TODO.
    """
    jobs: List[ExperimentJob] = []
    family = FamilyName.RANDOM
    density = 0.30
    size = 50
    source = SourceName.RANDOM
    reveal_cfg = RevealConfig(value=1.0, type="ratio")
    for latent_states in [8, 16, 32, 64, 128]:
        for variant in [VariantName.ROLLOUT, VariantName.COMPATIBLE_ROLLOUT]:
            jobs.append(
                make_job(
                    GroupName.K,
                    f"latentCapacity_{family.value}_S{size}_d{density:.2f}_L{latent_states}_model-{variant.value}",
                    size,
                    density,
                    variant,
                    source,
                    family=family,
                    train_size=100000,
                    test_size=100000,
                    reveal_cfg=reveal_cfg,
                    latent_states=latent_states,
                )
            )
    return jobs

def build_jobs(suite_cfg: SuiteConfig) -> Dict[str, List[ExperimentJob]]:
    out: Dict[str, List[ExperimentJob]] = {}
    for g in suite_cfg.groups:
        gv = _enum_value(g)
        base_jobs: List[ExperimentJob]
        if gv == GroupName.S.value:
            base_jobs = build_group_S(suite_cfg.train)
        elif gv == GroupName.A.value:
            base_jobs = build_group_A()
        elif gv == GroupName.B.value:
            base_jobs = build_group_B()
        elif gv == GroupName.C.value:
            base_jobs = build_group_C()
        elif gv == GroupName.J.value:
            base_jobs = build_group_J()
        elif gv == GroupName.D.value:
            base_jobs = build_group_D()
        elif gv == GroupName.E.value:
            base_jobs = build_group_E()
        elif gv == GroupName.K.value:
            base_jobs = build_group_K()
        else:
            raise ValueError(f"Unknown group: {g}")
        out[g] = expand_seed_sweep(base_jobs, suite_cfg)
    return out


def available_cuda_device_count() -> int:
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def resolve_suite_device(suite_cfg: SuiteConfig) -> str:
    device = str(suite_cfg.train.device)
    if not suite_cfg.assign_device_from_shard:
        return device
    if not device.startswith("cuda"):
        return device
    n_cuda = available_cuda_device_count()
    if n_cuda <= 1:
        return "cuda:0" if n_cuda == 1 else device
    return f"cuda:{suite_cfg.shard_rank % n_cuda}"


def shard_jobs(jobs: List[ExperimentJob], suite_cfg: SuiteConfig) -> List[ExperimentJob]:
    num_shards = int(max(1, suite_cfg.num_shards))
    shard_rank = int(suite_cfg.shard_rank)
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(f"Invalid shard_rank={shard_rank} for num_shards={num_shards}")
    if num_shards == 1:
        return jobs
    return [job for idx, job in enumerate(jobs) if idx % num_shards == shard_rank]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFA experiment suite with seed sweeps and optional job sharding across GPUs.")
    parser.add_argument("--groups", nargs="*", default=None, help="Subset of groups to run, e.g. A B C")
    parser.add_argument("--max-experiments", type=int, default=None, help="Optional cap after sharding")
    parser.add_argument("--output-dir", type=str, default="experiment_runs")
    parser.add_argument("--device", type=str, default=None, help="Explicit device override, e.g. cuda:0 or cpu")
    parser.add_argument("--shard-rank", type=int, default=0, help="This worker's shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of workers / shards")
    parser.add_argument("--no-assign-device-from-shard", action="store_true", help="Do not map shard rank to cuda device automatically")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dataset-tqdm", action="store_true", default=None, help="Enable dataset-building progress bars")
    parser.add_argument("--no-dataset-tqdm", action="store_true", default=None, help="Disable dataset-building progress bars")
    return parser.parse_args()


def run_experiment_suite(suite_cfg: SuiteConfig) -> List[Dict[str, Any]]:
    suite_cfg.train.device = resolve_suite_device(suite_cfg)
    jobs_by_group = build_jobs(suite_cfg)
    ordered_groups = list(suite_cfg.groups)
    all_jobs = [job for g in ordered_groups for job in jobs_by_group[g]]
    jobs = shard_jobs(all_jobs, suite_cfg)
    if suite_cfg.max_experiments is not None:
        jobs = jobs[: suite_cfg.max_experiments]

    print("=== Experiment suite plan ===")
    print(json.dumps({
        "requested_groups": [_enum_value(x) for x in ordered_groups],
        "output_dir": suite_cfg.output_dir,
        "skip_existing": suite_cfg.skip_existing,
        "dry_run": suite_cfg.dry_run,
        "device": suite_cfg.train.device,
        "available_cuda_devices": available_cuda_device_count(),
        "num_all_jobs": len(all_jobs),
        "num_jobs": len(jobs),
        "shard_rank": int(suite_cfg.shard_rank),
        "num_shards": int(suite_cfg.num_shards),
    }, indent=2))

    for g in ordered_groups:
        cur = [j for j in jobs if _enum_value(j.group) == _enum_value(g)]
        print(f"--- Group {_enum_value(g)} ({len(cur)} experiments) ---")
        for i, job in enumerate(cur, start=1):
            fp = fingerprint_job(job)
            print(
                f"[{i:03d}/{len(cur):03d}] {job.name} | "
                f"variant={_enum_value(job.variant)} | "
                f"seed={int(job.seed) if job.seed is not None else int(suite_cfg.train.seed)} | "
                f"latent={job.latent_states if job.latent_states is not None else 'default'} | "
                f"family={_enum_value(job.dfa.family)} | "
                f"S={int(job.dfa.num_states)} | A={int(job.dfa.alphabet_size)} | "
                f"density={float(job.dfa.density):.2f} | "
                f"train={_enum_value(job.data.train_source)}:{int(job.data.train_size)} | "
                f"test={_enum_value(job.data.test_source)}:{int(job.data.test_size)} | "
                f"reveal={job.data.reveal_config.type}:{job.data.reveal_config.value} | "
                f"fp={fp}"
            )

    results: List[Dict[str, Any]] = []
    pbar = tqdm(jobs, desc=f"experiments shard {suite_cfg.shard_rank}/{suite_cfg.num_shards}", unit="exp")
    for job in pbar:
        fp = fingerprint_job(job)
        pbar.set_postfix(group=_enum_value(job.group), variant=_enum_value(job.variant), fp=fp)
        results.append(run_experiment(job, suite_cfg))
    return results


def main() -> None:
    args = parse_args()
    groups = tuple(args.groups) if args.groups else (GroupName.S, GroupName.A, GroupName.B, GroupName.C, GroupName.J, GroupName.D, GroupName.E, GroupName.K)
    dataset_tqdm = True
    if args.no_dataset_tqdm:
        dataset_tqdm = False
    elif args.dataset_tqdm:
        dataset_tqdm = True
    suite_cfg = SuiteConfig(
        output_dir=args.output_dir,
        groups=groups,
        skip_existing=not args.no_skip_existing,
        dry_run=bool(args.dry_run),
        max_experiments=args.max_experiments,
        dataset_tqdm=dataset_tqdm,
        shard_rank=int(args.shard_rank),
        num_shards=int(args.num_shards),
        assign_device_from_shard=not args.no_assign_device_from_shard,
    )
    if args.device is not None:
        suite_cfg.train.device = str(args.device)
        suite_cfg.assign_device_from_shard = False
    run_experiment_suite(suite_cfg)


if __name__ == "__main__":
    main()
