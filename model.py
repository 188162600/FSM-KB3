from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RecTask:
    name: str
    value: float
    type: Literal["absolute", "ratio"]


@dataclass
class RecConfig:
    tasks: Tuple[RecTask, ...] = (
        RecTask(name="rec1", value=1, type="absolute"),
        RecTask(name="rec15p", value=0.15, type="ratio"),
    )


@dataclass(frozen=True)
class CompatibleConfig:
    value: float = 1.0
    type: Literal["absolute", "ratio"] = "ratio"


@dataclass
class RolloutConfig:
    ks: Tuple[int, ...] = (1, 2, 3, 5)
    horizon_decay: Literal["inv", "exp", "none"] = "inv"


@dataclass
class ModelConfig:
    alphabet_size: int
    num_labels: int = 2
    num_latent_states: int = 64
    d_model: int = 256
    n_head: int = 8
    n_layer: int = 4
    dim_ff: int = 1024
    dropout: float = 0.05
    max_len: int = 64

    @property
    def bos_id(self) -> int:
        return self.alphabet_size

    @property
    def pad_id(self) -> int:
        return self.alphabet_size + 1

    @property
    def vocab_size(self) -> int:
        return self.alphabet_size + 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MLPReadout(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layer)
        self.rec_head = nn.Linear(cfg.d_model, cfg.alphabet_size)
        self.rtd_head = nn.Linear(cfg.d_model, 1)
        self.state_head = nn.Linear(cfg.d_model, cfg.num_latent_states)
        self.compat_head = nn.Linear(cfg.d_model, cfg.alphabet_size)

        state_hidden = max(cfg.num_latent_states, min(cfg.dim_ff, max(64, 2 * cfg.num_latent_states)))
        token_hidden = max(cfg.d_model, min(cfg.dim_ff, max(64, cfg.d_model)))

        self.cls_state_head = nn.Linear(cfg.num_latent_states, cfg.num_labels)
        self.cls_state_mlp_head = MLPReadout(cfg.num_latent_states, cfg.num_labels, state_hidden, cfg.dropout)

        self.cls_roll_head = nn.Linear(cfg.num_latent_states, cfg.num_labels)
        self.cls_roll_mlp_head = MLPReadout(cfg.num_latent_states, cfg.num_labels, state_hidden, cfg.dropout)

        self.cls_delta_head = nn.Linear(cfg.num_latent_states, cfg.num_labels)
        self.cls_delta_mlp_head = MLPReadout(cfg.num_latent_states, cfg.num_labels, state_hidden, cfg.dropout)

        self.cls_token_head = nn.Linear(cfg.d_model, cfg.num_labels)
        self.cls_token_mlp_head = MLPReadout(cfg.d_model, cfg.num_labels, token_hidden, cfg.dropout)

        self.T_logits = nn.Parameter(torch.zeros(cfg.alphabet_size, cfg.num_latent_states, cfg.num_latent_states))
        nn.init.normal_(self.T_logits, mean=0.0, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.embed(input_ids)
        x = self.pos(x)
        key_padding = attention_mask == 0
        T = input_ids.size(1)
        h_bi = self.encoder(x, src_key_padding_mask=key_padding)
        causal = self._causal_mask(T, input_ids.device)
        h_ca = self.encoder(x, mask=causal, src_key_padding_mask=key_padding)
        return {
            "h_bi": h_bi,
            "h_ca": h_ca,
            "rec_logits": self.rec_head(h_bi),
            "rtd_logits": self.rtd_head(h_bi).squeeze(-1),
            "state_logits": self.state_head(h_ca),
            "compat_logits": self.compat_head(h_bi[:, 1:, :]),
        }

    def cls_logits(self, h_ca: torch.Tensor, seq_end_idx: torch.Tensor) -> torch.Tensor:
        return self.cls_token_head(gather_by_index(h_ca, seq_end_idx))

    def cls_logits_from_token(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.cls_token_head(token_features)

    def cls_mlp_logits_from_token(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.cls_token_mlp_head(token_features)

    def cls_logits_from_state(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.cls_state_head(state_features)

    def cls_mlp_logits_from_state(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.cls_state_mlp_head(state_features)

    def cls_roll_logits(self, roll_features: torch.Tensor) -> torch.Tensor:
        return self.cls_roll_head(roll_features)

    def cls_roll_mlp_logits(self, roll_features: torch.Tensor) -> torch.Tensor:
        return self.cls_roll_mlp_head(roll_features)

    def cls_delta_logits(self, delta_features: torch.Tensor) -> torch.Tensor:
        return self.cls_delta_head(delta_features)

    def cls_delta_mlp_logits(self, delta_features: torch.Tensor) -> torch.Tensor:
        return self.cls_delta_mlp_head(delta_features)

    def T_probs_float(self, temp: float = 1.0) -> torch.Tensor:
        return F.softmax(self.T_logits.float() / float(temp), dim=-1)


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)


def gather_by_index(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    b = torch.arange(x.size(0), device=x.device)
    return x[b, idx.long()]


def last_nonpad_index(attn: torch.Tensor) -> torch.Tensor:
    return attn.long().sum(dim=1).clamp_min(1) - 1


def masked_label_smoothing_ce(logits: torch.Tensor, labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    V = logits.size(-1)
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    if valid.sum().item() == 0:
        return flat_logits.sum() * 0.0
    x = flat_logits[valid]
    y = flat_labels[valid]
    log_probs = F.log_softmax(x, dim=-1)
    nll = -log_probs.gather(dim=-1, index=y.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()


def kl_stopgrad_p_to_q(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.detach()
    return (p * (torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12)))).sum(dim=-1).mean()


def aux_forward_next_state(T_probs: torch.Tensor, p_state_t: torch.Tensor, tokens_t: torch.Tensor) -> torch.Tensor:
    M = T_probs.index_select(0, tokens_t.reshape(-1)).reshape(tokens_t.size(0), tokens_t.size(1), T_probs.size(1), T_probs.size(2))
    return torch.matmul(p_state_t.unsqueeze(2), M).squeeze(2)


def rollout_from_bos(T_probs: torch.Tensor, p0: torch.Tensor, tokens: torch.Tensor, valid_tok: torch.Tensor, alphabet_size: int) -> torch.Tensor:
    B, Ttok = tokens.shape
    S = p0.size(-1)
    p_roll = torch.zeros((B, Ttok + 1, S), device=p0.device, dtype=p0.dtype)
    p = p0
    p_roll[:, 0, :] = p0
    for t in range(Ttok):
        tok_t = tokens[:, t].clamp(0, alphabet_size - 1)
        T_t = T_probs.index_select(0, tok_t)
        p_next = torch.bmm(p.unsqueeze(1), T_t).squeeze(1)
        m = valid_tok[:, t].unsqueeze(1).float()
        p = m * p_next + (1.0 - m) * p
        p_roll[:, t + 1, :] = p
    return p_roll


def rollout_from_anywhere(T_probs: torch.Tensor, p_states: torch.Tensor, tokens: torch.Tensor, valid_tok: torch.Tensor, max_k: int):
    B, L, S = p_states.shape
    assert tokens.size(1) == L - 1
    rollouts = {}
    masks = {}
    max_h = min(max_k, L - 1)
    for h in range(1, max_h + 1):
        nstart = L - h
        pred = p_states[:, :nstart, :].float()
        mask = torch.ones((B, nstart), device=p_states.device, dtype=torch.bool)
        for s in range(h):
            tok_step = tokens[:, s:s + nstart]
            valid_step = valid_tok[:, s:s + nstart]
            T_step = T_probs.index_select(0, tok_step.reshape(-1)).reshape(B, nstart, S, S)
            pred = torch.matmul(pred.unsqueeze(2), T_step).squeeze(2)
            mask = mask & valid_step
        rollouts[h] = pred
        masks[h] = mask
    return rollouts, masks


def multi_start_rollout_loss(T_probs: torch.Tensor, enc_states: torch.Tensor, tokens: torch.Tensor, valid_tok: torch.Tensor, max_k: int, horizon_decay: str = "inv") -> torch.Tensor:
    rollouts, masks = rollout_from_anywhere(T_probs, enc_states, tokens, valid_tok, max_k)
    total = enc_states.sum() * 0.0
    wsum = 0.0
    for h, pred in rollouts.items():
        mask = masks[h]
        if not mask.any():
            continue
        tgt = enc_states[:, h:, :]
        p = tgt[mask]
        q = pred[mask]
        loss_h = 0.5 * (kl_stopgrad_p_to_q(p, q) + kl_stopgrad_p_to_q(q, p))
        if horizon_decay == "inv":
            w = 1.0 / float(h)
        elif horizon_decay == "exp":
            w = 0.5 ** float(h - 1)
        else:
            w = 1.0
        total = total + w * loss_h
        wsum += w
    return total if wsum == 0.0 else total / wsum
