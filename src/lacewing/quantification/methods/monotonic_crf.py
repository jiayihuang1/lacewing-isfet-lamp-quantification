"""Hand-rolled linear-chain CRF with a hardcoded monotonic transition matrix.

Motivation: the F-D 4-class U-Net collapses on 66.5% of test pixels because
nothing in cross-entropy training prevents the model from predicting
backward class transitions (e.g. rising -> baseline). This module adds a
linear-chain CRF layer whose transition matrix is *fixed* (not learned) to
only allow "stay" (c -> c) or "advance" (c -> c+1) transitions between the
4 ordered classes {baseline, drift, rising, post-amp}. Any other transition
is assigned an effectively -inf score, so the Viterbi decode can never
produce a non-monotonic path, and the CRF NLL loss pushes the emission
logits towards paths that are monotonic.

Emissions come from the existing U-Net's per-timestep 4-class logits — no
change to the backbone architecture, only to the loss/decoding on top.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NEG_INF = float("-inf")


class MonotonicCRF(nn.Module):
    """Linear-chain CRF with a hardcoded monotonic (non-decreasing) transition matrix."""

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.n_classes = n_classes
        # Fixed transitions — NOT nn.Parameter (never learned/updated).
        transitions = torch.full((n_classes, n_classes), NEG_INF)
        for c in range(n_classes):
            transitions[c, c] = 0.0  # stay
            if c + 1 < n_classes:
                transitions[c, c + 1] = 0.0  # advance
        self.register_buffer("transitions", transitions)

    def forward(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """NLL loss. emissions: (B, T, C). labels: (B, T) int64. Returns scalar loss."""
        score_label = self._score_path(emissions, labels)   # (B,)
        log_Z = self._forward_algorithm(emissions)           # (B,)
        return -(score_label - log_Z).mean()

    def decode(self, emissions: torch.Tensor) -> torch.Tensor:
        """Viterbi decode. emissions: (B, T, C). Returns labels: (B, T) int64."""
        B, T, C = emissions.shape
        device = emissions.device
        transitions = self.transitions  # (C, C) — transitions[c_prev, c_next]

        # dp[b, c] = best score of a path ending in class c at current t.
        dp = emissions[:, 0, :].clone()               # (B, C)
        backptr = torch.zeros((B, T, C), dtype=torch.long, device=device)

        for t in range(1, T):
            # candidate[b, c_prev, c] = dp_prev[b, c_prev] + transitions[c_prev, c]
            candidate = dp.unsqueeze(2) + transitions.unsqueeze(0)   # (B, C_prev, C)
            best_prev_score, best_prev_idx = candidate.max(dim=1)     # (B, C), (B, C)
            dp = best_prev_score + emissions[:, t, :]
            backptr[:, t, :] = best_prev_idx

        # Backtrack from the best final class.
        path = torch.zeros((B, T), dtype=torch.long, device=device)
        path[:, T - 1] = dp.argmax(dim=1)
        for t in range(T - 1, 0, -1):
            path[:, t - 1] = backptr[torch.arange(B, device=device), t, path[:, t]]
        return path

    def _score_path(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum of emissions[t, labels[t]] + transitions[labels[t-1], labels[t]] over t."""
        B, T, C = emissions.shape
        # Emission scores along the label path.
        emit_scores = emissions.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
        score = emit_scores.sum(dim=1)  # (B,)

        if T > 1:
            prev_labels = labels[:, :-1]   # (B, T-1)
            next_labels = labels[:, 1:]    # (B, T-1)
            trans_scores = self.transitions[prev_labels, next_labels]  # (B, T-1)
            score = score + trans_scores.sum(dim=1)
        return score

    def _forward_algorithm(self, emissions: torch.Tensor) -> torch.Tensor:
        """log-sum-exp forward algorithm to compute log partition function."""
        B, T, C = emissions.shape
        transitions = self.transitions  # (C, C)

        # alpha[b, c] = log-sum-exp over all paths ending in class c at t.
        alpha = emissions[:, 0, :].clone()   # (B, C)
        for t in range(1, T):
            # broadcast: (B, C_prev, 1) + (1, C_prev, C) -> (B, C_prev, C)
            scores = alpha.unsqueeze(2) + transitions.unsqueeze(0)
            alpha = torch.logsumexp(scores, dim=1) + emissions[:, t, :]  # (B, C)
        log_Z = torch.logsumexp(alpha, dim=1)  # (B,)
        return log_Z
