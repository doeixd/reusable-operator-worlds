"""E5: an amortized writer `q_phi(p | D_support)` over a FROZEN learned vocabulary.

`E5_SYNTHESIZER_PLAN.md` (Amendment 1). The sealed block established that short
discrete programs over the learned library solve structurally novel tasks; this
model asks whether such a program can be WRITTEN from support examples in one
forward pass, rather than searched for.

Permutation-invariant by construction (the standard deep-sets form): each
support pair `(x, y)` is embedded independently, the set is mean-pooled, and `D`
independent softmax heads read out one slot index per program position. The
library is never touched — the recognizer only proposes programs for it to run.

Trained against each task's OWN argmax route, which E3 established IS that
task's program bitwise, so no teacher label enters training.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ProgramRecognizer(nn.Module):
    def __init__(self, d: int, slots: int, depth: int, hidden: int = 256,
                 embedding: int = 128, seed: int = 5000) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.d, self.slots, self.depth = int(d), int(slots), int(depth)
        self.pair = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, embedding),
        )
        self.trunk = nn.Sequential(
            nn.Linear(embedding, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden, slots) for _ in range(self.depth))

    def forward(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        """(n, d), (n, d) -> (depth, slots) logits. Order of the pairs is irrelevant."""
        pooled = self.pair(torch.cat([support_x, support_y], dim=-1)).mean(dim=0)
        trunk = self.trunk(pooled)
        return torch.stack([head(trunk) for head in self.heads], dim=0)

    @torch.no_grad()
    def top_k_programs(self, support_x: Tensor, support_y: Tensor, k: int) -> list[list[int]]:
        """The `k` most probable programs under the factorized posterior.

        The heads are independent, so the joint top-`k` is obtained by expanding
        the best partial prefixes — exact for this factorization, and `k` is tiny.
        """
        logprobs = torch.log_softmax(self(support_x, support_y), dim=-1)
        beams: list[tuple[float, list[int]]] = [(0.0, [])]
        for step in range(self.depth):
            scored = []
            for score, prefix in beams:
                for slot in range(self.slots):
                    scored.append((score + float(logprobs[step, slot]), prefix + [slot]))
            scored.sort(key=lambda item: -item[0])
            beams = scored[:max(k, 1)]
        return [program for _, program in beams[:k]]
