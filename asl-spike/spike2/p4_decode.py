"""P4: decoders for the phrase-mode CTC model.

greedy        argmax per frame, collapse repeats, drop blanks
beam          CTC prefix beam search (width 10) with a gloss bigram LM (café script lines + generator grammar),
              score = log p_ctc + LM_W * log p_lm + INS_BONUS * length
script        CTC forced-alignment log-likelihood of every candidate customer line (the 30 café gloss lines);
              top 3. Out-of-script fallback: if the free (beam) decode's CTC log-likelihood beats the best
              script line by more than TAU nats, the free decode is returned instead (TAU tuned on validation).
rerank        optional Claude reranker (claude-haiku-5-5, structured JSON, low effort): barista's previous line,
              top-5 free decodes and top-3 script candidates -> top-3 English sentences.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as Fn

from p3_model import BLANK, UNK_ID

LM_W = 0.5
INS_BONUS = 0.5
BEAM = 10


def greedy(lp: np.ndarray):
    """lp: (T, 145) log-probs."""
    best = lp.argmax(1)
    out, prev = [], -1
    for b in best:
        if b != prev and b != BLANK:
            out.append(int(b))
        prev = b
    return out


class BigramLM:
    def __init__(self, seqs, n_tok=145, add=0.05):
        # tokens 0..143 (143 = <unk>), BOS = 145, EOS = 146
        self.C = np.full((147, 147), add)
        for s in seqs:
            s2 = [145] + list(s) + [146]
            for a, b in zip(s2, s2[1:]):
                self.C[a, b] += 1
        self.L = np.log(self.C / self.C.sum(1, keepdims=True))

    def score(self, prev, tok):
        return self.L[145 if prev is None else prev, tok]

    def end(self, prev):
        return self.L[145 if prev is None else prev, 146]


def beam_search(lp: np.ndarray, lm: BigramLM | None, beam=BEAM, lm_w=LM_W, ins=INS_BONUS):
    """CTC prefix beam search with a bigram LM. Returns list of (prefix, score) sorted best first."""
    # beams: prefix(tuple) -> (p_blank, p_nonblank) in log space
    NEG = -1e30
    beams = {(): (0.0, NEG)}
    lm_cache = {}

    def lmscore(prefix):
        if lm is None:
            return 0.0
        if prefix not in lm_cache:
            s, prev = 0.0, None
            for t in prefix:
                s += lm.score(prev, t)
                prev = t
            lm_cache[prefix] = s
        return lm_cache[prefix]

    def lse(a, b):
        if a < b:
            a, b = b, a
        return a if b <= NEG else a + math.log1p(math.exp(b - a))
    T = lp.shape[0]
    top_k = np.argsort(-lp, axis=1)[:, :8]  # prune per frame
    for t in range(T):
        nb = defaultdict(lambda: (NEG, NEG))
        cands = set(int(c) for c in top_k[t]) | {BLANK}
        for prefix, (pb, pnb) in beams.items():
            ptot = lse(pb, pnb)
            for c in cands:
                p = float(lp[t, c])
                if c == BLANK:
                    b_, nb_ = nb[prefix]
                    nb[prefix] = (lse(b_, ptot + p), nb_)
                    continue
                last = prefix[-1] if prefix else None
                new = prefix + (c,)
                b_, nb_ = nb[new]
                if c == last:
                    nb[new] = (b_, lse(nb_, pb + p))
                    b2, nb2 = nb[prefix]
                    nb[prefix] = (b2, lse(nb2, pnb + p))
                else:
                    nb[new] = (b_, lse(nb_, ptot + p))
        scored = sorted(nb.items(), key=lambda kv: -(lse(*kv[1]) + lm_w * lmscore(kv[0]) + ins * len(kv[0])))
        beams = dict(scored[:beam])

    def final(prefix, v):
        e = lm.end(prefix[-1] if prefix else None) if lm is not None else 0.0
        return lse(*v) + lm_w * (lmscore(prefix) + e) + ins * len(prefix)
    return sorted([(list(k), final(k, v)) for k, v in beams.items()], key=lambda kv: -kv[1])


def ctc_loglik(lp: torch.Tensor, seqs: list[list[int]]):
    """log p(seq | x) for several label sequences. lp: (T, 145) log-probs tensor."""
    T = lp.shape[0]
    out = []
    for s in seqs:
        if len(s) == 0:
            out.append(float(lp[:, BLANK].sum()))
            continue
        loss = Fn.ctc_loss(lp[:, None, :], torch.tensor([s]), torch.tensor([T]), torch.tensor([len(s)]),
                           blank=BLANK, reduction="sum", zero_infinity=True)
        out.append(-float(loss))
    return np.array(out)


def script_decode(lp: torch.Tensor, lines: list[list[int]], free: list[int] | None = None, tau: float | None = None):
    """Returns (ranked line indices top 3, scores, used_fallback)."""
    ll = ctc_loglik(lp, lines)
    order = list(np.argsort(-ll)[:3])
    used = False
    if free is not None and tau is not None:
        fl = ctc_loglik(lp, [free])[0]
        used = bool(fl - ll[order[0]] > tau)
    return order, ll, used


def wer(ref, hyp):
    D = np.zeros((len(ref) + 1, len(hyp) + 1), int)
    D[:, 0] = np.arange(len(ref) + 1)
    D[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return D[-1, -1], len(ref)


def align_ops(ref, hyp):
    """Levenshtein alignment -> list of (op, ref_tok, hyp_tok) for error analysis."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops


def lcs(a, b):
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            D[i + 1][j + 1] = D[i][j] + 1 if a[i] == b[j] else max(D[i][j + 1], D[i + 1][j])
    return D[-1][-1]


def vocab_pr(ref, hyp):
    """recall/precision counts of vocabulary signs (excluding <unk>), order-preserving (round-1 method)."""
    r = [t for t in ref if t != UNK_ID]
    h = [t for t in hyp if t != UNK_ID]
    return lcs(r, h), len(r), len(h)
