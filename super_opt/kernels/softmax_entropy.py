# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The entropy of the softmax of every row, ``H = -Σ_j p_j log p_j``, in one pass.

The algorithm is chain_ops' ``entropy_fused``. The state of a set of elements
is ``(m, l, s)``: a reference ``m`` (their maximum), ``l = Σ exp(x - m)`` and
``s = Σ (x - m) exp(x - m)``. Moving a state to a larger reference
``m' = m + d`` (``d >= 0``, ``r = exp(-d)``) gives ``l' = r l`` and
``s' = r (s - d l)``; merging two states moves both to the larger one and adds
(``combine``). An element ``x`` is the state ``(x, 1, 0)``.

The kernel keeps the state in base 2: ``m`` is ``max x * log2 e``, every
element's ``t = x log2 e - m`` is one FMA, its weight one ``ex2``, and
``s`` sums ``t 2^t``; ``s ln 2`` is the natural-base sum. Since
``H = log l - s / l`` holds for *any* common reference, the rounding of
``m`` itself cancels.

The epilogue, as ``entropy_fused`` finishes: ``H = log l - s / l``. Every
term of ``s`` is ``<= 0`` (``x <= max``), and so is every rescaling ``-d l``:
``s`` never cancels, and ``H = log l + (-s) / l`` is a sum of two
non-negative terms (``l >= 1``: the maximum's own term is ``2^0``). ``log l``
is ``log1p(l - 1)``: ``l - 1`` is exact for ``l <= 2`` (a peaked row), and
``log1p`` is the precise libdevice function where ``log`` would be
fast-math's ``__logf``; it runs once per row.

How it is laid out:

* ``splits`` CTAs of 256 threads per row (one when there are enough rows to
  fill the GPU) and a persistent grid, as many CTAs as are resident at once,
  that walks the ``(row, split)`` work items;
* each thread issues ``UNROLL`` independent 16-byte loads (8 bf16 each) at a
  time, takes the maximum of the ``8 UNROLL`` elements, moves its state to it
  only if it grew (one ``ex2``, usually skipped), then adds each element with
  one FMA, one ``ex2``, one add and one FMA; when ``m`` is not a multiple of 8
  the rows are not 16-byte aligned and it reads one element at a time instead
  (decided when the kernel is built);
* the CTA reduction takes the maximum first (a butterfly of ``fmaxf``), moves
  each lane to it (one ``ex2``) and then only adds; the eight warp totals go
  through double-buffered shared memory to warp 0, which finishes the row while
  the other warps go on to the next: one barrier per row;
* with one split, warp 0 writes ``H``; with several, it writes the partial
  state to a cached workspace of ``n * splits * 3`` floats and a second
  kernel (one CTA per row) merges them and writes ``H``.

``variant="unfused"`` is the ablation: the three passes of the unfused
program, ``max``, ``den = Σ exp(x - max)`` and ``H = Σ p (log den - (x -
max))``, each a launch of the same kernel that reads the whole input again;
only the per-row ``max`` and ``den`` travel between them.
"""

from __future__ import annotations

import math

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable

NEG = -3.0e38  # a finite "-inf": the identity of max, and exp(NEG - NEG) is no NaN
L2E = math.log2(math.e)
LN2 = math.log(2.0)
THREADS = 256
WARPS = THREADS // 32
UNROLL = 4  # 16-byte loads in flight per thread
CTAS_PER_SM = 8  # resident 256-thread CTAs per SM (<= 32 registers a thread)
MIN_SPLIT = 8192  # the fewest elements a CTA's part of a split row has (32 per thread)
K = 3  # the state (m, l, s)
FULL_MASK = 0xFFFFFFFF

# the passes: the fused one, and the three of the unfused program
FUSED, MAX, DEN, ENT = "fused", "max", "den", "ent"

KERNEL_META = {
    "op": "softmax_entropy",
    "dtype": "bfloat16 in, float32 out",
    "algorithm": "one pass, state (m, l, s) = (max, Σ 2^t, Σ t 2^t) with t = x log2 e - m; "
    "H = log1p(l - 1) - ln2 s / l",
    "structure": "persistent grid of 256-thread CTAs, one per row (or per part of a row when "
    "rows are few, merged by a second kernel); 4 x 16-byte loads in flight per thread; "
    "max-then-sum warp reduction, one barrier per row",
}


def _fmax(a, b):
    """One ``FMNMX``; ``T.max`` is a NaN-propagating compare and select."""
    return T.call_pure_extern("fmaxf", a, b, ty="float32")


def _fma(a, b, c):
    return T.call_pure_extern("__fmaf_rn", a, b, c, ty="float32")


def _f32(x):
    return T.Cast("float32", x)


# ---------------------------------------------------------------------------
# the state (m, l, s) in base 2
# ---------------------------------------------------------------------------
@T.inline
def init_state(st):
    st[0] = T.float32(NEG)
    st[1] = T.float32(0)
    st[2] = T.float32(0)


@T.inline
def rebase(st, mx):
    """Move ``st`` to the reference ``mx >= st[0]``."""
    d = mx - st[0]
    r = T.exp2(T.float32(0) - d)
    st[2] = r * (st[2] - d * st[1])
    st[1] = r * st[1]
    st[0] = mx


@T.inline
def combine(st, ot):
    """(m, l, s) <- (m, l, s) ⊕ (m', l', s'): move both to the larger reference and add."""
    mx = _fmax(st[0], ot[0])
    da = mx - st[0]
    db = mx - ot[0]
    ra = T.exp2(T.float32(0) - da)
    rb = T.exp2(T.float32(0) - db)
    st[2] = ra * (st[2] - da * st[1]) + rb * (ot[2] - db * ot[1])
    st[1] = ra * st[1] + rb * ot[1]
    st[0] = mx


@T.inline
def warp_reduce(st, tmp):
    """Every lane gets the warp's state: the maximum first, then one rebase, then sums."""
    tmp[0] = st[0]
    for i in T.unroll(5):
        tmp[0] = _fmax(tmp[0], T.gpu_warp_shuffle_xor(FULL_MASK, tmp[0], 16 >> i, 32, 32))
    rebase(st, tmp[0])
    for i in T.unroll(5):
        st[1] = st[1] + T.gpu_warp_shuffle_xor(FULL_MASK, st[1], 16 >> i, 32, 32)
        st[2] = st[2] + T.gpu_warp_shuffle_xor(FULL_MASK, st[2], 16 >> i, 32, 32)


@T.inline
def cta_reduce(st, tmp, smem, buf, warp, lane):
    """Warp 0 gets the CTA's state; one barrier, ``smem[buf]`` alternates between rows."""
    warp_reduce(st, tmp)
    if lane == 0:
        for k in T.unroll(K):
            smem[buf, warp, k] = st[k]
    T.cuda.cta_sync()
    if warp == 0:
        if lane < WARPS:
            for k in T.unroll(K):
                st[k] = smem[buf, lane, k]
        else:
            init_state(st)
        warp_reduce(st, tmp)


@T.inline
def write_entropy(Out, row, st):
    """``H = log l + (-s ln 2) / l``: two non-negative terms."""
    Out[row] = T.log1p(st[1] - T.float32(1)) + (T.float32(-LN2) * st[2]) / st[1]


# ---------------------------------------------------------------------------
# folding elements into a thread's state, per pass
# ---------------------------------------------------------------------------
@T.inline
def add_fused(st, x):
    """The element ``x`` (``x log2 e <= st[0]``): ``t = x log2 e - m``, ``l += 2^t``, ``s += t 2^t``."""  # noqa: E501
    t = _fma(x, T.float32(L2E), T.float32(0) - st[0])
    e = T.exp2(t)
    st[1] = st[1] + e
    st[2] = _fma(t, e, st[2])


@T.inline
def add_den(st, x, par):
    """``l += 2^t`` with ``t = x log2 e - par[0]``."""
    st[1] = st[1] + T.exp2(_fma(x, T.float32(L2E), par[0]))


@T.inline
def add_ent(st, x, par):
    """``l += 2^t (log2 den - t)``: ``p (log den - (x - max))`` times ``den / ln 2``."""
    t = _fma(x, T.float32(L2E), par[0])
    st[1] = _fma(T.exp2(t), par[1] - t, st[1])


def _splits(n: int, m: int) -> int:
    """How many CTAs share a row: enough work items to fill the GPU, parts of at least MIN_SPLIT."""
    want = -(-SMS * CTAS_PER_SM // n)
    return max(1, min(want, m // MIN_SPLIT))


def _workspace(n: int, splits: int) -> int:
    """Floats of partial states: ``n * splits * 3``, or a 1-float placeholder with one split."""
    return n * splits * K if splits > 1 else 1


def _pass_kernel(n: int, m: int, splits: int, mode: str):
    """One pass over every row: ``(row, split)`` work items -> a per-row result or partials ``W``.

    ``mode``: ``FUSED`` writes ``H``; ``MAX`` the row maximum to ``P``; ``DEN``
    ``Σ exp(x - max)`` to ``P2`` (reading the maximum from ``P``); ``ENT`` ``H``
    (reading both). With ``splits > 1`` every mode writes its partial state to
    ``W`` instead, for ``_merge_kernel``.
    """
    vec = vector_width("bfloat16")
    vectorised = vectorisable(m, "bfloat16")
    nvec = m // vec
    # each split's part: whole vectors on the vectorised path, elements otherwise
    part = -(-nvec // splits) if vectorised else -(-m // splits)
    items = n * splits
    blocks = min(items, SMS * CTAS_PER_SM)
    nw = _workspace(n, splits)
    chunk = UNROLL * vec
    fused, is_max, is_den, is_ent = mode == FUSED, mode == MAX, mode == DEN, mode == ENT
    split = splits > 1

    @T.function
    def entropy_pass(
        A: T.Tensor((n, m), "bfloat16", align=16),
        Out: T.Tensor((n,), "float32"),
        P: T.Tensor((n,), "float32"),
        P2: T.Tensor((n,), "float32"),
        W: T.Tensor((nw,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")  # this thread's state
        tmp = T.alloc_local((1,), "float32")
        par = T.alloc_local((2,), "float32")  # the row's -max log2 e and log2 den (unfused)
        buf = T.alloc_local((chunk,), "bfloat16")
        xs = T.alloc_local((chunk,), "float32")
        smem = T.alloc_shared((2, WARPS, K), "float32")
        sbuf: T.int32 = 0
        item: T.int32 = bx
        while item < items:
            row = item // splits
            sp = item % splits
            if fused or is_max:
                init_state(st)
            else:
                st[0] = T.float32(0)
                st[1] = T.float32(0)
                st[2] = T.float32(0)
                par[0] = T.float32(-L2E) * P[row]
                if is_ent:
                    par[1] = T.log1p(P2[row] - T.float32(1)) * T.float32(1.0 / LN2)
            if vectorised:
                v: T.int32 = sp * part + tx
                vend = T.min((sp + 1) * part, nvec)
                while v < vend:
                    # UNROLL independent 16-byte loads; vector u is there when v + u THREADS < vend
                    for u in T.unroll(UNROLL):
                        if v + u * THREADS < vend:
                            buf[T.Ramp(u * vec, 1, vec)] = A[
                                row, T.Ramp((v + u * THREADS) * vec, 1, vec)
                            ]
                    for k in T.unroll(chunk):
                        xs[k] = _f32(buf[k])
                    if fused or is_max:
                        mx: T.float32 = xs[0]
                        for u in T.unroll(UNROLL):
                            if v + u * THREADS < vend:
                                for k in T.unroll(vec):
                                    mx = _fmax(mx, xs[u * vec + k])
                        if is_max:
                            st[0] = _fmax(st[0], mx)
                        else:
                            mx = mx * T.float32(L2E)
                            if mx > st[0]:
                                rebase(st, mx)
                    if not is_max:
                        for u in T.unroll(UNROLL):
                            if v + u * THREADS < vend:
                                for k in T.unroll(vec):
                                    if fused:
                                        add_fused(st, xs[u * vec + k])
                                    elif is_den:
                                        add_den(st, xs[u * vec + k], par)
                                    else:
                                        add_ent(st, xs[u * vec + k], par)
                    v += UNROLL * THREADS
            else:
                j: T.int32 = sp * part + tx
                jend = T.min((sp + 1) * part, m)
                while j < jend:
                    xs[0] = _f32(A[row, j])
                    if is_max:
                        st[0] = _fmax(st[0], xs[0])
                    elif fused:
                        tmp[0] = xs[0] * T.float32(L2E)
                        if tmp[0] > st[0]:
                            rebase(st, tmp[0])
                        add_fused(st, xs[0])
                    elif is_den:
                        add_den(st, xs[0], par)
                    else:
                        add_ent(st, xs[0], par)
                    j += THREADS
            cta_reduce(st, tmp, smem, sbuf, warp, lane)
            if tx == 0:
                if split:
                    for k in T.unroll(K):
                        W[item * K + k] = st[k]
                elif fused:
                    write_entropy(Out, row, st)
                elif is_max:
                    P[row] = st[0]
                elif is_den:
                    P2[row] = st[1]
                else:
                    Out[row] = st[1] * T.float32(LN2) / P2[row]
            sbuf = 1 - sbuf
            item += blocks

    return entropy_pass


def _merge_kernel(n: int, splits: int, mode: str):
    """The ``splits`` partial states of every row -> the pass's per-row result; a CTA per row."""
    blocks = min(n, SMS * CTAS_PER_SM)
    nw = _workspace(n, splits)
    fused, is_max, is_den = mode == FUSED, mode == MAX, mode == DEN

    @T.function
    def entropy_merge(
        W: T.Tensor((nw,), "float32"),
        Out: T.Tensor((n,), "float32"),
        P: T.Tensor((n,), "float32"),
        P2: T.Tensor((n,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")
        ot = T.alloc_local((K,), "float32")
        tmp = T.alloc_local((1,), "float32")
        smem = T.alloc_shared((2, WARPS, K), "float32")
        sbuf: T.int32 = 0
        row: T.int32 = bx
        while row < n:
            init_state(st)
            p: T.int32 = tx
            while p < splits:
                for k in T.unroll(K):
                    ot[k] = W[(row * splits + p) * K + k]
                combine(st, ot)
                p += THREADS
            cta_reduce(st, tmp, smem, sbuf, warp, lane)
            if tx == 0:
                if fused:
                    write_entropy(Out, row, st)
                elif is_max:
                    P[row] = st[0]
                elif is_den:
                    P2[row] = st[1]
                else:
                    Out[row] = st[1] * T.float32(LN2) / P2[row]
            sbuf = 1 - sbuf
            row += blocks

    return entropy_merge


def _modules(n: int, m: int, variant: str):
    """``[(name, (pass module, merge module or None))]``, in launch order."""
    splits = _splits(n, m)
    modes = [FUSED] if variant == "fused" else [MAX, DEN, ENT]
    out = []
    for mode in modes:
        name = f"softmax_entropy_{mode}_{n}x{m}"
        pass_mod = compile_sm90(_pass_kernel(n, m, splits, mode), name=name)
        merge_mod = None
        if splits > 1:
            merge_mod = compile_sm90(_merge_kernel(n, splits, mode), name=name + "_merge")
        out.append((name, (pass_mod, merge_mod)))
    return out


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, m = cfg["n"], cfg["m"]
    splits = _splits(n, m)
    passes = [mods for _, mods in _modules(n, m, variant)]
    scratch = {}  # per device, allocated once: the workspace, and the unfused per-row max and den

    def buffers(device):
        if device not in scratch:
            work = torch.empty(_workspace(n, splits), device=device, dtype=F32)
            rows = [torch.empty(n, device=device, dtype=F32) for _ in range(2)]
            scratch[device] = (work, *rows) if variant == "unfused" else (work, None, None)
        return scratch[device]

    def run(X):
        out = torch.empty(n, device=X.device, dtype=F32)
        work, mx, den = buffers(X.device)
        if variant == "fused":
            mx = den = out  # unread: the fused pass takes them only to share the signature
        for pass_mod, merge_mod in passes:
            pass_mod(X, out, mx, den, work)
            if merge_mod is not None:
                merge_mod(work, out, mx, den)
        return (out,)

    return run
