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
"""Softmax of every row of a bfloat16 ``(n, m)`` matrix, in TIRx.

The algorithm is chain_ops' ``softmax_fused``: one pass for the online-softmax
state ``(m, l)`` -- a running maximum and the sum of ``exp(x - m)`` -- whose
combine keeps the larger maximum and rescales both sums to it
(``lse_combine``); then the elementwise epilogue ``p = exp(x - m) / l``. A
thread folds its elements into a state of its own by two sweeps over registers
(their maximum, then one exponential per element), and states are merged only
across threads.

Every exponential is ``exp(x - m) = ex2((x - m) log2(e))`` with ``x <= m``
(``_exp``): the difference first, so it is exact at the maximum (``ex2(0) = 1``)
and finite for every finite input. The form with one instruction fewer,
``ex2(fma(x, log2 e, -m log2 e))``, breaks on both counts: ``-m log2 e``
overflows float32 past ``|m| = 2.36e38``, which masks of
``finfo(bfloat16).min = -3.39e38`` reach, and its rounding swamps the exponent
past ``|m| ~ 1e9``. An epilogue folds ``1 / l`` into the exponent instead,
``ex2(fma(x - m, log2 e, -log2 l))`` (``_exp_over``). The maximum starts at
``NEG = -FLT_MAX``, below every finite bfloat16, so a thread's maximum is that
of its finite elements; ``-inf`` elements (masks) add ``exp(-inf) = 0``; a
thread that holds nothing else keeps ``(NEG, 0)``, the combine's identity, and
a row of ``-inf`` only is NaN, as in torch.

The epilogue needs the row's final state, so the row is visited twice; where it
is kept in between decides the layouts (``plan``):

``resident`` (aligned rows a CTA, or a cluster of CTAs, can hold on chip)
    ``C`` CTAs per row (a thread-block cluster when ``C > 1``), one launch over
    all rows. Every thread loads its vectors of the row once, 16 bytes at a
    time: ``kr`` into registers and ``ks`` into its own slots of shared memory
    by ``cp.async``. The CTAs of a cluster exchange their states through
    distributed shared memory. HBM is read once and written once. Long rows are
    sized so that about 256 KB of rows sit on each SM (four CTAs of 256 threads
    holding 16 vectors each): while one CTA reduces, the others' loads keep HBM
    busy. For short rows (``keep``) the exponentials of the state stay in
    float32 registers and the epilogue only scales them.
``rounds`` (few aligned rows, each spread over the whole GPU)
    one cooperative launch of ``G`` CTAs, all resident, the grid on one row at a
    time. CTA ``c`` holds its slice of the row on chip (registers, and shared
    memory filled by ``cp.async``) and computes the slice's state, leaving each
    element's ``exp(x - m_t)`` (``m_t`` the thread's maximum) in its place as
    float16, so the epilogue costs a multiply, not a second exponential. It then
    publishes the state, prefetches its slice of the next row into L2
    (``cp.async.bulk.prefetch``: HBM stays busy while the grid waits) and waits at
    the row's barrier; then it merges the row's ``G`` states, normalises its
    slice and, slot by slot, loads the next row's slice in its place. A row
    longer than the chip holds (``sp > 0``) streams the first ``sp`` vectors of
    every slice through registers into the state, keeps the rest, and after the
    barrier reads the streamed part again, newest first, while L2 still holds
    some of it. The barrier is a sense-reversing counter per row: CTA 0 adds
    ``2^31 - (G - 1)`` and the others 1, so the last arrival flips the top bit
    the waiters watch, and the low bits are 0 again. Between calls the counter
    holds 0 or ``2^31``: it cannot wrap, however many calls it serves.
``cache`` / ``stream`` / ``split`` (rows whose length is not a multiple of 8)
    one element at a time: a CTA per row with the row in registers, a CTA per
    row streaming it twice, or two kernels -- ``partial`` writes each chunk's
    state, and ``normalize`` merges the row's states and writes its chunk.

``variant="unfused"`` is the ablation: the unfused program's three passes --
the row maximum, then ``Σ exp(x - max)``, then the output -- as three launches
that each read X again, with only per-row (or per-chunk) maxima and sums
between them, on the same grid sizes, loads and reductions (``passes``).

The state and all arithmetic are float32; the only rounding to bfloat16 is the
store.

Workspaces (the split's and the passes' per-chunk states, the rounds' states
and counters) are kept per CUDA stream (``_per_stream``): the kernels run on
torch's current stream, so calls on one stream run in order and share them,
calls on two streams never do, and a call made while a CUDA graph is captured
gets its own.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90, ctas_per_sm
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce, fmax

# the identity of max: -FLT_MAX, below every finite bfloat16 (finfo.min is -3.39e38), and
# finite, so that exp(NEG - NEG) is 1, not NaN
NEG = -3.4028234663852886e38
L2E = 1.4426950408889634  # log2(e)
VEC = vector_width("bfloat16")  # 8 bf16 = 16 bytes
THREADS_PER_SM = 2048
SMEM_PER_SM = 233472  # bytes of shared memory an SM shares among its CTAs (228 KB)
SMEM_RESERVED = 1024  # bytes the runtime keeps per CTA
CACHE_MAX_SCALAR = 8192  # longest unaligned row kept in registers
R_MAX_SCALAR = 32  # elements per thread in the unaligned register cache
U = 4  # vectors in flight per thread in the streaming loops
SPLIT_THREADS = 256
SPLIT_MIN_CHUNK = 8192  # elements
LONG_THREADS = 256  # resident long rows: threads of a CTA
LONG_SLOTS = 16  # resident long rows: vectors a thread holds (registers + shared memory)
MAX_CLUSTER = 8  # the portable cluster size
PF_VECS = 1024  # rounds: vectors per L2 prefetch instruction (16 KB)

KERNEL_META = {
    "op": "softmax",
    "dtype": "bfloat16 in, bfloat16 out",
    "algorithm": "one pass, state (m, l) = (max, Σ exp(x - max)); then p = exp(x - m) / l",
    "structure": (
        "aligned rows: one CTA (or a cluster of CTAs, states exchanged through distributed "
        "shared memory) per row, the row on chip between the passes (registers, plus shared "
        "memory filled by cp.async), read once, ~256 KB of rows per SM; few long rows: one "
        "cooperative launch, the grid on one row at a time, each CTA's slice on chip with "
        "its exponentials kept as float16, an atomic barrier per row with the next row "
        "prefetched into L2 (rows past the chip re-read the rest newest first); unaligned "
        "rows: one element at a time; 16-byte loads/stores, warp then CTA all-reduce; "
        "unfused: max, Σ exp, output as three streaming launches"
    ),
}


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(a: int, b: int) -> int:
    return _ceil(a, b) * b


def _pow2(x: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1, x))))


@dataclasses.dataclass(frozen=True)
class Plan:
    path: str  # "resident", "rounds", "cache", "stream", "split" or "passes"
    threads: int
    vectorised: bool
    per_thread: int = 0  # cache: elements each thread holds (unaligned)
    chunk: int = 0  # split / passes: units per CTA; rounds: vectors per CTA
    chunks: int = 0  # split / passes: CTAs per row
    kr: int = 0  # resident / rounds: vectors per thread in registers
    ks: int = 0  # resident / rounds: vectors per thread in shared memory
    cluster: int = 1  # resident: CTAs per row
    minb: int = 1  # resident / rounds: CTAs an SM must fit (__launch_bounds__)
    keep: bool = False  # resident: keep exp(x - m_thread) in float32 registers
    ctas: int = 0  # rounds: the grid, all resident
    sp: int = 0  # rounds: vectors of each slice streamed (read twice) before the kept ones
    prefetch: bool = True  # rounds: prefetch the next row's slice into L2 at the barrier
    u: int = U  # rounds / passes: vectors a thread has in flight in a streaming loop


def _resident_plan(m: int) -> Plan | None:
    """A row of ``m // 8`` vectors held on chip: up to 4 vectors a thread in registers
    (short rows, at most 64 registers a thread); past that CTAs of 128 or 256 threads
    holding 16 vectors each, 4 in registers and 12 in shared memory, about 256 KB of rows
    per SM, in a cluster of as many CTAs as the row needs."""
    if m % VEC:
        return None
    nvec = m // VEC
    if nvec <= 1024:
        threads = min(256, max(64, _pow2(_ceil(nvec, 4))))
        kr = _ceil(nvec, threads)
        return Plan("resident", threads, True, kr=kr, minb=max(1, 1024 // threads), keep=True)
    threads = min(LONG_THREADS, max(128, _pow2(_ceil(nvec, LONG_SLOTS))))
    cluster = _pow2(_ceil(nvec, threads * LONG_SLOTS))
    if cluster > MAX_CLUSTER:
        return None
    slots = _ceil(_ceil(nvec, cluster), threads)
    kr = min(4, slots)
    return Plan(
        "resident",
        threads,
        True,
        kr=kr,
        ks=slots - kr,
        cluster=cluster,
        minb=THREADS_PER_SM // 2 // threads,
    )


def _rounds_shape(
    m: int, threads: int, per_sm: int, kr: int, ks_cap: int | None = None
) -> Plan | None:
    """The rounds on ``SMS * per_sm`` CTAs of ``threads``, or None when the row is too
    short to give every thread a vector: ``kr`` vectors a thread in registers, as many in
    shared memory as ``per_sm`` CTAs leave room for (at most ``ks_cap``), and the rest of
    the slice streamed (``sp``)."""
    if m % VEC:
        return None
    units = m // VEC
    ctas = SMS * per_sm
    chunk = _ceil(units, ctas)
    if chunk < threads:
        return None
    slots = _ceil(chunk, threads)
    room = SMEM_PER_SM // per_sm - SMEM_RESERVED - 64 * (threads // 32) - 64
    ks_max = room // (threads * VEC * 2)
    if ks_cap is not None:
        ks_max = min(ks_max, ks_cap)
    kr = min(kr, slots)
    ks = min(slots - kr, ks_max)
    return Plan(
        "rounds",
        threads,
        True,
        chunk=chunk,
        kr=kr,
        ks=ks,
        minb=per_sm,
        ctas=ctas,
        sp=max(0, chunk - (kr + ks) * threads),
    )


def _rounds_plan(n: int, m: int) -> Plan | None:
    """Rows whose slices the chip holds: two CTAs of 512 threads an SM (fewer CTAs at the
    barriers); rows longer than that stream part of every slice on four CTAs of 256."""
    p = _rounds_shape(m, 512, 2, 2)
    if p is not None and p.sp == 0:
        return p
    return _rounds_shape(m, 256, 4, 4)


def _passes_plan(n: int, m: int, u: int = U) -> Plan:
    """The unfused passes: CTAs of up to 512 threads, ``u`` vectors (or ``8 u`` elements) a
    thread per step, rows cut into as many chunks as it takes to fill the GPU."""
    vectorised = vectorisable(m, "bfloat16")
    units = m // VEC if vectorised else m
    per = u if vectorised else u * VEC  # units a thread takes per step
    threads = min(512, max(64, _pow2(_ceil(units, per))))
    want = _ceil(SMS * (THREADS_PER_SM // threads), n)  # CTAs per row to fill the GPU
    chunk = _round_up(_ceil(units, want), threads * per)
    return Plan("passes", threads, vectorised, chunk=chunk, chunks=_ceil(units, chunk), u=u)


def plan(n: int, m: int, path: str | None = None) -> Plan:
    """How ``(n, m)`` is laid out; ``path`` forces one (for tests)."""
    vectorised = vectorisable(m, "bfloat16")
    if path is None:
        res = _resident_plan(m)
        if res is not None and n * res.cluster >= SMS:
            return res
        rounds = _rounds_plan(n, m)
        if rounds is not None:
            return rounds
        if res is not None:
            return res
        if not vectorised and m <= CACHE_MAX_SCALAR:
            path = "cache"
        elif n >= SMS:
            path = "stream"
        else:
            path = "split"
    if path == "resident":
        res = _resident_plan(m)
        assert res is not None, (n, m)
        return res
    if path == "rounds":
        rounds = _rounds_plan(n, m)
        assert rounds is not None, (n, m)
        return rounds
    if path == "passes":
        return _passes_plan(n, m)
    if path == "cache":
        assert not vectorised, (n, m)
        threads = min(1024, max(128, _round_up(_ceil(m, R_MAX_SCALAR), 32)))
        return Plan("cache", threads, False, per_thread=_ceil(m, threads))
    if path == "stream":
        return Plan("stream", 512, vectorised)
    assert path == "split", path
    threads = SPLIT_THREADS
    # a chunk is a whole number of these, so every chunk starts 16-byte aligned
    step = threads * VEC
    want = _ceil(SMS * (THREADS_PER_SM // threads), n)  # CTAs per row to fill the GPU
    chunk = max(SPLIT_MIN_CHUNK, _round_up(_ceil(m, want), step))
    return Plan("split", threads, vectorised, chunk=chunk, chunks=_ceil(m, chunk))


# ---------------------------------------------------------------------------
# the state (m, l) and its pieces
# ---------------------------------------------------------------------------
def _exp(x, m):
    """``exp(x - m)`` for ``x <= m`` (float32), as ``ex2((x - m) log2 e)``: exactly 1 at
    ``x == m``, 0 at ``x = -inf``, and never NaN for a finite ``m`` (see the module notes)."""
    return T.exp2((x - m) * T.float32(L2E))


def _exp_over(x, m, nl):
    """``exp(x - m) / l`` as ``ex2(fma(x - m, log2 e, -log2 l))``, ``nl = -log2 l``: the
    division folded into the exponent's FMA. ``l >= 1`` (the maximum adds 1), so ``nl`` is
    finite whenever the row has a finite element."""
    return T.exp2(T.call_pure_extern("__fmaf_rn", x - m, T.float32(L2E), nl, ty="float32"))


@T.inline
def lse_combine(st, ot):
    """(m, l) <- (m, l) ⊕ (m', l'): keep the larger maximum, rescale both sums to it."""
    mx = fmax(st[0], ot[0])
    st[1] = st[1] * _exp(st[0], mx) + ot[1] * _exp(ot[0], mx)
    st[0] = mx


@T.inline
def lse_combine_e(st, ot):
    """``lse_combine`` with natural exponentials (the unaligned paths)."""
    mx = T.max(st[0], ot[0])
    st[1] = st[1] * T.exp(st[0] - mx) + ot[1] * T.exp(ot[0] - mx)
    st[0] = mx


@T.inline
def max_combine(st, ot):
    st[0] = fmax(st[0], ot[0])


@T.inline
def add_combine(st, ot):
    st[0] = st[0] + ot[0]


@T.inline
def max_add_combine(st, ot):
    """(max, sum) <- (max, sum) ⊕ (max', sum'): sums of exponentials taken relative to the
    same (global) maximum simply add."""
    st[0] = fmax(st[0], ot[0])
    st[1] = st[1] + ot[1]


@T.inline
def absorb_vec(st, ot, buf, off, vec):
    """Fold ``buf[off : off + vec]`` into ``st``: the new maximum, then one exp per element."""
    ot[0] = st[0]
    for e in T.unroll(vec):
        ot[0] = T.max(ot[0], T.Cast("float32", buf[off + e]))
    st[1] = st[1] * T.exp(st[0] - ot[0])
    for e in T.unroll(vec):
        st[1] = st[1] + T.exp(T.Cast("float32", buf[off + e]) - ot[0])
    st[0] = ot[0]


@T.inline
def absorb2(st, ot, buf, cnt):
    """Fold ``buf[0:cnt]`` into ``st`` in base 2: the new maximum, the old sum rescaled to
    it once, then one exponential per element. ``ot`` is scratch."""
    ot[0] = st[0]
    for e in T.unroll(cnt):
        ot[0] = fmax(ot[0], T.Cast("float32", buf[e]))
    st[1] = st[1] * _exp(st[0], ot[0])
    for e in T.unroll(cnt):
        st[1] = st[1] + _exp(T.Cast("float32", buf[e]), ot[0])
    st[0] = ot[0]


@T.inline
def absorb_one(st, ot, x):
    """Fold one element ``x`` (float32) into ``st``."""
    ot[0] = T.max(st[0], x)
    st[1] = st[1] * T.exp(st[0] - ot[0]) + T.exp(x - ot[0])
    st[0] = ot[0]


@T.inline
def normalize_vec(obuf, buf, off, vec, fin):
    """``obuf[0:vec] = bf16(exp(buf[off:off+vec] - m) * (1 / l))``, ``fin = (m, 1 / l)``."""
    for e in T.unroll(vec):
        obuf[e] = T.Cast("bfloat16", T.exp(T.Cast("float32", buf[off + e]) - fin[0]) * fin[1])


@T.inline
def norm2(buf, off, fin):
    """``buf[off:off+8] = bf16(exp(x - m) / l)`` in place, ``fin = (m, -log2 l)``."""
    for e in T.unroll(VEC):
        buf[off + e] = T.Cast(
            "bfloat16", _exp_over(T.Cast("float32", buf[off + e]), fin[0], fin[1])
        )


@T.inline
def stream_state(st, ot, buf, A, row, lo, hi, tx, threads, vec):
    """Absorb ``A[row, lo*vec : hi*vec]`` (vector indices) into ``st``, ``U`` vectors in flight."""
    v: T.int32 = lo + tx
    while v + (U - 1) * threads < hi:
        for u in T.unroll(U):
            buf[T.Ramp(u * vec, 1, vec)] = A[row, T.Ramp((v + u * threads) * vec, 1, vec)]
        for u in T.unroll(U):
            absorb_vec(st, ot, buf, u * vec, vec)
        v += U * threads
    while v < hi:
        buf[T.Ramp(0, 1, vec)] = A[row, T.Ramp(v * vec, 1, vec)]
        absorb_vec(st, ot, buf, 0, vec)
        v += threads


@T.inline
def stream_out(Out, A, buf, obuf, row, lo, hi, tx, threads, vec, fin):
    """``Out[row, lo*vec : hi*vec] = softmax`` of ``A``'s, from ``fin = (m, 1 / l)``."""
    v: T.int32 = lo + tx
    while v + (U - 1) * threads < hi:
        for u in T.unroll(U):
            buf[T.Ramp(u * vec, 1, vec)] = A[row, T.Ramp((v + u * threads) * vec, 1, vec)]
        for u in T.unroll(U):
            normalize_vec(obuf, buf, u * vec, vec, fin)
            Out[row, T.Ramp((v + u * threads) * vec, 1, vec)] = obuf[T.Ramp(0, 1, vec)]
        v += U * threads
    while v < hi:
        buf[T.Ramp(0, 1, vec)] = A[row, T.Ramp(v * vec, 1, vec)]
        normalize_vec(obuf, buf, 0, vec, fin)
        Out[row, T.Ramp(v * vec, 1, vec)] = obuf[T.Ramp(0, 1, vec)]
        v += threads


@T.inline
def stream_state_scalar(st, ot, A, row, lo, hi, tx, threads):
    """Absorb ``A[row, lo:hi]`` into ``st``, one element at a time."""
    j: T.int32 = lo + tx
    while j < hi:
        absorb_one(st, ot, T.Cast("float32", A[row, j]))
        j += threads


@T.inline
def stream_out_scalar(Out, A, row, lo, hi, tx, threads, fin):
    j: T.int32 = lo + tx
    while j < hi:
        Out[row, j] = T.Cast("bfloat16", T.exp(T.Cast("float32", A[row, j]) - fin[0]) * fin[1])
        j += threads


@T.inline
def finish(st, fin):
    """``fin = (m, 1 / l)`` from the row's state."""
    fin[0] = st[0]
    fin[1] = T.float32(1.0) / st[1]


# ---------------------------------------------------------------------------
# resident: C CTAs per row, the row on chip
# ---------------------------------------------------------------------------
@T.inline
def fill_smem(S, A, row, lo, hi, tx, threads, kr, ks, nvec, full):
    """This thread's shared-memory slots of the row by ``cp.async``: vector
    ``lo + (kr + k) * threads + tx`` into ``S[k * threads + tx]`` (the source must be
    provably in bounds: the clamp, never taken under the guard, proves it)."""
    for k in T.unroll(ks):
        if full or lo + (kr + k) * threads + tx < hi:
            T.cuda.tile.apply(
                T.ptx["cp.async.cg.shared.global"],
                S[k * threads + tx, 0:VEC],
                A[
                    row,
                    T.min(lo + (kr + k) * threads + tx, nvec - 1) * VEC : T.min(
                        lo + (kr + k) * threads + tx, nvec - 1
                    )
                    * VEC
                    + VEC,
                ],
                chunk_bytes=16,
            )
    T.ptx.cp.async_.commit_group()


@T.inline
def fill_nothing(S, A, row, lo, hi, tx, threads, kr, ks, nvec, full):
    pass


@T.inline
def cluster_merge(st, ot, cs, addr, C):
    """Every CTA of the cluster publishes its state in ``cs``; after the cluster barrier
    every thread merges the ``C`` states, in rank order (so the CTAs agree)."""
    T.ptx.barrier.cluster.arrive.release.aligned()
    T.ptx.barrier.cluster.wait.acquire.aligned()
    st[0] = T.float32(NEG)
    st[1] = T.float32(0)
    for j in T.unroll(C):
        T.ptx.mapa.shared__cluster.u32(
            addr[0], T.cuda.cvta_generic_to_shared(cs.ptr_to([0])), T.uint32(j)
        )
        T.ptx.ld.shared__cluster.v2.f32(ot[0], ot[1], addr[0])
        lse_combine(st, ot)
    # no CTA may leave (its shared memory with it) before every peer has read it
    T.ptx.barrier.cluster.arrive.relaxed.aligned()


def _resident_kernel(n: int, m: int, p: Plan):
    threads, kr, ks, C = p.threads, p.kr, p.ks, p.cluster
    warps = threads // 32
    nvec = m // VEC
    per = _ceil(nvec, C)  # vectors per CTA
    full = per * C == nvec and per == (kr + ks) * threads  # every slot in the row
    grid = n * C
    s_rows = max(ks, 1) * threads
    keep = p.keep
    assert not keep or ks == 0
    e_len = max(kr, 1) * VEC
    if C > 1:
        launch = T.cuda.LaunchConfig(grid=(grid,), block=(threads,), cluster=(C,))
    else:
        launch = T.cuda.LaunchConfig(grid=(grid,), block=(threads,))
    attrs = T.cuda.KernelAttributes(min_blocks_per_sm=p.minb)
    fill = fill_smem if ks > 0 else fill_nothing

    @T.function
    def softmax_resident(
        A: T.Tensor((n, m), "bfloat16", align=16), Out: T.Tensor((n, m), "bfloat16", align=16)
    ):
        T.device_entry(launch=launch, kernel_attrs=attrs)
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        row = bx // C
        lo = (bx % C) * per
        hi = T.min(lo + per, nvec)
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        xs = T.alloc_local((max(kr, 1), VEC), "bfloat16")
        vs = T.alloc_local((VEC,), "bfloat16")
        ex = T.alloc_local((e_len,), "float32")
        mt = T.alloc_local((1,), "float32")
        addr = T.alloc_local((1,), "uint32")
        S = T.alloc_tensor(
            ty_args=[T.Tensor((s_rows, VEC), "bfloat16", scope="shared.dyn", layout="default")]
        )
        red = T.alloc_shared((warps, 2), "float32")
        cs = T.alloc_shared((2,), "float32")
        # the one read of the row: registers by plain loads, shared memory by cp.async
        for k in T.unroll(kr):
            if full or lo + k * threads + tx < hi:
                xs[k, T.Ramp(0, 1, VEC)] = A[row, T.Ramp((lo + k * threads + tx) * VEC, 1, VEC)]
        fill(S, A, row, lo, hi, tx, threads, kr, ks, nvec, full)
        # this thread's state: its maximum, then one exponential per element
        st[0] = T.float32(NEG)
        for k in T.unroll(kr):
            if full or lo + k * threads + tx < hi:
                for e in T.unroll(VEC):
                    st[0] = fmax(st[0], T.Cast("float32", xs[k, e]))
        if ks > 0:
            T.ptx.cp.async_.wait_group(0)
        for k in T.serial(ks):
            if full or lo + (kr + k) * threads + tx < hi:
                vs[T.Ramp(0, 1, VEC)] = S[k * threads + tx, T.Ramp(0, 1, VEC)]
                for e in T.unroll(VEC):
                    st[0] = fmax(st[0], T.Cast("float32", vs[e]))
        st[1] = T.float32(0)
        for k in T.unroll(kr):
            if full or lo + k * threads + tx < hi:
                for e in T.unroll(VEC):
                    if keep:
                        ex[k * VEC + e] = _exp(T.Cast("float32", xs[k, e]), st[0])
                        st[1] = st[1] + ex[k * VEC + e]
                    else:
                        st[1] = st[1] + _exp(T.Cast("float32", xs[k, e]), st[0])
        for k in T.serial(ks):
            if full or lo + (kr + k) * threads + tx < hi:
                vs[T.Ramp(0, 1, VEC)] = S[k * threads + tx, T.Ramp(0, 1, VEC)]
                for e in T.unroll(VEC):
                    st[1] = st[1] + _exp(T.Cast("float32", vs[e]), st[0])
        mt[0] = st[0]  # this thread's maximum, for the kept exponentials
        cta_allreduce(st, ot, 2, lse_combine, red, warp, lane, warps)
        if C > 1:
            if tx == 0:
                cs[0] = st[0]
                cs[1] = st[1]
            cluster_merge(st, ot, cs, addr, C)
        # the epilogue, from the chip: p = exp(x - m) / l
        if keep:
            # e^(x - m) = e^(x - m_t) e^(m_t - m), m_t this thread's maximum
            ot[0] = _exp(mt[0], st[0]) / st[1]
        else:
            ot[0] = st[0]
            ot[1] = -T.log2(st[1])
        for k in T.unroll(kr):
            if full or lo + k * threads + tx < hi:
                for e in T.unroll(VEC):
                    if keep:
                        xs[k, e] = T.Cast("bfloat16", ex[k * VEC + e] * ot[0])
                    else:
                        xs[k, e] = T.Cast(
                            "bfloat16", _exp_over(T.Cast("float32", xs[k, e]), ot[0], ot[1])
                        )
                Out[row, T.Ramp((lo + k * threads + tx) * VEC, 1, VEC)] = xs[k, T.Ramp(0, 1, VEC)]
        for k in T.serial(ks):
            if full or lo + (kr + k) * threads + tx < hi:
                vs[T.Ramp(0, 1, VEC)] = S[k * threads + tx, T.Ramp(0, 1, VEC)]
                for e in T.unroll(VEC):
                    vs[e] = T.Cast("bfloat16", _exp_over(T.Cast("float32", vs[e]), ot[0], ot[1]))
                Out[row, T.Ramp((lo + (kr + k) * threads + tx) * VEC, 1, VEC)] = vs[
                    T.Ramp(0, 1, VEC)
                ]
        if C > 1:
            T.ptx.barrier.cluster.wait.acquire.aligned()

    return softmax_resident


# ---------------------------------------------------------------------------
# rounds: the grid on one row at a time; one cooperative launch, a barrier per row
# ---------------------------------------------------------------------------
@T.inline
def slots_fill(S, A, row, klo, hi, tx, threads, kr, ks, n, units):
    """This thread's shared-memory slots of ``row``: vector ``klo + (kr + k)*threads + tx``
    into ``S[k*threads + tx]`` by ``cp.async`` (whose source must be provably in bounds: the
    clamps, never taken under the guards, prove it)."""
    for k in T.unroll(ks):
        if klo + (kr + k) * threads + tx < hi:
            T.cuda.tile.apply(
                T.ptx["cp.async.cg.shared.global"],
                S[k * threads + tx, 0:VEC],
                A[
                    T.max(T.min(row, n - 1), 0),
                    T.min(klo + (kr + k) * threads + tx, units - 1) * VEC : T.min(
                        klo + (kr + k) * threads + tx, units - 1
                    )
                    * VEC
                    + VEC,
                ],
                chunk_bytes=16,
            )


@T.inline
def slots_none(S, A, row, klo, hi, tx, threads, kr, ks, n, units):
    pass


@T.inline
def kept_load(xs, S, A, row, klo, hi, tx, threads, kr, ks, n, units, fill):
    """This thread's kept vectors of ``row``: vector ``klo + k*threads + tx`` into
    ``xs[k]`` for ``k < kr``, the next ``ks`` into its shared-memory slots (``fill``)."""
    for k in T.unroll(kr):
        if klo + k * threads + tx < hi:
            xs[k, T.Ramp(0, 1, VEC)] = A[row, T.Ramp((klo + k * threads + tx) * VEC, 1, VEC)]
    fill(S, A, row, klo, hi, tx, threads, kr, ks, n, units)
    T.ptx.cp.async_.commit_group()


@T.inline
def kept_state16(st, ot, xs, xh, vs, vh, S, Sh, klo, hi, tx, threads, kr, ks, mt):
    """Fold this thread's kept vectors into ``st`` -- their maximum ``m_t = mt[0]``, then
    their exponentials relative to it (``ot = (max, sum)`` is scratch) -- and leave each
    ``exp(x - m_t)``, in [0, 1], in its place as float16 (``xh`` / ``Sh`` view ``xs`` /
    ``S``): the epilogue then only scales them, one exponential per element, not two.
    (float16 keeps them to 2^-11, a quarter of the output's own bfloat16 rounding.)"""
    ot[0] = T.float32(NEG)
    for k in T.unroll(kr):
        if klo + k * threads + tx < hi:
            for e in T.unroll(VEC):
                ot[0] = fmax(ot[0], T.Cast("float32", xs[k, e]))
    if ks > 0:
        T.ptx.cp.async_.wait_group(0)
    for k in T.serial(ks):
        if klo + (kr + k) * threads + tx < hi:
            vs[T.Ramp(0, 1, VEC)] = S[k * threads + tx, T.Ramp(0, 1, VEC)]
            for e in T.unroll(VEC):
                ot[0] = fmax(ot[0], T.Cast("float32", vs[e]))
    mt[0] = ot[0]
    ot[1] = T.float32(0)
    for k in T.unroll(kr):
        if klo + k * threads + tx < hi:
            for e in T.unroll(VEC):
                mt[1] = _exp(T.Cast("float32", xs[k, e]), ot[0])
                ot[1] = ot[1] + mt[1]
                xh[k, e] = T.Cast("float16", mt[1])
    for k in T.serial(ks):
        if klo + (kr + k) * threads + tx < hi:
            vs[T.Ramp(0, 1, VEC)] = S[k * threads + tx, T.Ramp(0, 1, VEC)]
            for e in T.unroll(VEC):
                mt[1] = _exp(T.Cast("float32", vs[e]), ot[0])
                ot[1] = ot[1] + mt[1]
                vh[e] = T.Cast("float16", mt[1])
            Sh[k * threads + tx, T.Ramp(0, 1, VEC)] = vh[T.Ramp(0, 1, VEC)]
    lse_combine(st, ot)


@T.inline
def slots_out16(Out, A, vs, vh, S, Sh, r, klo, hi, tx, threads, kr, ks, n, units, scale):
    """``kept_out16`` of the shared-memory slots."""
    for k in T.unroll(ks):
        if klo + (kr + k) * threads + tx < hi:
            vh[T.Ramp(0, 1, VEC)] = Sh[k * threads + tx, T.Ramp(0, 1, VEC)]
            for e in T.unroll(VEC):
                vs[e] = T.Cast("bfloat16", T.Cast("float32", vh[e]) * scale)
            Out[r, T.Ramp((klo + (kr + k) * threads + tx) * VEC, 1, VEC)] = vs[T.Ramp(0, 1, VEC)]
            if r + 1 < n:
                T.cuda.tile.apply(
                    T.ptx["cp.async.cg.shared.global"],
                    S[k * threads + tx, 0:VEC],
                    A[
                        T.max(T.min(r + 1, n - 1), 0),
                        T.min(klo + (kr + k) * threads + tx, units - 1) * VEC : T.min(
                            klo + (kr + k) * threads + tx, units - 1
                        )
                        * VEC
                        + VEC,
                    ],
                    chunk_bytes=16,
                )


@T.inline
def slots_out16_none(Out, A, vs, vh, S, Sh, r, klo, hi, tx, threads, kr, ks, n, units, scale):
    pass


@T.inline
def kept_out16(
    Out, A, xs, xh, vs, vh, S, Sh, r, klo, hi, tx, threads, kr, ks, n, units, scale, sout
):
    """Store this thread's kept elements of row ``r`` as ``exp(x - m_t) * scale`` (kept as
    float16 by ``kept_state16``), ``scale = exp(m_t - m) / l``; each slot, once stored,
    takes its vector of row ``r + 1``."""
    for k in T.unroll(kr):
        if klo + k * threads + tx < hi:
            for e in T.unroll(VEC):
                xs[k, e] = T.Cast("bfloat16", T.Cast("float32", xh[k, e]) * scale)
            Out[r, T.Ramp((klo + k * threads + tx) * VEC, 1, VEC)] = xs[k, T.Ramp(0, 1, VEC)]
            if r + 1 < n:
                xs[k, T.Ramp(0, 1, VEC)] = A[r + 1, T.Ramp((klo + k * threads + tx) * VEC, 1, VEC)]
    sout(Out, A, vs, vh, S, Sh, r, klo, hi, tx, threads, kr, ks, n, units, scale)
    T.ptx.cp.async_.commit_group()


@T.inline
def slice_prefetch(A, row, lo, hi, tx, n):
    """This thread's 16 KB piece of the CTA's slice ``[lo, hi)`` of ``row`` into L2."""
    if row < n and lo + tx * PF_VECS < hi:
        T.ptx.cp.async_.bulk.prefetch.L2.global_(
            A.ptr_to([T.min(row, n - 1), (lo + tx * PF_VECS) * VEC]),
            T.Cast(
                "uint32", (T.min(lo + tx * PF_VECS + PF_VECS, hi) - lo - tx * PF_VECS) * (VEC * 2)
            ),
        )


@T.inline
def stream_state2(st, ot, buf, A, row, lo, hi, tx, threads, uu):
    """Absorb vectors ``[lo, hi)`` of ``row`` into ``st`` in base 2, ``uu`` vectors in flight."""
    v: T.int32 = lo + tx
    while v + (uu - 1) * threads < hi:
        for u in T.unroll(uu):
            buf[T.Ramp(u * VEC, 1, VEC)] = A[row, T.Ramp((v + u * threads) * VEC, 1, VEC)]
        absorb2(st, ot, buf, uu * VEC)
        v += uu * threads
    while v < hi:
        buf[T.Ramp(0, 1, VEC)] = A[row, T.Ramp(v * VEC, 1, VEC)]
        absorb2(st, ot, buf, VEC)
        v += threads


@T.inline
def stream_out_rev(Out, A, buf, row, lo, hi, tx, threads, fin, uu):
    """Normalise vectors ``[lo, hi)`` of ``row``, read again newest first (the last ones
    the first pass read may still be in L2): the ragged tail, then whole steps of ``uu``
    vectors a thread backwards."""
    nfull = (hi - lo) // (uu * threads)
    v: T.int32 = lo + nfull * (uu * threads) + tx
    while v < hi:
        buf[T.Ramp(0, 1, VEC)] = A[row, T.Ramp(v * VEC, 1, VEC)]
        norm2(buf, 0, fin)
        Out[row, T.Ramp(v * VEC, 1, VEC)] = buf[T.Ramp(0, 1, VEC)]
        v += threads
    b: T.int32 = lo + (nfull - 1) * (uu * threads) + tx
    while b >= lo:
        for u in T.unroll(uu):
            buf[T.Ramp(u * VEC, 1, VEC)] = A[row, T.Ramp((b + u * threads) * VEC, 1, VEC)]
        for u in T.unroll(uu):
            norm2(buf, u * VEC, fin)
            Out[row, T.Ramp((b + u * threads) * VEC, 1, VEC)] = buf[T.Ramp(u * VEC, 1, VEC)]
        b -= uu * threads


def _rounds_kernel(n: int, m: int, p: Plan):
    """CTA ``c`` owns vectors ``[c*chunk, (c+1)*chunk)`` of every row: the first ``sp``
    streamed (folded into the state as they pass, read again after the barrier), the rest
    kept on chip -- vector ``klo + k*threads + tx`` in ``xs[k]`` (``k < kr``) or in shared
    memory (``S[(k - kr)*threads + tx]``)."""
    units = m // VEC
    G, threads, kr, ks, sp, chunk, uu = p.ctas, p.threads, p.kr, p.ks, p.sp, p.chunk, p.u
    warps = threads // 32
    assert sp + (kr + ks) * threads >= chunk, p
    s_rows = max(ks, 1) * threads
    # the next row's slice is prefetched into L2 while the grid waits at the barrier; with a
    # streamed part, the re-read slice keeps L2 busy instead
    prefetch = p.prefetch and sp == 0 and n > 1
    assert _ceil(chunk, PF_VECS) <= threads, p  # one prefetch per thread at most
    attrs = T.cuda.KernelAttributes(min_blocks_per_sm=p.minb)
    launch = T.cuda.LaunchConfig(grid=(G,), block=(threads,), cooperative=True)
    fill = slots_fill if ks > 0 else slots_none
    sout = slots_out16 if ks > 0 else slots_out16_none
    # row r's barrier is sense-reversing: CTA 0 adds 2^31 - (G - 1) and the others 1, so the
    # G arrivals add 2^31 (mod 2^32) -- the last one flips the top bit the waiters watch and
    # leaves the low bits at 0 -- and the counter holds 0 or 2^31 between calls, for ever
    lead = (1 << 31) - (G - 1)
    lead = lead - (1 << 32) if lead >= 1 << 31 else lead  # as an int32

    @T.function
    def softmax_rounds(
        A: T.Tensor((n, m), "bfloat16", align=16),
        Out: T.Tensor((n, m), "bfloat16", align=16),
        P: T.Tensor((n, G, 2), "float32"),
        Cnt: T.Tensor((n,), "int32"),
    ):
        T.device_entry(launch=launch, kernel_attrs=attrs)
        c = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        lo = c * chunk
        hi = T.min(lo + chunk, units)
        klo = T.min(lo + sp, hi)  # streamed [lo, klo), kept [klo, hi)
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        fin = T.alloc_local((2,), "float32")
        mt = T.alloc_local((2,), "float32")
        xs = T.alloc_local((max(kr, 1), VEC), "bfloat16")
        xh = xs.view("float16")
        vs = T.alloc_local((VEC,), "bfloat16")
        vh = T.alloc_local((VEC,), "float16")
        buf = T.alloc_local((uu * VEC,), "bfloat16")
        old = T.alloc_local((1,), "int32")
        spin = T.alloc_local((1,), "int32")
        S = T.alloc_tensor(
            ty_args=[T.Tensor((s_rows, VEC), "bfloat16", scope="shared.dyn", layout="default")]
        )
        Sh = S.view("float16")
        red = T.alloc_shared((warps, 2), "float32")
        kept_load(xs, S, A, 0, klo, hi, tx, threads, kr, ks, n, units, fill)
        r: T.int32 = 0
        while r < n:
            # the slice's state: the streamed part as it passes, then the kept part
            st[0] = T.float32(NEG)
            st[1] = T.float32(0)
            if sp > 0:
                stream_state2(st, ot, buf, A, r, lo, klo, tx, threads, uu)
            kept_state16(st, ot, xs, xh, vs, vh, S, Sh, klo, hi, tx, threads, kr, ks, mt)
            cta_allreduce(st, ot, 2, lse_combine, red, warp, lane, warps)
            # this CTA is done reading row r: its slice of row r + 1 on its way to L2
            if prefetch:
                slice_prefetch(A, r + 1, lo, hi, tx, n)
            # row r's barrier: publish this slice's state, wait for the other G - 1 (until
            # Cnt[r]'s top bit differs from the one this CTA's own arrival saw)
            if tx == 0:
                P[r, c, 0] = st[0]
                P[r, c, 1] = st[1]
                T.ptx.atom.release.gpu.global_.add.s32(
                    old[0], Cnt.ptr_to([r]), T.if_then_else(c == 0, T.int32(lead), T.int32(1))
                )
                spin[0] = old[0]
                T.cuda.wait_until(
                    spin[0], Cnt.ptr_to([r]), predicate=lambda v: (v ^ old[0]) < 0, scope="gpu"
                )
            T.cuda.cta_sync()
            # the row's state: the G published states merged (every CTA in the same order)
            st[0] = T.float32(NEG)
            st[1] = T.float32(0)
            j: T.int32 = tx
            while j < G:
                ot[0] = P[r, j, 0]
                ot[1] = P[r, j, 1]
                lse_combine(st, ot)
                j += threads
            cta_allreduce(st, ot, 2, lse_combine, red, warp, lane, warps)
            fin[0] = st[0]
            fin[1] = -T.log2(st[1])
            if sp > 0:
                stream_out_rev(Out, A, buf, r, lo, klo, tx, threads, fin, uu)
            # e^(x - m) / l = e^(x - m_t) e^(m_t - m) / l, m_t this thread's maximum
            mt[1] = _exp_over(mt[0], st[0], fin[1])
            kept_out16(
                Out,
                A,
                xs,
                xh,
                vs,
                vh,
                S,
                Sh,
                r,
                klo,
                hi,
                tx,
                threads,
                kr,
                ks,
                n,
                units,
                mt[1],
                sout,
            )
            r += 1

    return softmax_rounds


# ---------------------------------------------------------------------------
# unaligned rows: one element at a time
# ---------------------------------------------------------------------------
def _cache_kernel(n: int, m: int, p: Plan):
    threads, warps, r = p.threads, p.threads // 32, p.per_thread
    blocks = min(n, SMS * max(1, THREADS_PER_SM // threads))
    full = m % threads == 0  # every thread's last slot is in the row

    @T.function
    def softmax_cache(
        A: T.Tensor((n, m), "bfloat16", align=16), Out: T.Tensor((n, m), "bfloat16", align=16)
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        fin = T.alloc_local((2,), "float32")
        cache = T.alloc_local((r,), "bfloat16")  # this thread's share of the row
        smem = T.alloc_shared((warps, 2), "float32")
        row: T.int32 = bx
        while row < n:
            for k in T.unroll(r):
                if full or tx + k * threads < m:
                    cache[k] = A[row, tx + k * threads]
            st[0] = T.float32(NEG)
            st[1] = T.float32(0)
            for k in T.unroll(r):
                if full or tx + k * threads < m:
                    st[0] = T.max(st[0], T.Cast("float32", cache[k]))
            for k in T.unroll(r):
                if full or tx + k * threads < m:
                    st[1] = st[1] + T.exp(T.Cast("float32", cache[k]) - st[0])
            cta_allreduce(st, ot, 2, lse_combine_e, smem, warp, lane, warps)
            finish(st, fin)
            for k in T.unroll(r):
                if full or tx + k * threads < m:
                    Out[row, tx + k * threads] = T.Cast(
                        "bfloat16", T.exp(T.Cast("float32", cache[k]) - fin[0]) * fin[1]
                    )
            row += blocks

    return softmax_cache


def _stream_kernel(n: int, m: int, p: Plan):
    threads, warps = p.threads, p.threads // 32
    vec = VEC
    blocks = min(n, SMS * max(1, THREADS_PER_SM // threads))
    nvec = m // vec

    @T.function
    def softmax_stream(
        A: T.Tensor((n, m), "bfloat16", align=16), Out: T.Tensor((n, m), "bfloat16", align=16)
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        fin = T.alloc_local((2,), "float32")
        buf = T.alloc_local((U * vec,), "bfloat16")
        obuf = T.alloc_local((vec,), "bfloat16")
        smem = T.alloc_shared((warps, 2), "float32")
        row: T.int32 = bx
        while row < n:
            st[0] = T.float32(NEG)
            st[1] = T.float32(0)
            if p.vectorised:
                stream_state(st, ot, buf, A, row, 0, nvec, tx, threads, vec)
            else:
                stream_state_scalar(st, ot, A, row, 0, m, tx, threads)
            cta_allreduce(st, ot, 2, lse_combine_e, smem, warp, lane, warps)
            finish(st, fin)
            if p.vectorised:
                stream_out(Out, A, buf, obuf, row, 0, nvec, tx, threads, vec, fin)
            else:
                stream_out_scalar(Out, A, row, 0, m, tx, threads, fin)
            row += blocks

    return softmax_stream


def _split_kernels(n: int, m: int, p: Plan):
    threads, warps = p.threads, p.threads // 32
    vec = VEC
    chunks = p.chunks
    if p.vectorised:
        step, units = p.chunk // vec, m // vec  # in vectors
    else:
        step, units = p.chunk, m  # in elements
    grid = n * chunks

    @T.function
    def softmax_partial(
        A: T.Tensor((n, m), "bfloat16", align=16), W: T.Tensor((n, chunks, 2), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(grid,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        row = bx // chunks
        c = bx % chunks
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        buf = T.alloc_local((U * vec,), "bfloat16")
        smem = T.alloc_shared((warps, 2), "float32")
        st[0] = T.float32(NEG)
        st[1] = T.float32(0)
        if p.vectorised:
            stream_state(
                st, ot, buf, A, row, c * step, T.min(c * step + step, units), tx, threads, vec
            )
        else:
            stream_state_scalar(
                st, ot, A, row, c * step, T.min(c * step + step, units), tx, threads
            )
        cta_allreduce(st, ot, 2, lse_combine_e, smem, warp, lane, warps)
        if tx == 0:
            W[row, c, 0] = st[0]
            W[row, c, 1] = st[1]

    @T.function
    def softmax_normalize(
        A: T.Tensor((n, m), "bfloat16", align=16),
        W: T.Tensor((n, chunks, 2), "float32"),
        Out: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(grid,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        row = bx // chunks
        c = bx % chunks
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        fin = T.alloc_local((2,), "float32")
        buf = T.alloc_local((U * vec,), "bfloat16")
        obuf = T.alloc_local((vec,), "bfloat16")
        smem = T.alloc_shared((warps, 2), "float32")
        # merge the row's partial states
        st[0] = T.float32(NEG)
        st[1] = T.float32(0)
        k: T.int32 = tx
        while k < chunks:
            ot[0] = W[row, k, 0]
            ot[1] = W[row, k, 1]
            lse_combine_e(st, ot)
            k += threads
        cta_allreduce(st, ot, 2, lse_combine_e, smem, warp, lane, warps)
        finish(st, fin)
        if p.vectorised:
            stream_out(
                Out,
                A,
                buf,
                obuf,
                row,
                c * step,
                T.min(c * step + step, units),
                tx,
                threads,
                vec,
                fin,
            )
        else:
            stream_out_scalar(
                Out, A, row, c * step, T.min(c * step + step, units), tx, threads, fin
            )

    return softmax_partial, softmax_normalize


# ---------------------------------------------------------------------------
# unfused: the ablation, one launch per link of the unfused chain
# ---------------------------------------------------------------------------
@T.inline
def sweep_max(acc, buf, A, row, lo, hi, tx, threads, uu):
    """``acc[0] = max(acc[0], A[row, 8 lo : 8 hi])``, ``uu`` vectors in flight."""
    v: T.int32 = lo + tx
    while v + (uu - 1) * threads < hi:
        for u in T.unroll(uu):
            buf[T.Ramp(u * VEC, 1, VEC)] = A[row, T.Ramp((v + u * threads) * VEC, 1, VEC)]
        for e in T.unroll(uu * VEC):
            acc[0] = fmax(acc[0], T.Cast("float32", buf[e]))
        v += uu * threads
    while v < hi:
        buf[T.Ramp(0, 1, VEC)] = A[row, T.Ramp(v * VEC, 1, VEC)]
        for e in T.unroll(VEC):
            acc[0] = fmax(acc[0], T.Cast("float32", buf[e]))
        v += threads


@T.inline
def sweep_sum(acc, buf, A, row, lo, hi, tx, threads, mx, uu):
    """``acc[0] += Σ exp(x - mx)`` over ``A[row, 8 lo : 8 hi]``, ``mx`` the row's maximum."""
    v: T.int32 = lo + tx
    while v + (uu - 1) * threads < hi:
        for u in T.unroll(uu):
            buf[T.Ramp(u * VEC, 1, VEC)] = A[row, T.Ramp((v + u * threads) * VEC, 1, VEC)]
        for e in T.unroll(uu * VEC):
            acc[0] = acc[0] + _exp(T.Cast("float32", buf[e]), mx)
        v += uu * threads
    while v < hi:
        buf[T.Ramp(0, 1, VEC)] = A[row, T.Ramp(v * VEC, 1, VEC)]
        for e in T.unroll(VEC):
            acc[0] = acc[0] + _exp(T.Cast("float32", buf[e]), mx)
        v += threads


@T.inline
def sweep_out(Out, A, buf, row, lo, hi, tx, threads, fin, uu):
    """``Out[row, 8 lo : 8 hi] = bf16(exp(x - m) / l)``, ``fin = (m, -log2 l)``."""
    v: T.int32 = lo + tx
    while v + (uu - 1) * threads < hi:
        for u in T.unroll(uu):
            buf[T.Ramp(u * VEC, 1, VEC)] = A[row, T.Ramp((v + u * threads) * VEC, 1, VEC)]
        for u in T.unroll(uu):
            norm2(buf, u * VEC, fin)
            Out[row, T.Ramp((v + u * threads) * VEC, 1, VEC)] = buf[T.Ramp(u * VEC, 1, VEC)]
        v += uu * threads
    while v < hi:
        buf[T.Ramp(0, 1, VEC)] = A[row, T.Ramp(v * VEC, 1, VEC)]
        norm2(buf, 0, fin)
        Out[row, T.Ramp(v * VEC, 1, VEC)] = buf[T.Ramp(0, 1, VEC)]
        v += threads


@T.inline
def sweep_max_scalar(acc, buf, A, row, lo, hi, tx, threads, uu):
    j: T.int32 = lo + tx
    while j < hi:
        acc[0] = fmax(acc[0], T.Cast("float32", A[row, j]))
        j += threads


@T.inline
def sweep_sum_scalar(acc, buf, A, row, lo, hi, tx, threads, mx, uu):
    j: T.int32 = lo + tx
    while j < hi:
        acc[0] = acc[0] + _exp(T.Cast("float32", A[row, j]), mx)
        j += threads


@T.inline
def sweep_out_scalar(Out, A, buf, row, lo, hi, tx, threads, fin, uu):
    j: T.int32 = lo + tx
    while j < hi:
        Out[row, j] = T.Cast("bfloat16", _exp_over(T.Cast("float32", A[row, j]), fin[0], fin[1]))
        j += threads


def _passes_kernels(n: int, m: int, p: Plan):
    """max, then Σ exp(x - max), then the output: three launches, each reading X again.
    Only the per-chunk maxima ``M1`` and sums ``S1``, ``(n, chunks)``, travel between them."""
    threads, warps, chunks, step, uu = p.threads, p.threads // 32, p.chunks, p.chunk, p.u
    units = m // VEC if p.vectorised else m
    grid = n * chunks
    if p.vectorised:
        smax, ssum, sout = sweep_max, sweep_sum, sweep_out
    else:
        smax, ssum, sout = sweep_max_scalar, sweep_sum_scalar, sweep_out_scalar

    @T.function
    def softmax_max_pass(
        A: T.Tensor((n, m), "bfloat16", align=16), M1: T.Tensor((n, chunks), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(grid,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        row = bx // chunks
        c = bx % chunks
        acc = T.alloc_local((1,), "float32")
        ot = T.alloc_local((1,), "float32")
        buf = T.alloc_local((uu * VEC,), "bfloat16")
        red = T.alloc_shared((warps, 1), "float32")
        acc[0] = T.float32(NEG)
        smax(acc, buf, A, row, c * step, T.min(c * step + step, units), tx, threads, uu)
        cta_allreduce(acc, ot, 1, max_combine, red, warp, lane, warps)
        if tx == 0:
            M1[row, c] = acc[0]

    @T.function
    def softmax_sum_pass(
        A: T.Tensor((n, m), "bfloat16", align=16),
        M1: T.Tensor((n, chunks), "float32"),
        S1: T.Tensor((n, chunks), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(grid,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        row = bx // chunks
        c = bx % chunks
        mx = T.alloc_local((1,), "float32")
        acc = T.alloc_local((1,), "float32")
        ot = T.alloc_local((1,), "float32")
        buf = T.alloc_local((uu * VEC,), "bfloat16")
        red = T.alloc_shared((warps, 1), "float32")
        if chunks == 1:
            mx[0] = M1[row, 0]
        else:
            mx[0] = T.float32(NEG)
            j: T.int32 = tx
            while j < chunks:
                mx[0] = fmax(mx[0], M1[row, j])
                j += threads
            cta_allreduce(mx, ot, 1, max_combine, red, warp, lane, warps)
        acc[0] = T.float32(0)
        ssum(
            acc,
            buf,
            A,
            row,
            c * step,
            T.min(c * step + step, units),
            tx,
            threads,
            mx[0],
            uu,
        )
        cta_allreduce(acc, ot, 1, add_combine, red, warp, lane, warps)
        if tx == 0:
            S1[row, c] = acc[0]

    @T.function
    def softmax_out_pass(
        A: T.Tensor((n, m), "bfloat16", align=16),
        M1: T.Tensor((n, chunks), "float32"),
        S1: T.Tensor((n, chunks), "float32"),
        Out: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(grid,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        row = bx // chunks
        c = bx % chunks
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        fin = T.alloc_local((2,), "float32")
        buf = T.alloc_local((uu * VEC,), "bfloat16")
        red = T.alloc_shared((warps, 2), "float32")
        if chunks == 1:
            st[0] = M1[row, 0]
            st[1] = S1[row, 0]
        else:
            st[0] = T.float32(NEG)
            st[1] = T.float32(0)
            j: T.int32 = tx
            while j < chunks:
                st[0] = fmax(st[0], M1[row, j])
                st[1] = st[1] + S1[row, j]
                j += threads
            cta_allreduce(st, ot, 2, max_add_combine, red, warp, lane, warps)
        fin[0] = st[0]
        fin[1] = -T.log2(st[1])
        sout(Out, A, buf, row, c * step, T.min(c * step + step, units), tx, threads, fin, uu)

    return softmax_max_pass, softmax_sum_pass, softmax_out_pass


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _per_stream(make):
    """``get(device)``: the workspace ``make(device)`` of torch's current stream, which the
    kernels run on. Calls on one stream run in order, so they share it; a call on another
    stream gets its own, and a call made while a CUDA graph is captured a new one (the
    graph keeps it). ``get.spaces`` maps ``(device, stream handle)`` to the workspaces."""
    spaces = {}

    def get(device):
        if torch.cuda.is_current_stream_capturing():
            return make(device)
        s = torch.cuda.current_stream(device)
        key = (s.device_index, s.cuda_stream)
        ws = spaces.get(key)
        if ws is None:
            ws = spaces[key] = make(device)
        return ws

    get.spaces = spaces
    return get


def build_plan(n: int, m: int, p: Plan):
    """Compile the kernel(s) of plan ``p`` for ``(n, m)`` and return ``run``."""
    assert n * m < 2**31, "indices are int32"
    tag = f"softmax_{p.path}_{n}x{m}"
    if p.path == "split":
        f_partial, f_normalize = _split_kernels(n, m, p)
        partial = compile_sm90(f_partial, name=tag + "_partial")
        normalize = compile_sm90(f_normalize, name=tag + "_normalize")
        workspace = _per_stream(lambda dev: torch.empty((n, p.chunks, 2), device=dev, dtype=F32))

        def run(X):
            out = torch.empty_like(X)
            ws = workspace(X.device)
            partial(X, ws)
            normalize(X, ws, out)
            return (out,)

        run.mods = [partial, normalize]
        run.workspaces = workspace.spaces
        return run

    if p.path == "passes":
        k_max, k_sum, k_out = _passes_kernels(n, m, p)
        mod_1 = compile_sm90(k_max, name=tag + "_max")
        mod_2 = compile_sm90(k_sum, name=tag + "_sum")
        mod_3 = compile_sm90(k_out, name=tag + "_out")
        workspace = _per_stream(  # the chunks' maxima and sums
            lambda dev: tuple(torch.empty((n, p.chunks), device=dev, dtype=F32) for _ in "ms")
        )

        def run(X):
            out = torch.empty_like(X)
            m1, s1 = workspace(X.device)
            mod_1(X, m1)
            mod_2(X, m1, s1)
            mod_3(X, m1, s1, out)
            return (out,)

        run.mods = [mod_1, mod_2, mod_3]
        run.workspaces = workspace.spaces
        return run

    if p.path == "rounds":
        mod = compile_sm90(_rounds_kernel(n, m, p), name=tag)
        # the row barriers need the whole grid resident at once (the cooperative launch
        # refuses otherwise): if minb CTAs do not fit an SM, or the device has fewer SMs,
        # take the two-launch split
        smem = max(p.ks, 1) * p.threads * VEC * 2
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        if sms < SMS or min(ctas_per_sm(mod, p.threads, smem).values()) < p.minb:
            return build_plan(n, m, plan(n, m, "split"))
        # the slices' states of every row, and the rows' barrier counters (0 or 2^31 between
        # calls: they start at 0)
        workspace = _per_stream(
            lambda dev: (
                torch.empty((n, p.ctas, 2), device=dev, dtype=F32),
                torch.zeros((n,), device=dev, dtype=torch.int32),
            )
        )

        def run(X):
            out = torch.empty_like(X)
            part, count = workspace(X.device)
            mod(X, out, part, count)
            return (out,)

        run.mods = [mod]
        run.workspaces = workspace.spaces
        return run

    if p.path == "resident":
        kernel = _resident_kernel(n, m, p)
    elif p.path == "cache":
        kernel = _cache_kernel(n, m, p)
    else:
        kernel = _stream_kernel(n, m, p)
    mod = compile_sm90(kernel, name=tag)

    def run(X):
        out = torch.empty_like(X)
        mod(X, out)
        return (out,)

    run.mod = mod
    run.mods = [mod]
    return run


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, m = cfg["n"], cfg["m"]
    if variant == "unfused":
        return build_plan(n, m, _passes_plan(n, m))
    return build_plan(n, m, plan(n, m))
