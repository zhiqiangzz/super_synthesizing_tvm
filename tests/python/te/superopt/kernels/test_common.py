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
"""The shared kernel helpers: subnormals with ftz=False, workspaces per stream and graph."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the kernels run on CUDA", allow_module_level=True)

from kernels.common.build import compile_sm90  # noqa: E402
from kernels.common.workspace import workspace  # noqa: E402

from tvm_tirx.script import tirx as T  # noqa: E402

N = 128


@T.function
def _square(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(N,)))
    tx = T.cuda.thread_idx("x")
    B[tx] = A[tx] * A[tx]


@pytest.mark.parametrize("ftz, expected", [(True, 0.0), (False, 1e-40)])
def test_subnormals_survive_only_without_ftz(ftz, expected):
    a = torch.full((N,), 1e-20, device="cuda")
    b = torch.empty_like(a)
    compile_sm90(_square, ftz=ftz)(a, b)
    torch.cuda.synchronize()
    assert b[0].item() == pytest.approx(expected, rel=1e-4, abs=0.0)


def test_one_workspace_per_stream():
    made = []

    def make():
        made.append(torch.zeros(4, device="cuda"))
        return made[-1]

    first = workspace("test_one_workspace_per_stream", make)
    assert workspace("test_one_workspace_per_stream", make) is first  # cached on this stream
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        other = workspace("test_one_workspace_per_stream", make)
    assert other is not first and len(made) == 2


def test_a_fresh_workspace_under_graph_capture():
    made = []

    def make():
        made.append(torch.zeros(4, device="cuda"))
        return made[-1]

    cached = workspace("test_a_fresh_workspace_under_graph_capture", make)
    graph = torch.cuda.CUDAGraph()
    capture = torch.cuda.Stream()
    with torch.cuda.stream(capture):
        with torch.cuda.graph(graph, stream=capture):
            captured = workspace("test_a_fresh_workspace_under_graph_capture", make)
            captured.add_(1.0)
    assert captured is not cached and len(made) == 2
    graph.replay()
    torch.cuda.synchronize()
    assert captured.sum().item() > 0 and cached.sum().item() == 0
