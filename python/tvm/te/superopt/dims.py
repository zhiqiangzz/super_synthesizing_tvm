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
"""Identity of symbolic extents.

Two tensor axes are *compatible* (can be contracted, broadcast against each
other, or share a reduction) iff their extents are the same ``te.var`` object
or the same integer constant. :class:`DimTable` assigns every distinct extent
a small integer ``DimKey`` so the rest of the search can compare axes by
``==``.
"""

from __future__ import annotations

import itertools

from tvm import tirx as tir

DimKey = int

# Keys are unique across tables: the symbolic IR is interned process-wide and
# names reduction axes by key, so two tables must never share one.
_NEXT_KEY = itertools.count()


class DimTable:
    """Interns extent expressions (``te.var`` or ``IntImm``) to integer keys."""

    def __init__(self) -> None:
        self._extents: dict[DimKey, object] = {}
        self._names: dict[DimKey, str] = {}
        self._order: list[DimKey] = []

    def key(self, extent) -> DimKey:
        from .symbolic import ir  # local: dims is imported by the symbolic package

        extent = tir.convert(extent)
        for k in self._order:
            if _same_extent(self._extents[k], extent):
                return k
        if isinstance(extent, tir.IntImm):
            name = str(int(extent.value))
            value = ir.const(int(extent.value))
        elif isinstance(extent, tir.Var):
            name = extent.name
            if name in self._names.values():
                raise ValueError(f"two distinct symbolic extents share the name {name!r}")
            value = ir.shape_sym(name)
        else:
            raise ValueError(f"unsupported extent expression {extent!r}")
        k = next(_NEXT_KEY)
        self._extents[k] = extent
        self._names[k] = name
        self._order.append(k)
        ir.register_extent(k, value)
        return k

    def keys(self, shape) -> tuple[DimKey, ...]:
        return tuple(self.key(s) for s in shape)

    def all_keys(self) -> list[DimKey]:
        return list(self._order)

    def extent(self, key: DimKey):
        return self._extents[key]

    def name(self, key: DimKey) -> str:
        return self._names[key]

    def key_of_name(self, name: str) -> DimKey:
        for k, n in self._names.items():
            if n == name:
                return k
        raise KeyError(name)

    def is_symbolic(self, key: DimKey) -> bool:
        return isinstance(self._extents[key], tir.Var)

    def __len__(self) -> int:
        return len(self._order)


def _same_extent(a, b) -> bool:
    if isinstance(a, tir.Var) or isinstance(b, tir.Var):
        return a.same_as(b)
    if isinstance(a, tir.IntImm) and isinstance(b, tir.IntImm):
        return int(a.value) == int(b.value)
    return False
