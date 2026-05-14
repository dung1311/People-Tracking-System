"""Union-Find (Disjoint Set) with path compression and union by rank."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def groups(self) -> List[List[int]]:
        """Return all connected components as lists of indices."""
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            clusters[self.find(i)].append(i)
        return list(clusters.values())
