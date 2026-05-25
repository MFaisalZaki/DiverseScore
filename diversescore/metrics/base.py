from itertools import combinations

import numpy as np


class Metric:
    def __init__(self, name, task, plans) -> None:
        self.name = name
        self.task = task
        self.plans = list(plans) if plans is not None else []
        # Keep strong references so id(plan) keys stay valid for the lifetime of the metric.
        self._plans_ref = self.plans

    def __call__(self, x, y):
        raise NotImplementedError("Metric must be implemented.")

    def pairwise(self, plans=None):
        """Return an (n, n) np.ndarray of pairwise distances.

        Subclasses should override with a vectorized implementation. The default
        falls back to a Python-level double loop over `__call__`.
        """
        plans = self.plans if plans is None else list(plans)
        n = len(plans)
        D = np.zeros((n, n), dtype=np.float64)
        for i, j in combinations(range(n), 2):
            d = self(plans[i], plans[j])
            D[i, j] = d
            D[j, i] = d
        return D

    def __str__(self) -> str:
        return self.name