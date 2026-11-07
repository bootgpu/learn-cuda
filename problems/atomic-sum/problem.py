import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    # Blocks add in any order, so the last digits change from run to run.
    # Sized for one atomicAdd per element, the least precise correct way (a variant in course-solutions).
    atol = 1e-3
    rtol = 1e-4

    def reference(self, x, sum, N):
        sum[0] = x.sum()

    def signature(self):
        return {
            "x": (ctypes.c_void_p, "in"),
            "sum": (ctypes.c_void_p, "out"),
            "N": (ctypes.c_int, "in"),
        }

    def _case(self, x):
        x = x.to(dtype=torch.float32, device=self.device)
        # The grader fills sum with garbage first, so a solution that never zeroes it fails every case.
        return {"x": x, "sum": torch.empty(1, device=self.device), "N": x.numel()}

    def examples(self):
        return [
            self._case(torch.tensor([1.0, 2.0, 3.0, 4.0])),
            self._case(torch.tensor([0.5, -1.5, 2.0])),
        ]

    def tests(self):
        return [
            self._case(torch.tensor([3.0])),
            self._case(torch.ones(256)),
            # A second block: storing a block's sum instead of adding it loses the first block.
            self._case(torch.ones(257)),
            # Exact in float32, so any add lost to a race (sum[0] += without atomicAdd) shows.
            self._case(torch.ones(10_000)),
            self._case(torch.rand(1000)),
        ]
