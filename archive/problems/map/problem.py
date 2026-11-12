import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    atol = 1e-5
    rtol = 1e-5

    def reference(self, x, y, N):
        torch.add(x * 2, 1, out=y)

    def signature(self):
        return {
            "x": (ctypes.c_void_p, "in"),
            "y": (ctypes.c_void_p, "out"),
            "N": (ctypes.c_int, "in"),
        }

    def _case(self, x):
        x = x.to(dtype=torch.float32, device=self.device)
        return {"x": x, "y": torch.empty_like(x), "N": x.numel()}

    def _random(self, n):
        return self._case(torch.empty(n).uniform_(-100, 100))

    def examples(self):
        return [
            self._case(torch.tensor([1.0, 2.0, 3.0, 4.0])),
            self._case(torch.tensor([-0.5, 0.0, 10.0])),
        ]

    def tests(self):
        return [
            self._case(torch.tensor([7.0])),
            self._random(32),
            # Not a multiple of 32: the last block is partly idle, which needs the i < N check.
            self._random(33),
            # More than one block of up to 256 threads: an index without blockIdx.x misses the rest.
            self._random(500),
            self._random(1024),
        ]
