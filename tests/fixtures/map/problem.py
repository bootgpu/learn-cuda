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

    def _case(self, vals):
        x = torch.tensor(vals, dtype=torch.float32, device=self.device)
        return {"x": x, "y": torch.empty_like(x), "N": x.numel()}

    def _rand(self, n):
        x = torch.empty(n, device=self.device).uniform_(-100, 100)
        return {"x": x, "y": torch.empty_like(x), "N": n}

    def examples(self):
        return [self._case([1.0, 2.0, 3.0, 4.0])]

    def tests(self):
        return [
            self._case([5.0]),
            self._case([-1.5, 0.0, 2.5]),
            self._rand(256),
            self._rand(257),
            self._rand(1000),
            self._rand(10_000),
        ]
