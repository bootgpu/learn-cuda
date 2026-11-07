import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    # Most outputs are tiny, so the check is relative; atol only covers outputs near 0.
    # Sized for a denominator of one atomicAdd per element, the least precise correct way
    # (a variant in course-solutions).
    atol = 1e-9
    rtol = 1e-4

    def reference(self, x, y, N):
        torch.softmax(x, dim=0, out=y)

    def signature(self):
        return {
            "x": (ctypes.c_void_p, "in"),
            "y": (ctypes.c_void_p, "out"),
            "N": (ctypes.c_int, "in"),
        }

    def _case(self, x):
        x = x.to(dtype=torch.float32, device=self.device)
        return {"x": x, "y": torch.empty_like(x), "N": x.numel()}

    def _random(self, n, low, high):
        return self._case(torch.empty(n).uniform_(low, high))

    def examples(self):
        return [
            self._case(torch.tensor([1.0, 2.0, 3.0])),
            # exp(1000) overflows to inf: skipping the maximum gives inf / inf = NaN.
            self._case(torch.tensor([1000.0, 1000.0])),
        ]

    def tests(self):
        return [
            self._case(torch.tensor([5.0])),
            # Past one block of 256: a max or sum over x[0..255] only is wrong here.
            self._case(torch.zeros(257)),
            self._random(1000, -10, 10),
            # exp overflows from about 88 on, so the maximum must come off first.
            self._random(1000, 80, 100),
            self._random(10_000, -100, 0),
        ]
