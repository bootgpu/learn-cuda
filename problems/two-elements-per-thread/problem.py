import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    atol = 1e-5
    rtol = 1e-5

    def reference(self, a, b, c, n):
        torch.add(a, b, out=c)

    def signature(self):
        return {
            "a": (ctypes.c_void_p, "in"),
            "b": (ctypes.c_void_p, "in"),
            "c": (ctypes.c_void_p, "out"),
            "n": (ctypes.c_int, "in"),
        }

    def _case(self, a, b):
        a = torch.as_tensor(a, dtype=torch.float32).to(self.device)
        b = torch.as_tensor(b, dtype=torch.float32).to(self.device)
        return {"a": a, "b": b, "c": torch.empty_like(a), "n": a.numel()}

    def _random(self, n):
        return self._case(torch.empty(n).uniform_(-100, 100), torch.empty(n).uniform_(-100, 100))

    # solve launches blocks * 256 threads for n elements, about n / 2: a kernel that computes one element
    # per thread leaves the rest of c unwritten once n > 256 (from test_3 on).
    def examples(self):
        return [
            self._case([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]),
            self._case([0.5, -1.5, 2.0, 3.0], [0.5, 1.5, -4.0, 7.0]),
        ]

    def tests(self):
        return [
            # Odd n: the last thread's second element is past the end, so it needs its own check against n.
            self._case([7.0], [-2.0]),
            self._random(2),
            self._random(511),
            self._random(512),
            # A second block: its first element is 512, not 256.
            self._random(513),
            self._random(1001),
            self._random(9999),
        ]
