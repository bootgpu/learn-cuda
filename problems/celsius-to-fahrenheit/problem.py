import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    atol = 1e-5
    rtol = 1e-5

    def reference(self, celsius, fahrenheit, n):
        torch.add(celsius * 9 / 5, 32, out=fahrenheit)

    def signature(self):
        return {
            "celsius": (ctypes.c_void_p, "in"),
            "fahrenheit": (ctypes.c_void_p, "out"),
            "n": (ctypes.c_int, "in"),
        }

    def _case(self, celsius):
        celsius = torch.as_tensor(celsius, dtype=torch.float32).to(self.device)
        return {"celsius": celsius, "fahrenheit": torch.empty_like(celsius), "n": celsius.numel()}

    def _random(self, n):
        return self._case(torch.empty(n).uniform_(-100, 100))

    def examples(self):
        return [
            self._case([0.0, 100.0, -40.0, 37.0]),
            self._case([20.0, 25.5, -10.0]),
        ]

    def tests(self):
        return [
            self._case([7.0]),
            # One block with idle threads past n: without the i < n check they write past the end.
            self._random(200),
            self._random(256),
            # A second block: an index without blockIdx.x * blockDim.x never reaches elements 256 and up.
            self._random(257),
            self._random(1000),
            self._random(10_000),
        ]
