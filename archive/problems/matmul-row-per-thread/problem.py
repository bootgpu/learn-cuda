import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    # Each element sums up to 100 products of size up to 100.
    atol = 1e-3
    rtol = 1e-4

    def reference(self, M, N, P, n):
        torch.mm(M, N, out=P)

    def signature(self):
        return {
            "M": (ctypes.c_void_p, "in"),
            "N": (ctypes.c_void_p, "in"),
            "P": (ctypes.c_void_p, "out"),
            "n": (ctypes.c_int, "in"),
        }

    def _case(self, M, N):
        M = torch.as_tensor(M, dtype=torch.float32).to(self.device)
        N = torch.as_tensor(N, dtype=torch.float32).to(self.device)
        n = M.shape[0]
        return {"M": M, "N": N, "P": torch.empty(n, n, device=self.device), "n": n}

    def _random(self, n):
        return self._case(torch.empty(n, n).uniform_(-10, 10), torch.empty(n, n).uniform_(-10, 10))

    def examples(self):
        return [
            self._case([[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [1.0, 3.0]]),
            self._case(
                [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0]],
                [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]],
            ),
        ]

    # 32 threads per block: 31, 32 and 33 sit at the block edge; 70 and 100 span several blocks.
    def tests(self):
        return [
            self._case([[-3.5]], [[2.0]]),
            self._random(31),
            self._random(32),
            self._random(33),
            self._random(70),
            self._random(100),
        ]
