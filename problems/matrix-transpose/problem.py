import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    # A transpose only copies values, so atol = rtol = 0: the result must match exactly.

    def reference(self, input, output, rows, cols):
        output.copy_(input.T)

    def signature(self):
        return {
            "input": (ctypes.c_void_p, "in"),
            "output": (ctypes.c_void_p, "out"),
            "rows": (ctypes.c_int, "in"),
            "cols": (ctypes.c_int, "in"),
        }

    def _case(self, values):
        input = torch.as_tensor(values, dtype=torch.float32).to(self.device)
        rows, cols = input.shape
        output = torch.empty(cols, rows, device=self.device)
        return {"input": input, "output": output, "rows": rows, "cols": cols}

    def _random(self, rows, cols):
        return self._case(torch.empty(rows, cols).uniform_(-100, 100))

    def examples(self):
        return [
            self._case([[0.5, -1.0, 2.5, 4.0], [3.0, 7.5, -2.0, 6.0]]),
            self._case([[9.0, 8.5, -3.0]]),
        ]

    # Every case but 1x1 and 100x100 is non-square: a grid with x and y swapped, or an output index
    # that still steps by cols, fails there while passing on a square matrix.
    def tests(self):
        return [
            self._case([[4.0]]),
            self._random(1, 37),
            self._random(37, 1),
            # More than one block along both x and y.
            self._random(17, 33),
            self._random(64, 100),
            self._random(250, 40),
            self._random(100, 100),
        ]
