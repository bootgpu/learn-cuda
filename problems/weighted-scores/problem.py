import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    # Each total sums up to `subjects` products of size up to 100.
    atol = 1e-3
    rtol = 1e-4

    def reference(self, scores, weights, totals, students, subjects):
        torch.mv(scores, weights, out=totals)

    def signature(self):
        return {
            "scores": (ctypes.c_void_p, "in"),
            "weights": (ctypes.c_void_p, "in"),
            "totals": (ctypes.c_void_p, "out"),
            "students": (ctypes.c_int, "in"),
            "subjects": (ctypes.c_int, "in"),
        }

    def _case(self, scores, weights):
        scores = torch.as_tensor(scores, dtype=torch.float32).to(self.device)
        weights = torch.as_tensor(weights, dtype=torch.float32).to(self.device)
        students, subjects = scores.shape
        return {
            "scores": scores,
            "weights": weights,
            "totals": torch.empty(students, device=self.device),
            "students": students,
            "subjects": subjects,
        }

    def _random(self, students, subjects):
        return self._case(torch.empty(students, subjects).uniform_(0, 100), torch.rand(subjects))

    def examples(self):
        return [
            self._case([[80.0, 90.0, 70.0], [60.0, 100.0, 50.0]], [0.3, 0.5, 0.2]),
            self._case([[90.0, 80.0], [70.0, 60.0], [100.0, 40.0]], [0.5, 0.5]),
        ]

    # All but 1x1 are non-square: reading scores column by column (j * students + i) fails there.
    def tests(self):
        return [
            self._case([[88.0]], [0.75]),
            self._random(3, 7),
            self._random(7, 3),
            # One student with a long row.
            self._random(1, 500),
            # More than one block.
            self._random(300, 30),
            self._random(1000, 10),
        ]
