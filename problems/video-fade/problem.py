import ctypes

import torch

from bootgpu.problem import ProblemBase


class Problem(ProblemBase):
    atol = 1e-6
    rtol = 1e-5

    def reference(self, input, output, frames, height, width):
        t = torch.arange(frames, dtype=torch.float32, device=input.device).view(-1, 1, 1)
        torch.mul(input, (frames - t) / frames, out=output)

    def signature(self):
        return {
            "input": (ctypes.c_void_p, "in"),
            "output": (ctypes.c_void_p, "out"),
            "frames": (ctypes.c_int, "in"),
            "height": (ctypes.c_int, "in"),
            "width": (ctypes.c_int, "in"),
        }

    def _case(self, values):
        input = torch.as_tensor(values, dtype=torch.float32).to(self.device)
        frames, height, width = input.shape
        return {
            "input": input,
            "output": torch.empty_like(input),
            "frames": frames,
            "height": height,
            "width": width,
        }

    def _random(self, frames, height, width):
        return self._case(torch.rand(frames, height, width))

    def examples(self):
        return [
            self._case(
                [
                    [[0.2, 0.4, 0.6], [0.8, 1.0, 0.0]],
                    [[1.0, 0.6, 0.2], [0.0, 0.4, 0.8]],
                ]
            ),
            self._case([[[1.0, 0.4]]] * 4),
        ]

    # solve launches 8 x 8 x 4 blocks. Frame numbers past 3 need blockIdx.z, so a kernel that takes
    # the frame from threadIdx.z alone fails from test_3 on.
    def tests(self):
        return [
            # One frame: z threads 1-3 of the only block are past the end.
            self._random(1, 5, 7),
            # Exactly one block.
            self._random(4, 8, 8),
            # More than one block along x, y and z.
            self._random(5, 9, 9),
            self._random(10, 25, 40),
            # 25 blocks along z.
            self._random(100, 10, 10),
        ]
