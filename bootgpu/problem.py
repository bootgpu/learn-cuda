from abc import ABC, abstractmethod
from typing import Any


class ProblemBase(ABC):
    atol: float = 0.0
    rtol: float = 0.0

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def reference(self, **kwargs) -> None:
        """Write the expected result into the "out" and "inout" tensors in place."""

    @abstractmethod
    def signature(self) -> dict[str, tuple]:
        """Ordered {arg: (ctypes type, "in" | "out" | "inout")}, same order as solve()."""

    @abstractmethod
    def examples(self) -> list[dict[str, Any]]:
        """The statement's examples, in order, with fixed values; at least one."""

    @abstractmethod
    def tests(self) -> list[dict[str, Any]]: ...
