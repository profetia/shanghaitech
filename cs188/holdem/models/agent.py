from typing import Tuple
from abc import ABC


class Agent(ABC):
    def __init__(self) -> None:
        pass

    def step(self, state: dict) -> int:
        raise NotImplementedError

    def eval_step(self, state: dict) -> Tuple[int, dict]:
        raise NotImplementedError
