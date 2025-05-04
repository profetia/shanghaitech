import hashlib
from typing import List, Tuple
import torch
import random
import numpy as np
import struct
import os
from numpy.random import RandomState


__global_seed = 3407
__global_generator = torch.Generator()
__global_generator.manual_seed(__global_seed)


def set_global_seed(seed: int = 3407) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    __global_seed = seed
    __global_generator.manual_seed(__global_seed)


def split_int(target: int) -> List[int]:
    assert target >= 0

    if target == 0:
        return [0]

    result = []
    while target > 0:
        target, residue = divmod(target, 2**32)
        result.append(residue)
    return result


def from_bytes(target: bytes) -> int:
    target += b"\0" * (4 - len(target) % 4)
    result: int = 0
    for i, val in enumerate(
        struct.unpack("{}I".format(len(target) // 4), target)
    ):
        result += 2 ** (32 * i) * val
    return result


def hash_seed(seed: int | None = None, max_bytes: int = 8) -> int:
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)

    result: int = from_bytes(
        hashlib.sha512(str(seed).encode("utf8")).digest()[:max_bytes]
    )
    return result


def create_seed(seed: int | str | None = None, max_bytes: int = 8) -> int:
    match seed:
        case None:
            result = from_bytes(os.urandom(max_bytes))
        case int():
            result = seed % 2 ** (8 * max_bytes)
        case str():
            result = from_bytes(
                hashlib.sha512(seed.encode("utf8")).digest()[:max_bytes]
            )

    return result


def create_random_generator(
    seed: int | None = None,
) -> Tuple[RandomState, int]:
    if seed is None:
        seed = create_seed()

    generator = RandomState()
    generator.seed(split_int(hash_seed(seed)))
    return generator, seed
