from numbers import Number
from typing import Callable, List, Union, Tuple, Iterable


__all__ = ["to_tuple"]


def to_tuple(v: Union[Tuple[Number, ...], Number, Iterable], n: int):
    """Converts input to tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, Number):
        return (v,) * n
    else:
        return tuple(v)
