import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """
    I is input dimension. O is output dimension.

    x: N.. x I
    w: O x I
    bias: O
    output: N.. x O
    """

    y = x @ w.T
    if bias is not None:
        y += bias
    return y


def silu(x: mx.array) -> mx.array:
    pass
