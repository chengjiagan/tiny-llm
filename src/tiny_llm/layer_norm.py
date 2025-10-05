import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        assert len(weight.shape) == 1, f"expect weight to be a 1D array, but get {len(weight.shape)}."
        assert weight.shape[0] == dim, f"expect weight's size to equal to {dim}, but get {weight.shape[0]}."

        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.dim, f"expect the size input's last dimension to equal to {self.dim}, but get {x.shape[-1]}."

        scale = mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32)) + self.eps, axis=-1, keepdims=True))
        return (x * scale * self.weight).astype(x.dtype)
