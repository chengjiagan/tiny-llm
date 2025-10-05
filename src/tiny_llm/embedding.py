import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        assert weight.shape == (vocab_size, embedding_dim), f"expect weight's shape to be (vocab_size, embedding_dim), but get weight.shape={weight.shape}, vocab_size={vocab_size}, embedding_dim={embedding_dim}."

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return mx.take(self.weight, indices=x, axis=0)

    def as_linear(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.embedding_dim, f"expect input x's last dimension to be {self.embedding_dim}, but get {x.shape[-1]}."
        return x @ self.weight.transpose()
