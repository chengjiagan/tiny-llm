import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "expect dims to be even"

        self.dims = dims
        self.max_seq_len = seq_len
        self.traditional = traditional

        theta = base ** (-2 * mx.arange(dims//2) / dims) # shape: (dims/2)
        freqs = mx.outer(mx.arange(seq_len), theta) # shape: (max_seq_len, dims/2)
        self.sin_freqs = mx.sin(freqs)
        self.cos_freqs = mx.cos(freqs)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        assert x.shape[-1] == self.dims, f"expect input has last dim size {self.dims}, but get {x.shape[-1]}."

        seq_len = x.shape[-3]
        # get the cos/sin values from pre-compute
        if offset is None:
            assert seq_len <= self.max_seq_len, f"expect input's sequence length is less than or equal to max sequence length f{self.max_seq_len}, but get {seq_len}."
            cos_slice = self.cos_freqs[:seq_len]
            sin_slice = self.sin_freqs[:seq_len]
        else:
            # do not handle offset is list[slice] now
            assert isinstance(offset, slice), "only support slice as offset now."

            assert offset.stop <= self.max_seq_len, f"expect offset's stop is less than or equal to max sequence length f{self.max_seq_len}, but get {offset.stop}."
            cos_slice = self.cos_freqs[offset]
            sin_slice = self.sin_freqs[offset]

        offset_len = cos_slice.shape[0]
        assert offset_len == seq_len, f"expect input's sequence length ({seq_len}) and offset's length ({offset_len}) are equal."

        # insert attention head dim, shape: (seq_len, 1, dims/2)
        cos_slice = mx.expand_dims(cos_slice, axis=1)
        sin_slice = mx.expand_dims(sin_slice, axis=1)

        # get the real and imaginary part from complex conjucate of input
        if self.traditional:
            x0 = x[..., ::2]
            x1 = x[..., 1::2]
        else:
            x0 = x[..., :self.dims//2]
            x1 = x[..., self.dims//2:]

        # compute the rotray output
        y0 = x0 * cos_slice - x1 * sin_slice
        y1 = x0 * sin_slice + x1 * cos_slice

        if self.traditional:
            y = mx.stack([y0, y1], axis=-1).flatten(-2, -1)
        else:
            y = mx.concat([y0, y1], axis=-1)

        return y
