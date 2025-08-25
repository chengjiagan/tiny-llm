import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """
    L is seq_len, in PyTorch API it's S (source len)
    D is head_dim

    key: N.. x L x D
    value: N.. x L x D
    query: N.. x L x D
    output: N.. x L x D
    scale = 1/sqrt(D) if not specified
    output: N.. x L x D
    """

    D = value.shape[-1]
    attn_scale = mx.rsqrt(D) if scale is None else mx.array(scale)

    attn_weight = query @ key.swapaxes(-2, -1) * attn_scale
    if mask is not None:
        attn_weight += mask
    attn = softmax(attn_weight, -1) @ value

    return attn


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.H = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        E is hidden_size or embed_dim or dims or model_dim
        H is num_heads
        D is head_dim
        L is seq_len, in PyTorch API it's S (source len)

        query/key/value: N.. x L x E
        mask: N.. x L x L
        w_q/w_k/w_v: (H x D) x E
        w_o: E x (H x D)
        output: N.. x L x E
        """

        batch_dim = query.shape[:-1]
        query_proj = linear(query, self.wq).reshape(*batch_dim, self.H, -1).swapaxes(-2, -3)
        key_proj = linear(key, self.wk).reshape(*batch_dim, self.H, -1).swapaxes(-2, -3)
        value_proj = linear(value, self.wv).reshape(*batch_dim, self.H, -1).swapaxes(-2, -3)

        attn = scaled_dot_product_attention_simple(query_proj, key_proj, value_proj, mask=mask)
        attn_proj = linear(attn.swapaxes(-2, -3).reshape(*batch_dim, -1), self.wo)

        return attn_proj


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
