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
    mask: N.. x L x L
    scale = 1/sqrt(D) if not specified
    output: N.. x L x D
    """
    assert len(query.shape) >= 2, f"expect query to have at least 2 dims, but get {len(query.shape)}"
    assert len(key.shape) >= 2, f"expect key to have at least 2 dims, but get {len(key.shape)}"
    assert len(value.shape) >= 2, f"expect value to have at least 2 dims, but get {len(value.shape)}"
    if mask is not None:
        assert len(mask.shape) >= 2, f"expect mask to have least 2 dims, but get {len(mask.shape)}"

    L, D = query.shape[-2:]
    assert key.shape[-2] == L, f"expect key to have the same sequence length as query ({L}), but get {key.shape[-2]}"
    assert key.shape[-1] == D, f"expect key to have the same head dimension as query ({D}), but get {key.shape[-1]}"
    assert value.shape[-2] == L, f"expect value to have the same sequence length as query ({L}), but get {value.shape[-2]}"
    assert value.shape[-1] == D, f"expect value to have the same head dimension as query ({D}), but get {value.shape[-1]}"
    if mask is not None:
        assert mask.shape[-1] == L, f"expect the last dim of mask to have the same size as sequence length ({L}), but get {mask.shape[-1]}."
        assert mask.shape[-2] == L, f"expect the second to last dim of mask to have the same size as sequence length ({L}), but get {mask.shape[-2]}."

    attn_scale = mx.rsqrt(D) if scale is None else mx.array(scale)

    attn_score = query @ key.swapaxes(-2, -1) * attn_scale
    if mask is not None:
        attn_score += mask
    attn = softmax(attn_score, axis=-1) @ value

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
    assert L <= S, f"expect L <= S, but get L={L}, S={S}."
    return mx.triu(mx.full((L, S), -mx.inf, dtype), S - L + 1)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """
    N.. is zero or more dimensions for batches
    H_q is the number of query heads
    H is the number of key/value heads (H_q must be divisible by H)
    L is the query sequence length
    S is the key/value sequence length
    D is the head dimension

    query: N.. x H_q x L x D
    key: N.. x H x S x D
    value: N.. x H x S x D
    mask: N.. x H_q x L x S
    output: N.. x H_q x L x D
    """
    assert len(query.shape) >= 3, f"expect query to have at least 3 dims, but get {len(query.shape)}."
    assert len(key.shape) >= 3, f"expect key to have at least 3 dims, but get {len(key.shape)}."
    assert len(value.shape) >= 3, f"expect value to have at least 3 dims, but get {len(value.shape)}."
    if type(mask) == mx.array:
        assert len(mask.shape) >= 3, f"expect mask to have at least 3 dims, but get {len(mask.shape)}."
    
    H_q, L, D = query.shape[-3:]
    H, S = key.shape[-3:-1]
    assert H_q % H == 0, f"expect query's number of heads ({H_q}) to be divisible by key's number of heads ({H})."
    assert key.shape[-1] == D, f"expect key to have the same head dimension as query ({D}), but get {key.shape[-1]}."
    assert value.shape[-1] == D, f"expect value to have the same head dimension as query ({D}), but get {value.shape[-1]}."
    assert value.shape[-2] == S, f"expect value to have the same sequence length as key ({S}), but get {value.shape[-2]}."
    assert value.shape[-3] == H, f"expect value to have the same number of heads as key ({H}), but get {key.shape[-3]}."
    if type(mask) == mx.array:
        assert mask.shape[-1] == S, f"expect the last dim of mask to have the same size as key sequence length ({S}), but get {mask.shape[-1]}"
        assert mask.shape[-2] == L, f"expect the second to last dims of mask to have the same size as query sequence length ({L}), but get {mask.shape[-2]}"
        assert mask.shape[-3] == H_q, f"expect mask to have the same number of heads as query ({H_q}), but get {mask.shape[-3]}"
    elif type(mask) == str:
        assert mask == "causal", f"only causal mask is supported, but get {mask}."
    else:
        assert mask is None, f"unsupported mask type {type(mask)}."


    n_repeats = H_q // H
    # reshape query, key, and value to add the head group dim
    query = mx.unflatten(query, axis=-3, shape=(H, n_repeats))
    key = mx.expand_dims(key, axis=-3)
    value = mx.expand_dims(value, axis=-3)

    mask_array: None | mx.array
    if mask is None:
        mask_array = mx.array(0, dtype=query.dtype)
    elif type(mask) == mx.array:
        # reshape mask to add the head group dim
        mask_array = mx.unflatten(mask, axis=-3, shape=(H, n_repeats))
    else:
        mask_array = causal_mask(L, S, dtype=query.dtype)

    # normal multi-head attention
    attn_scale = mx.rsqrt(D) if scale is None else mx.array(scale)
    attn_score = query @ key.swapaxes(-2, -1) * attn_scale + mask_array
    attn = softmax(attn_score, axis=-1) @ value

    # flatten head dim and head group dim
    attn = attn.flatten(-4, -3)
    return attn


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
