import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any, List
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        assert hidden_size % num_heads == 0, f"expect hidden_size to be divisible by num_heads, but get hidden_size={hidden_size}, num_heads={num_heads}."
        assert num_heads % num_kv_heads == 0, f"expect num_heads to be divisible by num_kv_heads, but get num_heads={num_heads}, num_kv_heads={num_kv_heads}."
        head_dim = hidden_size // num_heads
        assert len(wq.shape) == 2, f"expect wq to be a 2D array, but get {len(wq.shape)}."
        assert len(wk.shape) == 2, f"expect wk to be a 2D array, but get {len(wk.shape)}."
        assert len(wv.shape) == 2, f"expect wv to be a 2D array, but get {len(wv.shape)}."
        assert len(wo.shape) == 2, f"expect wo to be a 2D array, but get {len(wo.shape)}."
        assert len(bq.shape) == 1, f"expect bq to be a 1D array, but get {len(bq.shape)}."
        assert len(bk.shape) == 1, f"expect bk to be a 1D array, but get {len(bk.shape)}."
        assert len(bv.shape) == 1, f"expect bv to be a 1D array, but get {len(bv.shape)}."
        assert wq.shape[0] == hidden_size, f"expect wq.shape[0] == hidden_size, but get wq.shape[0]={wq.shape[0]}, hidden_size={hidden_size}."
        assert wk.shape[0] == num_kv_heads * head_dim, f"expect wk.shape[0] == num_kv_heads * head_dim, but get wk.shape[0]={wk.shape[0]}, num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        assert wv.shape[0] == num_kv_heads * head_dim, f"expect wv.shape[0] == num_kv_heads * head_dim, but get wv.shape[0]={wv.shape[0]}, num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        assert wo.shape[1] == hidden_size, f"expect wo.shape[1] == hidden_size, but get wo.shape[1]={wo.shape[1]}, hidden_size={hidden_size}."
        assert bq.shape[0] == hidden_size, f"expect bq.shape[0] == hidden_size, but get bq.shape[0]={bq.shape[0]}, hidden_size={hidden_size}."
        assert bk.shape[0] == num_kv_heads * head_dim, f"expect bk.shape[0] == num_kv_heads * head_dim, but get bk.shape[0]={bk.shape[0]}, num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        assert bv.shape[0] == num_kv_heads * head_dim, f"expect bv.shape[0] == num_kv_heads * head_dim, but get bv.shape[0]={bv.shape[0]}, num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        E = wq.shape[1]
        assert wk.shape[1] == E and wv.shape[1] == E and wo.shape[0] == E, f"expect the second dimensions of wq, wk, wv and the first dimension of wo are the same, but get wq.shape[1]={wq.shape[1]}, wk.shape[1]={wk.shape[1]}, wv.shape[1]={wv.shape[1]}, wo.shape[0]={wo.shape[0]}."

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.input_size = E
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta, False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        assert len(x.shape) == 3, f"expect x to have 3 dimensions, but get {len(x.shape)}."
        B, L, E = x.shape
        assert L <= self.max_seq_len, f"expect input sequence length to be less than or equal to max_seq_len ({self.max_seq_len}), but get {L}."
        assert E == self.input_size, f"expect input hidden size to be {self.input_size}, but get {E}."

        # generate q, k, v and apply rope
        q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim)
        k = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim)
        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))
        
        # put the head dim before sequence dim
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        # apply GQA in FP32
        out = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            mask=mask,
        ).astype(x.dtype)

        # put the head dim after sequence dim and apply output transform
        out = out.swapaxes(1, 2).reshape(B, L, self.hidden_size)
        out = linear(out, self.wo)

        return out


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        """
        N.. is zero or more dimensions for batches
        E is hidden_size (embedding dimension of the model) (dim)
        I is intermediate_size (dimension of the hidden layer in MLP) (hidden_dim)
        L is the sequence length

        input: N.. x L x E
        w_gate: I x E
        w_up: I x E
        w_down: E x I
        output: N.. x L x E
        """
        assert w_gate.shape == (hidden_dim, dim), f"expect w_gate's shape to be (hidden_dim, dim), but get w_gate.shape={w_gate.shape}, dim={dim}, hidden_dim={hidden_dim}."
        assert w_up.shape == (hidden_dim, dim), f"expect w_up's shape to be (hidden_dim, dim), but get w_up.shape={w_up.shape}, dim={dim}, hidden_dim={hidden_dim}."
        assert w_down.shape == (dim, hidden_dim), f"expect w_down's shape to be (dim, hidden_dim), but get w_down.shape={w_down.shape}, dim={dim}, hidden_dim={hidden_dim}."

        self.dim = dim # input size
        self.hidden_dim = hidden_dim # intermediate size
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        assert len(x.shape) >= 2, f"expect input x to have at least 2 dimensions, but get {len(x.shape)}."
        assert x.shape[-1] == self.dim, f"expect the last dimension of input x to be {self.dim}, but get {x.shape[-1]}."

        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len

        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.self_attn = Qwen2MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads, wq, wk, wv, wo, bq, bk, bv, max_seq_len, theta)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        assert len(x.shape) >= 2, f"expect input x to have at least 2 dimensions, but get {len(x.shape)}."
        assert x.shape[-1] == self.hidden_size, f"expect input x's last dimension to equal to {self.hidden_size}, but get {x.shape[-1]}."
        assert x.shape[-1] <= self.max_seq_len, f"expect input x's sequence length to be less than or equal to max_seq_len ({self.max_seq_len}), but get {x.shape[-1]}."

        x_attn = x + self.self_attn(self.input_layernorm(x), mask)
        x_out = x_attn + self.mlp(self.post_attention_layernorm(x_attn))
        return x_out


from mlx_lm.models.qwen2 import Model
class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Model):
        self.precision = mx.float16 # use FP16 as compute precision

        args = mlx_model.args
        self.tie_word_embeddings = args.tie_word_embeddings

        # helper function to get dequantized weight
        def dq_weight(l: Any) -> mx.array:
            return dequantize_linear(l).astype(self.precision)

        w_embed = dq_weight(mlx_model.model.embed_tokens)
        self.embed_tokens = Embedding(args.vocab_size, args.hidden_size, w_embed)

        self.layers: List[Qwen2TransformerBlock] = []
        for i in range(args.num_hidden_layers):
            mlx_layer = mlx_model.model.layers[i]
            mlx_mlp = mlx_layer.mlp
            mlx_attn = mlx_layer.self_attn
            layer = Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=dq_weight(mlx_attn.q_proj),
                wk=dq_weight(mlx_attn.k_proj),
                wv=dq_weight(mlx_attn.v_proj),
                wo=dq_weight(mlx_attn.o_proj),
                bq=mlx_attn.q_proj.bias.astype(self.precision),
                bk=mlx_attn.k_proj.bias.astype(self.precision),
                bv=mlx_attn.v_proj.bias.astype(self.precision),
                w_gate=dq_weight(mlx_mlp.gate_proj),
                w_up=dq_weight(mlx_mlp.up_proj),
                w_down=dq_weight(mlx_mlp.down_proj),
                w_input_layernorm=mlx_layer.input_layernorm.weight.astype(self.precision),
                w_post_attention_layernorm=mlx_layer.post_attention_layernorm.weight.astype(self.precision),
                max_seq_len=args.max_position_embeddings,
                theta=int(args.rope_theta)
            )
            self.layers.append(layer)

        w_norm = mlx_model.model.norm.weight.astype(self.precision)
        self.norm = RMSNorm(args.hidden_size, w_norm, args.rms_norm_eps)

        if not args.tie_word_embeddings:
            self.w_lm_head = dq_weight(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        # get embbedings of input tokens
        x = self.embed_tokens(inputs)

        # go through transformer layers
        for l in self.layers:
            x = l(x, "causal")

        # last rms norm
        x = self.norm(x)

        # get token probability distribution
        if self.tie_word_embeddings:
            prob = self.embed_tokens.as_linear(x)
        else:
            prob = linear(x, self.w_lm_head)

        return prob

