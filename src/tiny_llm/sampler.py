import mlx.core as mx
from .basics import softmax


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logits: mx.array):
        # use greedy when temperature is 0
        if temp == 0:
            return mx.argmax(logits, axis=-1)

        masked_logits = logits
        if top_p is not None:
            assert top_p > 0, f"expect top_p is a positive number, but get {top_p}."
            # get the real probability values from logits because we need to compare them with top_p
            prob = softmax(logits, axis=-1)
            # sort in descending order
            sort_idx = mx.argsort(prob, axis=-1)[..., ::-1]
            # indices to put array back to the unsorted order
            revert_idx = mx.argsort(sort_idx, axis=-1)

            # find the tokens in top p
            sort_prob = mx.take_along_axis(prob, indices=sort_idx, axis=-1)
            cumsum = mx.cumsum(sort_prob, axis=-1)
            less_than_p = cumsum < top_p
            less_than_p[..., 0] = True # at least one token with the highest probability
            top_p_idx = mx.take_along_axis(less_than_p, indices=revert_idx, axis=-1)

            # mark tokens not in top p with -inf
            masked_logits = mx.where(top_p_idx, masked_logits, -mx.inf)

        if top_k is not None:
            assert top_k > 0, f"expect top_k is a positive integer, but get {top_k}."
            vocab_size = logits.shape[-1]
            # find tokens not in top k
            idx = mx.argpartition(logits, kth=vocab_size-top_k, axis=-1)
            not_top_k_idx = idx[..., :top_k+1]
            # mark tokens not in top k with -inf
            masked_logits = mx.put_along_axis(masked_logits, indices=not_top_k_idx, values=mx.array(-mx.inf), axis=-1)

        return mx.random.categorical(masked_logits / temp, axis=-1)

    return sample
