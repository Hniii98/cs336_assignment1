import torch
from jaxtyping import Bool, Float
from torch import Tensor
from einops import einsum

from .softmax import softmax


def scaled_dot_product_attention(
	Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None
) -> Float[Tensor, "... quries d_v"]:
	d_k = Q.shape[-1]
	pre_softmax_logits = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
	pre_softmax_logits = pre_softmax_logits / (d_k ** 0.5)
	
	if mask is not None:
		mask = torch.where(mask, 0.0, float('-inf'))
		pre_softmax_logits = pre_softmax_logits + mask

	logits = softmax(pre_softmax_logits, -1)
	
	scores = einsum(logits, V, "... queries keys, ... keys d_v -> ... queries d_v")
	return scores
	


