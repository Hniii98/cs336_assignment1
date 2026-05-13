import torch
from torch import nn

from .rmsnorm import RMSNorm
from .attention import MHSA
from .swiglu import SwiGLU

from jaxtyping import Float, Int
from torch import Tensor


class TransformerBlock(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		d_ff: int,
		theta: float,
		max_seq_len: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None
	):
		super().__init__()


		self.rmsnorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
		self.rmsnorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
		self.multi_head_attention = MHSA(
			d_model=d_model, 
			num_heads=num_heads, 
			max_seq_len=max_seq_len, 
			theta=theta, 
			device=device,
			dtype=dtype
		)
		self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
		

	def forward(
		self,
		in_features: Float[Tensor, "batch_size seq_len d_model"],
		token_positions: Int[Tensor, "seq_len"]
	) -> Float[Tensor, "batch_size seq_len d_model"]:
		
		# Attention block
		attn_output = self.multi_head_attention(self.rmsnorm1(in_features), token_positions)
		x = attn_output + in_features

		# FFN block
		ffn_output = self.swiglu(self.rmsnorm2(x))
		x = x + ffn_output

		return x

		
		