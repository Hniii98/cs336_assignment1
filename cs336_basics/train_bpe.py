from .pretokenization import parallel_pre_tokenization
from .merge import merge_pairs
from .utils import AutoKeyDictWrapper
import os


def bytes_to_unicode() -> dict[int: str]:
	valid_bytes = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
	unicode_map = valid_bytes[:]

	n = 0
	for i in range(256):
		if i not in valid_bytes:
			valid_bytes.append(i)
			unicode_map.append(256 + n)
			n +=1
	
	unicode_map = [chr(c) for c in unicode_map]
	return dict(zip(valid_bytes, unicode_map))


def encode_bytes(token_bytes: bytes,
				 encoder: dict[int: str]) -> str:
	return "".join(encoder[b] for b in token_bytes)


def train_bpe(
	input_path: str,
	vocab_size: int,
	special_tokens: list[str],
	verbose: bool = False,
) -> tuple[ dict[int, bytes],
		   	list[tuple[bytes, bytes]]]:
	
	num_processor = os.cpu_count() # including logical core
	
	freq_map = parallel_pre_tokenization(input_path=input_path, 
									  	 num_processes=num_processor, 
									  	 special_tokens=special_tokens)
	
	initial_vocab = AutoKeyDictWrapper()
	
	bytes_chars = [bytes([n]) for n in range(256)] 
	bytes_special_tokens = [bytes(token.encode("utf-8")) for token in special_tokens]

	initial_vocab.add_list(bytes_special_tokens + bytes_chars)

	merges = merge_pairs(freq_map=freq_map, 
					  	 initial_vocab=initial_vocab.dict, 
						 vocab_size=vocab_size,
						 verbose=verbose)

	return initial_vocab.dict, merges
	
	
	