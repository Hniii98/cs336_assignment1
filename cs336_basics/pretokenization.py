import os
import regex as re
from typing import BinaryIO, Tuple
from multiprocessing import Pool
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set[int](chunk_boundaries))

# Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip[tuple[int, int]](boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


def pre_tokenization_task (
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str]
) -> Counter[tuple[bytes, ...]]:
    """
    Counting the pre-tokens in each slice of the chunk.
    """
    assert special_tokens , "Special token must not be empty or None"
    freq_map = Counter()

    with open(input_path, "r") as f:
        f.seek(start)
        chunk = f.read(end - start)

        # Split chunks by using special tokens as hard boundaries
        # to prevent pre-tokenization merges across document boundaries
        pattern = "|".join(re.escape(token) for token in special_tokens)
        slices = re.split(pattern, chunk)

        for slice in slices:
            for match in re.finditer(PAT, slice):
                token = match.group()
                freq_map[tuple(bytes([b]) for b in token.encode("utf-8"))] += 1

    return freq_map


def parallel_pre_tokenization (
    input_path: str,
    special_tokens: list[str]          
) -> dict[tuple[bytes, ...], int] :
    """
    Parallelizing pre-tokenization procedure and return the map from
    token to frequency.
    """
    # Use Counter() to aggregate counts of the same key across multiple dicts.
    freq_map_in_all = Counter()
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        args_list: list[tuple[str, int, int, list[str]]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args_list.append((input_path, start, end, special_tokens))  

        
        with Pool() as pool:
            results = pool.starmap(pre_tokenization_task, args_list) 
            for map in results:
                freq_map_in_all.update(map)


    return dict(freq_map_in_all)
    




    
        





