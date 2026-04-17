import importlib.metadata

from .train_bpe import train_bpe, encode_bytes, bytes_to_unicode
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
