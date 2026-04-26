"""
This module defines a character-level tokenization system for a 
spelling correction model.
"""

import numpy as np


def vocab():
    """Return full vocabulary."""
    special_tokens = ["<pad>", "<bos>", "<eos>"]
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    return special_tokens + chars


class Tokenization:
    def __init__(self, max_len: int = 20):
        self.max_len = max_len
        self.vocab = vocab()

        self.token_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_token = {i: ch for ch, i in self.token_to_id.items()}

    def encode_input(self, txt: str) -> np.array:
        # encode misspelled input string
        ids = [self.token_to_id[c] for c in txt] + [self.eos]
        ids += [self.pad] * ((self.max_len + 1) - len(ids))
        return np.array(ids, dtype=np.int32)

    def encode_label(self, txt: str):
        """Encode correct label (decoder input + target output)."""
        dec_in = [self.bos] + [self.token_to_id[c] for c in txt]
        dec_in += [self.pad] * ((self.max_len + 2) - len(dec_in))

        dec_out = [self.token_to_id[c] for c in txt] + [self.eos]
        dec_out += [self.pad] * ((self.max_len + 2) - len(dec_out))

        return np.array(dec_in, dtype=np.int32), np.array(dec_out, dtype=np.int32)

    def decode(self, arr):
        """Convert tokens back to string."""
        return "".join(self.id_to_token[i] for i in arr if i > 2)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad(self):
        return self.token_to_id["<pad>"]

    @property
    def bos(self):
        return self.token_to_id["<bos>"]

    @property
    def eos(self):
        return self.token_to_id["<eos>"]