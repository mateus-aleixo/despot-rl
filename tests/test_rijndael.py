"""Rijndael with a 256-bit block (tools/rijndael256.py).

No mainstream library implements a 256-bit block, so there is nothing to
compare the Nb=8 cipher against directly. Instead: run the same code at Nb=4,
where Rijndael is AES, against the FIPS-197 known-answer vectors. That pins the
S-box, the key schedule, MixColumns and the round structure; Nb=8 then differs
only in the block width and the row shifts, which the round trips cover.
"""
import random

import pytest

import rijndael256 as rj


def cbc_encrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    w, nr = rj.expand_key(key)
    out, prev = bytearray(), iv
    for i in range(0, len(data), 32):
        block = bytes(a ^ b for a, b in zip(data[i:i + 32], prev))
        prev = rj.encrypt_block(block, w, nr)
        out += prev
    return bytes(out)


@pytest.mark.parametrize("key_len", [16, 24, 32])
def test_block_round_trip(key_len):
    rng = random.Random(key_len)
    for _ in range(20):
        key, block = rng.randbytes(key_len), rng.randbytes(32)
        w, nr = rj.expand_key(key)
        assert nr == 14, "Nr = max(Nk, Nb) + 6, and Nb = 8 dominates"
        ct = rj.encrypt_block(block, w, nr)
        assert ct != block
        assert rj.decrypt_block(ct, w, nr) == block


def test_cbc_decrypt_inverts_cbc():
    rng = random.Random(3)
    key, iv, data = rng.randbytes(32), rng.randbytes(32), rng.randbytes(32 * 5)
    ct = cbc_encrypt(data, key, iv)
    assert rj.cbc_decrypt(ct, key, iv) == data
    assert rj.cbc_decrypt(ct, key, iv, limit=64) == data[:64]
    # A trailing partial block is not decrypted.
    assert rj.cbc_decrypt(ct + b"\x00" * 7, key, iv) == data


# FIPS-197 Appendix C: plaintext 00112233...eeff, key 000102...
FIPS197 = [
    ("000102030405060708090a0b0c0d0e0f", 10, "69c4e0d86a7b0430d8cdb78070b4c55a"),
    ("000102030405060708090a0b0c0d0e0f1011121314151617", 12, "dda97ca4864cdfe06eaf70a0ec0d7191"),
    ("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", 14,
     "8ea2b7ca516745bfeafc49904b496089"),
]


@pytest.mark.parametrize("key, rounds, want", FIPS197, ids=["AES-128", "AES-192", "AES-256"])
def test_at_nb4_it_is_aes(monkeypatch, key, rounds, want):
    monkeypatch.setattr(rj, "NB", 4)
    monkeypatch.setattr(rj, "SHIFTS", (0, 1, 2, 3))
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    w, nr = rj.expand_key(bytes.fromhex(key))
    assert nr == rounds
    assert rj.encrypt_block(plaintext, w, nr).hex() == want
    assert rj.decrypt_block(bytes.fromhex(want), w, nr) == plaintext
