"""Rijndael with a 256-bit block (Nb=8), as produced by .NET's
RijndaelManaged with BlockSize=256. AES is Rijndael restricted to Nb=4,
so no standard AES library can decrypt this.
"""
SBOX = bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16")
INV_SBOX = bytearray(256)
for i, v in enumerate(SBOX):
    INV_SBOX[v] = i
INV_SBOX = bytes(INV_SBOX)

def xtime(a):
    a <<= 1
    return (a ^ 0x1B) & 0xFF if a & 0x100 else a

MUL = [[0] * 256 for _ in range(16)]
for c in range(16):
    for x in range(256):
        r, a, b = 0, x, c
        while b:
            if b & 1: r ^= a
            a = xtime(a); b >>= 1
        MUL[c][x] = r

NB = 8          # 256-bit block
SHIFTS = (0, 1, 3, 4)   # Rijndael row shifts for Nb=8
RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A]

def expand_key(key):
    nk = len(key) // 4
    nr = max(nk, NB) + 6
    w = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
    for i in range(nk, NB * (nr + 1)):
        t = list(w[i - 1])
        if i % nk == 0:
            t = t[1:] + t[:1]
            t = [SBOX[b] for b in t]
            t[0] ^= RCON[i // nk]
        elif nk > 6 and i % nk == 4:
            t = [SBOX[b] for b in t]
        w.append([w[i - nk][j] ^ t[j] for j in range(4)])
    return w, nr

def _add_round_key(s, w, rnd):
    for c in range(NB):
        k = w[rnd * NB + c]
        for r in range(4):
            s[r][c] ^= k[r]

def decrypt_block(block, w, nr):
    s = [[block[r + 4 * c] for c in range(NB)] for r in range(4)]
    _add_round_key(s, w, nr)
    for rnd in range(nr - 1, -1, -1):
        for r in range(1, 4):                     # InvShiftRows
            n = SHIFTS[r]
            s[r] = s[r][-n:] + s[r][:-n]
        for r in range(4):                        # InvSubBytes
            s[r] = [INV_SBOX[b] for b in s[r]]
        _add_round_key(s, w, rnd)
        if rnd:                                   # InvMixColumns
            for c in range(NB):
                a0, a1, a2, a3 = s[0][c], s[1][c], s[2][c], s[3][c]
                s[0][c] = MUL[14][a0] ^ MUL[11][a1] ^ MUL[13][a2] ^ MUL[9][a3]
                s[1][c] = MUL[9][a0] ^ MUL[14][a1] ^ MUL[11][a2] ^ MUL[13][a3]
                s[2][c] = MUL[13][a0] ^ MUL[9][a1] ^ MUL[14][a2] ^ MUL[11][a3]
                s[3][c] = MUL[11][a0] ^ MUL[13][a1] ^ MUL[9][a2] ^ MUL[14][a3]
    return bytes(s[r][c] for c in range(NB) for r in range(4))

def encrypt_block(block, w, nr):
    s = [[block[r + 4 * c] for c in range(NB)] for r in range(4)]
    _add_round_key(s, w, 0)
    for rnd in range(1, nr + 1):
        for r in range(4):
            s[r] = [SBOX[b] for b in s[r]]
        for r in range(1, 4):
            n = SHIFTS[r]
            s[r] = s[r][n:] + s[r][:n]
        if rnd != nr:
            for c in range(NB):
                a0, a1, a2, a3 = s[0][c], s[1][c], s[2][c], s[3][c]
                s[0][c] = MUL[2][a0] ^ MUL[3][a1] ^ a2 ^ a3
                s[1][c] = a0 ^ MUL[2][a1] ^ MUL[3][a2] ^ a3
                s[2][c] = a0 ^ a1 ^ MUL[2][a2] ^ MUL[3][a3]
                s[3][c] = MUL[3][a0] ^ a1 ^ a2 ^ MUL[2][a3]
        _add_round_key(s, w, rnd)
    return bytes(s[r][c] for c in range(NB) for r in range(4))

def cbc_decrypt(data, key, iv, limit=None):
    w, nr = expand_key(key)
    out, prev = bytearray(), iv
    end = len(data) if limit is None else min(len(data), limit)
    for i in range(0, end, 32):
        blk = data[i:i + 32]
        if len(blk) < 32: break
        pt = decrypt_block(blk, w, nr)
        out += bytes(a ^ b for a, b in zip(pt, prev))
        prev = blk
    return bytes(out)

if __name__ == "__main__":
    import os
    key, blk = os.urandom(32), os.urandom(32)
    w, nr = expand_key(key)
    assert decrypt_block(encrypt_block(blk, w, nr), w, nr) == blk
    print(f"round-trip OK (Nb={NB}, Nr={nr})")
