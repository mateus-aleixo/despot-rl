//! MT19937 matching CPython's `random.Random`.
//!
//! Crit and evasion consume random numbers, so without a bit-identical RNG the
//! core could only be compared with the oracle statistically. Reproducing
//! CPython's Mersenne Twister -- including its `init_by_array` seeding for
//! integer seeds and the 53-bit `genrand_res53` double -- keeps those fights
//! exactly diffable instead.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct PyRandom {
    mt: [u32; N],
    mti: usize,
}

impl PyRandom {
    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = UPPER_MASK;
    }

    /// Seed the way `random.Random(n)` does for a non-negative integer `n`:
    /// CPython converts `abs(n)` into an array of 32-bit words, little end
    /// first, and calls `init_by_array`.
    pub fn seed_u64(seed: u64) -> Self {
        let mut r = PyRandom { mt: [0; N], mti: N + 1 };
        let key: Vec<u32> = if seed == 0 {
            vec![0]
        } else {
            let mut k = Vec::new();
            let mut v = seed;
            while v > 0 {
                k.push((v & 0xffff_ffff) as u32);
                v >>= 32;
            }
            k
        };
        r.init_by_array(&key);
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            for i in 0..N {
                let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
                let mut next = self.mt[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.mt[i] = next;
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython's `genrand_res53`: the 53-bit double behind `random.random()`.
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `random.getrandbits(k)` for k <= 32. CPython returns 0 for k == 0
    /// *without drawing*, which matters: `choice` on a one-element sequence
    /// would otherwise consume a word the oracle never consumes.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        self.next_u32() >> (32 - k.min(32))
    }

    /// `Random._randbelow_with_getrandbits`: reject until the draw fits, which
    /// is why the number of words consumed depends on the values drawn.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let k = 32 - n.leading_zeros(); // n.bit_length()
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.choice(seq)` for a sequence of `len` items.
    #[inline]
    pub fn choice(&mut self, len: usize) -> usize {
        self.randbelow(len as u32) as usize
    }
}
