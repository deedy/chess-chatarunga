//! Zobrist hash keys, computed deterministically at compile time.

const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub struct ZobristKeys {
    /// `[color][piece][sq]`
    pub piece: [[[u64; 64]; 6]; 2],
    pub side: u64,
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
}

pub const ZOBRIST: ZobristKeys = {
    let mut s = 0x517c_c1b7_2722_0a95u64;
    let mut piece = [[[0u64; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut sq = 0;
            while sq < 64 {
                s = splitmix64(s);
                piece[c][p][sq] = s;
                sq += 1;
            }
            p += 1;
        }
        c += 1;
    }
    s = splitmix64(s);
    let side = s;
    let mut castling = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        s = splitmix64(s);
        castling[i] = s;
        i += 1;
    }
    let mut ep_file = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        s = splitmix64(s);
        ep_file[f] = s;
        f += 1;
    }
    ZobristKeys {
        piece,
        side,
        castling,
        ep_file,
    }
};
