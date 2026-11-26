//! Bitboard utilities and plain Magic Bitboards for fast sliding-piece attack lookup.

use crate::types::{Color, Square};
use std::sync::OnceLock;

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_B: u64 = FILE_A << 1;
pub const FILE_G: u64 = FILE_A << 6;
pub const FILE_H: u64 = FILE_A << 7;
pub const NOT_FILE_A: u64 = !FILE_A;
pub const NOT_FILE_H: u64 = !FILE_H;
pub const NOT_FILE_AB: u64 = !(FILE_A | FILE_B);
pub const NOT_FILE_GH: u64 = !(FILE_G | FILE_H);

pub const RANK_1: u64 = 0x0000_0000_0000_00FF;
pub const RANK_2: u64 = RANK_1 << 8;
pub const RANK_3: u64 = RANK_1 << 16;
pub const RANK_4: u64 = RANK_1 << 24;
pub const RANK_5: u64 = RANK_1 << 32;
pub const RANK_6: u64 = RANK_1 << 40;
pub const RANK_7: u64 = RANK_1 << 48;
pub const RANK_8: u64 = RANK_1 << 56;
pub const LIGHT_SQUARES: u64 = 0x55AA_55AA_55AA_55AA;

#[inline(always)]
pub const fn bb(sq: Square) -> u64 {
    1u64 << sq
}

#[inline(always)]
pub fn pop_lsb(b: &mut u64) -> Square {
    let sq = b.trailing_zeros() as Square;
    *b &= *b - 1;
    sq
}

#[inline(always)]
pub const fn file_bb(file: u8) -> u64 {
    FILE_A << file
}

#[inline(always)]
pub const fn rank_bb(rank: u8) -> u64 {
    RANK_1 << (rank * 8)
}

#[derive(Clone, Copy)]
struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

pub struct Tables {
    pub knight: [u64; 64],
    pub king: [u64; 64],
    pub pawn_att: [[u64; 64]; 2],
    /// Squares strictly between sq1 and sq2 (exclusive of endpoints), 0 if not aligned.
    pub between: [[u64; 64]; 64],
    /// Full ray/line passing through both sq1 and sq2 (inclusive), 0 if not aligned.
    pub line: [[u64; 64]; 64],
    /// Adjacent files mask for each file 0..8.
    pub adj_files: [u64; 8],
    /// Passed pawn front span mask `[color][sq]`.
    pub passed_mask: [[u64; 64]; 2],
    rook_magics: [Magic; 64],
    bishop_magics: [Magic; 64],
    rook_table: Vec<u64>,
    bishop_table: Vec<u64>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

#[inline(always)]
pub fn tables() -> &'static Tables {
    TABLES.get_or_init(init_tables)
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> u64 {
    tables().knight[sq as usize]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> u64 {
    tables().king[sq as usize]
}

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> u64 {
    tables().pawn_att[color.idx()][sq as usize]
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: u64) -> u64 {
    let t = tables();
    let m = unsafe { *t.bishop_magics.get_unchecked(sq as usize) };
    let idx = (((occ & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize;
    unsafe { *t.bishop_table.get_unchecked(m.offset + idx) }
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: u64) -> u64 {
    let t = tables();
    let m = unsafe { *t.rook_magics.get_unchecked(sq as usize) };
    let idx = (((occ & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize;
    unsafe { *t.rook_table.get_unchecked(m.offset + idx) }
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: u64) -> u64 {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

#[inline(always)]
pub fn between_bb(s1: Square, s2: Square) -> u64 {
    tables().between[s1 as usize][s2 as usize]
}

#[inline(always)]
pub fn line_bb(s1: Square, s2: Square) -> u64 {
    tables().line[s1 as usize][s2 as usize]
}

fn rook_mask(sq: usize) -> u64 {
    let r = sq / 8;
    let f = sq % 8;
    let mut m = 0u64;
    for rr in (r + 1)..7 {
        m |= 1u64 << (rr * 8 + f);
    }
    for rr in 1..r {
        m |= 1u64 << (rr * 8 + f);
    }
    for ff in (f + 1)..7 {
        m |= 1u64 << (r * 8 + ff);
    }
    for ff in 1..f {
        m |= 1u64 << (r * 8 + ff);
    }
    m
}

fn bishop_mask(sq: usize) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut m = 0u64;
    for &(dr, df) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let mut rr = r + dr;
        let mut ff = f + df;
        while rr > 0 && rr < 7 && ff > 0 && ff < 7 {
            m |= 1u64 << (rr * 8 + ff);
            rr += dr;
            ff += df;
        }
    }
    m
}

fn rook_attacks_slow(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut att = 0u64;
    for &(dr, df) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let mut rr = r + dr;
        let mut ff = f + df;
        while (0..8).contains(&rr) && (0..8).contains(&ff) {
            let bit = 1u64 << (rr * 8 + ff);
            att |= bit;
            if occ & bit != 0 {
                break;
            }
            rr += dr;
            ff += df;
        }
    }
    att
}

fn bishop_attacks_slow(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut att = 0u64;
    for &(dr, df) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let mut rr = r + dr;
        let mut ff = f + df;
        while (0..8).contains(&rr) && (0..8).contains(&ff) {
            let bit = 1u64 << (rr * 8 + ff);
            att |= bit;
            if occ & bit != 0 {
                break;
            }
            rr += dr;
            ff += df;
        }
    }
    att
}

fn find_magics(is_rook: bool) -> ([Magic; 64], Vec<u64>) {
    let mut magics = [Magic {
        mask: 0,
        magic: 0,
        shift: 0,
        offset: 0,
    }; 64];
    let mut table = Vec::with_capacity(if is_rook { 102400 } else { 5248 });
    let mut seed: u64 = if is_rook {
        0x1234_5678_9abc_def0
    } else {
        0xfedc_ba98_7654_3210
    };
    let mut next_rand = || -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        seed.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };

    let mut occs = Vec::new();
    let mut atts = Vec::new();
    let mut used = Vec::new();

    for sq in 0..64 {
        let mask = if is_rook {
            rook_mask(sq)
        } else {
            bishop_mask(sq)
        };
        let bits = mask.count_ones();
        let size = 1usize << bits;
        let shift = 64 - bits;

        // Carry-Rippler enumeration of all subsets of mask.
        occs.clear();
        atts.clear();
        let mut sub = 0u64;
        loop {
            occs.push(sub);
            atts.push(if is_rook {
                rook_attacks_slow(sq, sub)
            } else {
                bishop_attacks_slow(sq, sub)
            });
            sub = sub.wrapping_sub(mask) & mask;
            if sub == 0 {
                break;
            }
        }

        used.resize(size, 0u64);
        let mut epoch_used = vec![0u32; size];
        let mut epoch = 0u32;

        let magic = loop {
            let cand = next_rand() & next_rand() & next_rand();
            if ((mask.wrapping_mul(cand)) & 0xFF00_0000_0000_0000).count_ones() < 6 {
                continue;
            }
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                epoch_used.fill(0);
                epoch = 1;
            }
            let mut ok = true;
            for i in 0..size {
                let idx = (occs[i].wrapping_mul(cand) >> shift) as usize;
                if epoch_used[idx] != epoch {
                    epoch_used[idx] = epoch;
                    used[idx] = atts[i];
                } else if used[idx] != atts[i] {
                    ok = false;
                    break;
                }
            }
            if ok {
                break cand;
            }
        };

        let offset = table.len();
        table.extend_from_slice(&used[..size]);
        magics[sq] = Magic {
            mask,
            magic,
            shift,
            offset,
        };
    }
    (magics, table)
}

fn init_tables() -> Tables {
    let mut knight = [0u64; 64];
    let mut king = [0u64; 64];
    let mut pawn_att = [[0u64; 64]; 2];
    for sq in 0..64 {
        let b = 1u64 << sq;
        knight[sq] = ((b << 17) & NOT_FILE_A)
            | ((b << 15) & NOT_FILE_H)
            | ((b << 10) & NOT_FILE_AB)
            | ((b << 6) & NOT_FILE_GH)
            | ((b >> 17) & NOT_FILE_H)
            | ((b >> 15) & NOT_FILE_A)
            | ((b >> 10) & NOT_FILE_GH)
            | ((b >> 6) & NOT_FILE_AB);
        let lr = ((b << 1) & NOT_FILE_A) | ((b >> 1) & NOT_FILE_H);
        let b_lr = b | lr;
        king[sq] = lr | (b_lr << 8) | (b_lr >> 8);
        pawn_att[0][sq] = ((b << 7) & NOT_FILE_H) | ((b << 9) & NOT_FILE_A);
        pawn_att[1][sq] = ((b >> 9) & NOT_FILE_H) | ((b >> 7) & NOT_FILE_A);
    }

    let (rook_magics, rook_table) = find_magics(true);
    let (bishop_magics, bishop_table) = find_magics(false);

    let mut between = [[0u64; 64]; 64];
    let mut line = [[0u64; 64]; 64];
    for s1 in 0..64 {
        let r1 = rook_attacks_slow(s1, 0);
        let b1 = bishop_attacks_slow(s1, 0);
        for s2 in 0..64 {
            if s1 == s2 {
                continue;
            }
            let b_s1 = 1u64 << s1;
            let b_s2 = 1u64 << s2;
            if r1 & b_s2 != 0 {
                line[s1][s2] = (r1 & rook_attacks_slow(s2, 0)) | b_s1 | b_s2;
                between[s1][s2] = rook_attacks_slow(s1, b_s2) & rook_attacks_slow(s2, b_s1);
            } else if b1 & b_s2 != 0 {
                line[s1][s2] = (b1 & bishop_attacks_slow(s2, 0)) | b_s1 | b_s2;
                between[s1][s2] = bishop_attacks_slow(s1, b_s2) & bishop_attacks_slow(s2, b_s1);
            }
        }
    }

    let mut adj_files = [0u64; 8];
    for f in 0..8 {
        if f > 0 {
            adj_files[f] |= file_bb((f - 1) as u8);
        }
        if f < 7 {
            adj_files[f] |= file_bb((f + 1) as u8);
        }
    }

    let mut passed_mask = [[0u64; 64]; 2];
    for sq in 0..64 {
        let f = sq % 8;
        let r = sq / 8;
        let files = file_bb(f as u8) | adj_files[f];
        let mut w_front = 0u64;
        for rr in (r + 1)..8 {
            w_front |= rank_bb(rr as u8);
        }
        let mut b_front = 0u64;
        for rr in 0..r {
            b_front |= rank_bb(rr as u8);
        }
        passed_mask[0][sq] = files & w_front;
        passed_mask[1][sq] = files & b_front;
    }

    Tables {
        knight,
        king,
        pawn_att,
        between,
        line,
        adj_files,
        passed_mask,
        rook_magics,
        bishop_magics,
        rook_table,
        bishop_table,
    }
}
