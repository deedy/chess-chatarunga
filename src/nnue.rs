//! Extremely optimized SIMD NNUE evaluator with Finny Tables, Pairwise-CReLU (v2), SCReLU (v1),
//! Deep 2-Layer Quantized Head + Threat Inputs (v3), and lazy incremental accumulator updates.
//!
//! Supported CHNN formats:
//! - v1 (SCReLU): `(768 * KB -> HL) x 2 -> OB`, activation `clamp(x, 0, QA)^2`, `out_w` size `OB * 2 * HL`.
//! - v2 (Pairwise-CReLU / GLU): `(768 * KB -> HL) x 2 -> OB`, activation `clamp(x[0..HL/2], 0, QA) * clamp(x[HL/2..HL], 0, QA)`,
//!   `out_w` size `OB * HL`.
//! - v3 (Deep + Threats): `(768 * KB [+ 3*768 threats] -> HL) x 2 -> Pairwise i8 -> 16 (CReLU+SCReLU+skip) -> 32 -> 1`
//!   with `aarch64` `vdotq_s32` (`sdot`) int8 matrix-vector kernels.

use crate::bitboard::{
    bb, bishop_attacks, king_attacks, knight_attacks, pop_lsb, rook_attacks, NOT_FILE_A, NOT_FILE_H,
};
use crate::board::{mb_color, mb_piece, Board, EMPTY_MAILBOX};
use crate::types::{
    sq_file, sq_flip, sq_mirror, sq_rank, Color, Move, Piece, Square, A1, A8, D1, D8, F1, F8, FLAG_CAPTURE,
    FLAG_CASTLE_K, FLAG_CASTLE_Q, FLAG_DOUBLE_PUSH, FLAG_EP, FLAG_PROMO_B, FLAG_PROMO_CAP_B, FLAG_PROMO_CAP_N,
    FLAG_PROMO_CAP_Q, FLAG_PROMO_CAP_R, FLAG_PROMO_N, FLAG_PROMO_Q, FLAG_PROMO_R, FLAG_QUIET, H1, H8,
};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"CHNN";
pub const VERSION_SCRELU: u32 = 1;
pub const VERSION_PAIRWISE: u32 = 2;
pub const VERSION_DEEP_THREATS: u32 = 3;
pub const VERSION_SCRELU_ASYM: u32 = 4;
pub const DEFAULT_QA: i32 = 255;
pub const DEFAULT_QB: i32 = 64;
pub const DEFAULT_SCALE: i32 = 400;

/// Half-board king bucket lookup tables (king on files 0..4 after horizontal mirroring).
/// Indexed by `rank * 4 + file` (0..32).
#[rustfmt::skip]
const BUCKET_4: [usize; 32] = [
    0, 0, 1, 1,
    0, 0, 1, 1,
    2, 2, 2, 2,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];

#[rustfmt::skip]
const BUCKET_8: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 3, 3,
    4, 4, 5, 5,
    4, 4, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    7, 7, 7, 7,
    7, 7, 7, 7,
];

/// Asymmetric 16-bucket layout (CHNN v4): dedicated bucket for EVERY square on ranks 1-2 (where
/// kings spend >92% of chess games), wing/center pairs on ranks 3-8. Same memory & NPS as uniform 16.
#[rustfmt::skip]
const BUCKET_16_ASYM: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

#[inline(always)]
pub fn king_bucket_and_mirror(king_buckets: usize, persp: Color, ksq: Square) -> (usize, bool) {
    king_bucket_and_mirror_v(VERSION_SCRELU, king_buckets, persp, ksq)
}

#[inline(always)]
pub fn king_bucket_and_mirror_v(version: u32, king_buckets: usize, persp: Color, ksq: Square) -> (usize, bool) {
    if king_buckets <= 1 {
        return (0, false);
    }
    let ksq_p = if persp == Color::White { ksq } else { sq_flip(ksq) };
    let mirror = sq_file(ksq_p) >= 4;
    let km = if mirror { sq_mirror(ksq_p) } else { ksq_p };
    let idx32 = (sq_rank(km) as usize) * 4 + (sq_file(km) as usize);
    let b = match king_buckets {
        4 => BUCKET_4[idx32],
        8 => BUCKET_8[idx32],
        16 if version == VERSION_SCRELU_ASYM => BUCKET_16_ASYM[idx32],
        16 => (sq_rank(km) as usize / 2) * 4 + (sq_file(km) as usize),
        _ => idx32 % king_buckets,
    };
    (b, mirror)
}

#[inline(always)]
pub fn feature_index(
    king_buckets: usize,
    persp: Color,
    ksq: Square,
    piece_color: Color,
    piece: Piece,
    sq: Square,
) -> usize {
    feature_index_v(VERSION_SCRELU, king_buckets, persp, ksq, piece_color, piece, sq)
}

#[inline(always)]
pub fn feature_index_v(
    version: u32,
    king_buckets: usize,
    persp: Color,
    ksq: Square,
    piece_color: Color,
    piece: Piece,
    sq: Square,
) -> usize {
    let (kb, mirror) = king_bucket_and_mirror_v(version, king_buckets, persp, ksq);
    let mut sq_p = if persp == Color::White { sq } else { sq_flip(sq) };
    if mirror {
        sq_p = sq_mirror(sq_p);
    }
    let rel_c = if piece_color == persp { 0 } else { 1 };
    kb * 768 + rel_c * 384 + piece.idx() * 64 + sq_p as usize
}

#[inline(always)]
pub fn feature_index_bm(kb: usize, mirror: bool, persp: Color, piece_color: Color, piece: Piece, sq: Square) -> usize {
    let mut sq_p = if persp == Color::White { sq } else { sq_flip(sq) };
    if mirror {
        sq_p = sq_mirror(sq_p);
    }
    let rel_c = if piece_color == persp { 0 } else { 1 };
    kb * 768 + rel_c * 384 + piece.idx() * 64 + sq_p as usize
}

#[inline(always)]
pub fn output_bucket(b: &Board, output_buckets: usize) -> usize {
    if output_buckets <= 1 {
        return 0;
    }
    let count = b.occ().count_ones() as usize;
    let divisor = 32usize.div_ceil(output_buckets);
    (count.saturating_sub(2) / divisor).min(output_buckets - 1)
}

/// Compute layered attack bitboards `(att_pawn, att_minor, att_rook, att_all)` for both colors
/// (exact mirror of the CUDA v3 threat extractor).
#[inline]
pub fn board_attack_maps(b: &Board) -> ([u64; 2], [u64; 2], [u64; 2], [u64; 2]) {
    let occ = b.occ();
    let mut att_pawn = [0u64; 2];
    let mut att_minor = [0u64; 2];
    let mut att_rook = [0u64; 2];
    let mut att_all = [0u64; 2];

    for c_idx in 0..2 {
        let c = b.colors[c_idx];
        let pawns = c & b.pieces[Piece::Pawn.idx()];
        let pa = if c_idx == 0 {
            ((pawns & NOT_FILE_A) << 7) | ((pawns & NOT_FILE_H) << 9)
        } else {
            ((pawns & NOT_FILE_A) >> 9) | ((pawns & NOT_FILE_H) >> 7)
        };
        att_pawn[c_idx] = pa;

        let mut minor = pa;
        let mut knights = c & b.pieces[Piece::Knight.idx()];
        while knights != 0 {
            minor |= knight_attacks(pop_lsb(&mut knights));
        }
        let mut bishops = c & b.pieces[Piece::Bishop.idx()];
        while bishops != 0 {
            minor |= bishop_attacks(pop_lsb(&mut bishops), occ);
        }
        att_minor[c_idx] = minor;

        let mut rook = minor;
        let mut rooks = c & b.pieces[Piece::Rook.idx()];
        while rooks != 0 {
            rook |= rook_attacks(pop_lsb(&mut rooks), occ);
        }
        att_rook[c_idx] = rook;

        let mut all = rook;
        let mut queens = c & b.pieces[Piece::Queen.idx()];
        while queens != 0 {
            let sq = pop_lsb(&mut queens);
            all |= bishop_attacks(sq, occ) | rook_attacks(sq, occ);
        }
        let kbb = c & b.pieces[Piece::King.idx()];
        if kbb != 0 {
            all |= king_attacks(kbb.trailing_zeros() as Square);
        }
        att_all[c_idx] = all;
    }

    (att_pawn, att_minor, att_rook, att_all)
}

/// Quantized NNUE network weights (supports v1 SCReLU, v2 Pairwise-CReLU, and v3 Deep+Threats).
#[derive(Clone, Debug)]
pub struct NnueNet {
    pub version: u32,
    pub king_buckets: usize,
    pub hl: usize,
    pub output_buckets: usize,
    pub qa: i32,
    pub qb: i32,
    pub scale: i32,
    pub has_threats: bool,
    pub l1_size: usize,
    pub l2_size: usize,
    /// Feature transformer weights: `[(king_buckets + 3*has_threats) * 768 * hl]`, row-major per feature.
    pub ft_w: Vec<i16>,
    /// Feature transformer biases: `[hl]`.
    pub ft_b: Vec<i16>,
    /// v1/v2 output weights: v1 = `[ob * 2 * hl]`, v2 = `[ob * hl]`, v3 = empty.
    pub out_w: Vec<i16>,
    /// Output biases: `[output_buckets]` (v1/v2: scaled by `qa * qb`; v3: scaled by `127 * qb`).
    pub out_b: Vec<i32>,
    /// v3 Layer 1 weights: `[ob * l1_size * hl]` (i8, scale `qb`).
    pub l1_w: Vec<i8>,
    /// v3 Layer 1 weights pre-packed for 4-byte NNZ `sdot`: `[ob][hl/4][16][4]` (64 bytes per chunk).
    pub l1_w_t: Vec<i8>,
    /// v3 Layer 1 biases: `[ob * l1_size]` (i32, scale `127 * qb`).
    pub l1_b: Vec<i32>,
    /// v3 Layer 2 weights: `[ob * l2_size * (2 * l1_size)]` (i8, scale `qb`).
    pub l2_w: Vec<i8>,
    /// v3 Layer 2 biases: `[ob * l2_size]` (i32, scale `127 * qb`).
    pub l2_b: Vec<i32>,
    /// v3 Output layer weights: `[ob * l2_size]` (i8, scale `qb`).
    pub out_w_i8: Vec<i8>,
}

/// Pack `l1_w` (`[ob][16][hl]`) into `[ob][hl/4][16][4]` for single-cache-line NNZ `sdot`.
pub fn pack_l1_w_transposed(l1_w: &[i8], ob: usize, hl: usize) -> Vec<i8> {
    let chunks = hl / 4;
    let mut out = vec![0i8; ob * chunks * 16 * 4];
    for b in 0..ob {
        for c in 0..chunks {
            let dst_base = (b * chunks + c) * 64;
            for j in 0..16 {
                let src_base = (b * 16 + j) * hl + c * 4;
                out[dst_base + j * 4..dst_base + j * 4 + 4].copy_from_slice(&l1_w[src_base..src_base + 4]);
            }
        }
    }
    out
}

impl NnueNet {
    #[inline(always)]
    pub fn is_pairwise(&self) -> bool {
        self.version == VERSION_PAIRWISE
    }

    #[inline(always)]
    pub fn is_deep(&self) -> bool {
        self.version == VERSION_DEEP_THREATS
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&(self.king_buckets as u32).to_le_bytes())?;
        w.write_all(&(self.hl as u32).to_le_bytes())?;
        w.write_all(&(self.output_buckets as u32).to_le_bytes())?;
        w.write_all(&self.qa.to_le_bytes())?;
        w.write_all(&self.qb.to_le_bytes())?;
        w.write_all(&self.scale.to_le_bytes())?;
        if self.version == VERSION_DEEP_THREATS {
            let flags: u32 = u32::from(self.has_threats) | ((self.l1_size as u32) << 8) | ((self.l2_size as u32) << 16);
            w.write_all(&flags.to_le_bytes())?;
        }
        for &v in &self.ft_w {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.ft_b {
            w.write_all(&v.to_le_bytes())?;
        }
        if self.version == VERSION_DEEP_THREATS {
            for &v in &self.l1_w {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.l1_b {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.l2_w {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.l2_b {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.out_w_i8 {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.out_b {
                w.write_all(&v.to_le_bytes())?;
            }
        } else {
            for &v in &self.out_w {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.out_b {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.flush()
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let f = File::open(path.as_ref()).map_err(|e| format!("open {}: {e}", path.as_ref().display()))?;
        let mut r = BufReader::new(f);
        Self::from_reader(&mut r)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cur = std::io::Cursor::new(bytes);
        Self::from_reader(&mut cur)
    }

    pub fn from_reader<R: Read>(r: &mut R) -> Result<Self, String> {
        let mut hdr = [0u8; 32];
        r.read_exact(&mut hdr).map_err(|e| format!("read header: {e}"))?;
        if &hdr[0..4] != MAGIC {
            return Err("invalid NNUE magic".to_string());
        }
        let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if version != VERSION_SCRELU
            && version != VERSION_PAIRWISE
            && version != VERSION_DEEP_THREATS
            && version != VERSION_SCRELU_ASYM
        {
            return Err(format!("unsupported NNUE version {version}"));
        }
        let king_buckets = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let hl = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        let output_buckets = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
        let qa = i32::from_le_bytes(hdr[20..24].try_into().unwrap());
        let qb = i32::from_le_bytes(hdr[24..28].try_into().unwrap());
        let scale = i32::from_le_bytes(hdr[28..32].try_into().unwrap());

        if version == VERSION_DEEP_THREATS {
            let mut fbuf = [0u8; 4];
            r.read_exact(&mut fbuf).map_err(|e| format!("read v3 flags: {e}"))?;
            let flags = u32::from_le_bytes(fbuf);
            let has_threats = (flags & 1) != 0;
            let l1_size = ((flags >> 8) & 0xFF) as usize;
            let l2_size = ((flags >> 16) & 0xFF) as usize;
            if l1_size != 16 || l2_size != 32 {
                return Err(format!("unsupported v3 head dims l1={l1_size} l2={l2_size} (expected 16, 32)"));
            }
            let n_ft_rows = (king_buckets + if has_threats { 3 } else { 0 }) * 768;
            let ft_w = read_i16_vec(r, n_ft_rows * hl)?;
            let ft_b = read_i16_vec(r, hl)?;
            let l1_w = read_i8_vec(r, output_buckets * l1_size * hl)?;
            let l1_w_t = pack_l1_w_transposed(&l1_w, output_buckets, hl);
            let l1_b = read_i32_vec(r, output_buckets * l1_size)?;
            let l2_w = read_i8_vec(r, output_buckets * l2_size * (2 * l1_size))?;
            let l2_b = read_i32_vec(r, output_buckets * l2_size)?;
            let out_w_i8 = read_i8_vec(r, output_buckets * l2_size)?;
            let out_b = read_i32_vec(r, output_buckets)?;

            return Ok(Self {
                version,
                king_buckets,
                hl,
                output_buckets,
                qa,
                qb,
                scale,
                has_threats,
                l1_size,
                l2_size,
                ft_w,
                ft_b,
                out_w: Vec::new(),
                out_b,
                l1_w,
                l1_w_t,
                l1_b,
                l2_w,
                l2_b,
                out_w_i8,
            });
        }

        let n_ft_w = king_buckets * 768 * hl;
        let n_out_w = if version == VERSION_PAIRWISE { output_buckets * hl } else { output_buckets * 2 * hl };
        let ft_w = read_i16_vec(r, n_ft_w)?;
        let ft_b = read_i16_vec(r, hl)?;
        let out_w = read_i16_vec(r, n_out_w)?;
        let out_b = read_i32_vec(r, output_buckets)?;

        Ok(Self {
            version,
            king_buckets,
            hl,
            output_buckets,
            qa,
            qb,
            scale,
            has_threats: false,
            l1_size: 0,
            l2_size: 0,
            ft_w,
            ft_b,
            out_w,
            out_b,
            l1_w: Vec::new(),
            l1_w_t: Vec::new(),
            l1_b: Vec::new(),
            l2_w: Vec::new(),
            l2_b: Vec::new(),
            out_w_i8: Vec::new(),
        })
    }

    #[inline(always)]
    pub fn ft_row(&self, feat: usize) -> &[i16] {
        let off = feat * self.hl;
        unsafe { self.ft_w.get_unchecked(off..off + self.hl) }
    }

    /// Full refresh of one perspective's base piece-square accumulator from scratch (without threats).
    pub fn refresh_perspective(&self, b: &Board, persp: Color, acc: &mut [i16]) {
        acc.copy_from_slice(&self.ft_b);
        let ksq = b.king_sq(persp);
        let mut occ = b.occ();
        while occ != 0 {
            let sq = pop_lsb(&mut occ);
            let v = b.mailbox[sq as usize];
            debug_assert!(v != EMPTY_MAILBOX);
            let feat = feature_index_v(self.version, self.king_buckets, persp, ksq, mb_color(v), mb_piece(v), sq);
            vec_add_assign(acc, self.ft_row(feat));
        }
    }

    /// Add active v3 threat features for board `b` onto `w_acc` and `b_acc` (both perspectives).
    /// Returns `true` if any threat feature was added.
    #[inline]
    pub fn add_threat_features(&self, b: &Board, w_acc: &mut [i16], b_acc: &mut [i16]) {
        let (att_pawn, att_minor, att_rook, att_all) = board_attack_maps(b);
        let mut threatened = (b.colors[0] & att_all[1]) | (b.colors[1] & att_all[0]);
        if threatened == 0 {
            return;
        }
        let (_, mir_w) = king_bucket_and_mirror(self.king_buckets, Color::White, b.king_sq(Color::White));
        let (_, mir_b) = king_bucket_and_mirror(self.king_buckets, Color::Black, b.king_sq(Color::Black));
        let threat_off = self.king_buckets * 768;

        while threatened != 0 {
            let sq = pop_lsb(&mut threatened);
            let v = b.mailbox[sq as usize];
            let col = mb_color(v);
            let piece = mb_piece(v);
            let ci = col.idx();
            let enemy = ci ^ 1;
            let bit = bb(sq);
            let defended = usize::from((att_all[ci] & bit) != 0);

            let sq_w = if mir_w { sq_mirror(sq) } else { sq } as usize;
            let sq_b_raw = sq_flip(sq);
            let sq_b = if mir_b { sq_mirror(sq_b_raw) } else { sq_b_raw } as usize;

            let fw_base = ci * 384 + piece.idx() * 64 + sq_w;
            let fb_base = enemy * 384 + piece.idx() * 64 + sq_b;

            let tbase = threat_off + defended * 768;
            vec_add_assign(w_acc, self.ft_row(tbase + fw_base));
            vec_add_assign(b_acc, self.ft_row(tbase + fb_base));

            let lesser = match piece {
                Piece::Knight | Piece::Bishop => (att_pawn[enemy] & bit) != 0,
                Piece::Rook => (att_minor[enemy] & bit) != 0,
                Piece::Queen => (att_rook[enemy] & bit) != 0,
                _ => false,
            };
            if lesser {
                let lbase = threat_off + 2 * 768;
                vec_add_assign(w_acc, self.ft_row(lbase + fw_base));
                vec_add_assign(b_acc, self.ft_row(lbase + fb_base));
            }
        }
    }

    /// Evaluate directly from scratch (no incremental accumulator).
    pub fn evaluate_board(&self, b: &Board) -> i32 {
        let mut w_acc = vec![0i16; self.hl];
        let mut b_acc = vec![0i16; self.hl];
        self.refresh_perspective(b, Color::White, &mut w_acc);
        self.refresh_perspective(b, Color::Black, &mut b_acc);
        if self.has_threats {
            self.add_threat_features(b, &mut w_acc, &mut b_acc);
        }
        let ob = output_bucket(b, self.output_buckets);
        if b.side == Color::White {
            self.eval_from_accs(&w_acc, &b_acc, ob)
        } else {
            self.eval_from_accs(&b_acc, &w_acc, ob)
        }
    }

    /// Evaluate from materialized `[stm_acc, opp_acc]` (with threats already applied if v3) and output bucket using SIMD.
    #[inline]
    pub fn eval_from_accs(&self, stm_acc: &[i16], opp_acc: &[i16], ob: usize) -> i32 {
        if self.version == VERSION_DEEP_THREATS {
            return self.eval_v3_simd(stm_acc, opp_acc, ob);
        }
        let hl = self.hl;
        let sum = if self.version == VERSION_PAIRWISE {
            let half = hl / 2;
            let base = ob * hl;
            let w_stm = &self.out_w[base..base + half];
            let w_opp = &self.out_w[base + half..base + hl];
            pairwise_dot_simd(stm_acc, w_stm, self.qa as i16) + pairwise_dot_simd(opp_acc, w_opp, self.qa as i16)
        } else {
            let base = ob * 2 * hl;
            let w_stm = &self.out_w[base..base + hl];
            let w_opp = &self.out_w[base + hl..base + 2 * hl];
            screlu_dot_simd(stm_acc, w_stm, self.qa as i16) + screlu_dot_simd(opp_acc, w_opp, self.qa as i16)
        };
        let total = sum / self.qa as i64 + self.out_b[ob] as i64;
        ((total * self.scale as i64) / (self.qa as i64 * self.qb as i64)) as i32
    }

    /// Scalar reference implementation (for bit-exactness unit tests across v1, v2, and v3).
    pub fn eval_from_accs_scalar(&self, stm_acc: &[i16], opp_acc: &[i16], ob: usize) -> i32 {
        if self.version == VERSION_DEEP_THREATS {
            return self.eval_v3_scalar(stm_acc, opp_acc, ob);
        }
        let hl = self.hl;
        let sum = if self.version == VERSION_PAIRWISE {
            let half = hl / 2;
            let base = ob * hl;
            let w_stm = &self.out_w[base..base + half];
            let w_opp = &self.out_w[base + half..base + hl];
            pairwise_dot_scalar(stm_acc, w_stm, self.qa as i16) + pairwise_dot_scalar(opp_acc, w_opp, self.qa as i16)
        } else {
            let base = ob * 2 * hl;
            let w_stm = &self.out_w[base..base + hl];
            let w_opp = &self.out_w[base + hl..base + 2 * hl];
            screlu_dot_scalar(stm_acc, w_stm, self.qa as i16) + screlu_dot_scalar(opp_acc, w_opp, self.qa as i16)
        };
        let total = sum / self.qa as i64 + self.out_b[ob] as i64;
        ((total * self.scale as i64) / (self.qa as i64 * self.qb as i64)) as i32
    }

    /// CHNN v3 deep head SIMD forward pass (`HL -> 16 (dual+skip) -> 32 -> 1` via NNZ-transposed `sdot`).
    /// Assumes threat features (if any) are already incrementally maintained inside `stm_acc` / `opp_acc`.
    #[inline]
    fn eval_v3_simd(&self, stm_acc: &[i16], opp_acc: &[i16], ob: usize) -> i32 {
        let hl = self.hl;
        let half = hl / 2;
        debug_assert!(hl <= 2048 && hl % 32 == 0);
        #[repr(align(16))]
        struct AlignedI8([std::mem::MaybeUninit<i8>; 2048]);
        let mut ft_buf = AlignedI8([std::mem::MaybeUninit::uninit(); 2048]);
        let ft_slice = unsafe { std::slice::from_raw_parts_mut(ft_buf.0.as_mut_ptr() as *mut i8, hl) };
        pairwise_quantize_simd(stm_acc, &mut ft_slice[..half], self.qa as i16);
        pairwise_quantize_simd(opp_acc, &mut ft_slice[half..hl], self.qa as i16);
        self.eval_v3_from_ft_q(ft_slice, ob)
    }

    #[inline]
    fn eval_v3_from_ft_q(&self, ft_slice: &[i8], ob: usize) -> i32 {
        let hl = self.hl;
        let l1_b_off = ob * 16;
        let mut l1_pre = [0i32; 16];
        if !self.l1_w_t.is_empty() {
            l1_matmul_nnz_simd(
                ft_slice,
                &self.l1_w_t[ob * 16 * hl..(ob + 1) * 16 * hl],
                &self.l1_b[l1_b_off..l1_b_off + 16],
                &mut l1_pre,
            );
        } else {
            let l1_w_off = ob * 16 * hl;
            for j in 0..16 {
                let row = &self.l1_w[l1_w_off + j * hl..l1_w_off + (j + 1) * hl];
                l1_pre[j] = dot_i8_simd(ft_slice, row) + self.l1_b[l1_b_off + j];
            }
        }

        let mut l1_act = [0i8; 32];
        for j in 0..16 {
            let c = (l1_pre[j] >> 6).clamp(0, 127);
            l1_act[j] = c as i8;
            l1_act[16 + j] = ((c * c) >> 7).clamp(0, 127) as i8;
        }

        // Layer 2 (32 -> 32) via fused 4-row-at-a-time ARM64 sdot
        let l2_w_off = ob * 32 * 32;
        let l2_b_off = ob * 32;
        let mut l2_act = [0i8; 32];
        l2_matmul_32x32_simd(
            &l1_act,
            &self.l2_w[l2_w_off..l2_w_off + 1024],
            &self.l2_b[l2_b_off..l2_b_off + 32],
            &mut l2_act,
        );

        // Output: 32 -> 1 + direct skip from l1_pre[0]
        let out_w = &self.out_w_i8[ob * 32..(ob + 1) * 32];
        let out_raw = dot_i8_simd(&l2_act, out_w) + self.out_b[ob] + l1_pre[0];
        ((out_raw as i64 * self.scale as i64) / (127 * self.qb as i64)) as i32
    }

    /// Scalar reference for CHNN v3 (guaranteed bit-exact match to `eval_v3_simd`).
    fn eval_v3_scalar(&self, stm_acc: &[i16], opp_acc: &[i16], ob: usize) -> i32 {
        let hl = self.hl;
        let half = hl / 2;
        let qa = self.qa as i16;
        let mut ft_q = vec![0i8; hl];
        for i in 0..half {
            let a = stm_acc[i].clamp(0, qa) as u32;
            let b = stm_acc[i + half].clamp(0, qa) as u32;
            ft_q[i] = ((a * b) >> 9) as i8;
            let oa = opp_acc[i].clamp(0, qa) as u32;
            let ob_v = opp_acc[i + half].clamp(0, qa) as u32;
            ft_q[half + i] = ((oa * ob_v) >> 9) as i8;
        }

        let l1_w_off = ob * 16 * hl;
        let l1_b_off = ob * 16;
        let mut l1_pre = [0i32; 16];
        let mut l1_act = [0i8; 32];
        for j in 0..16 {
            let row = &self.l1_w[l1_w_off + j * hl..l1_w_off + (j + 1) * hl];
            let mut sum = self.l1_b[l1_b_off + j];
            for (&x, &w) in ft_q.iter().zip(row.iter()) {
                sum += (x as i32) * (w as i32);
            }
            l1_pre[j] = sum;
            let c = (sum >> 6).clamp(0, 127);
            l1_act[j] = c as i8;
            l1_act[16 + j] = ((c * c) >> 7).clamp(0, 127) as i8;
        }

        let l2_w_off = ob * 32 * 32;
        let l2_b_off = ob * 32;
        let mut l2_act = [0i8; 32];
        for k in 0..32 {
            let row = &self.l2_w[l2_w_off + k * 32..l2_w_off + (k + 1) * 32];
            let mut sum = self.l2_b[l2_b_off + k];
            for (&x, &w) in l1_act.iter().zip(row.iter()) {
                sum += (x as i32) * (w as i32);
            }
            l2_act[k] = (sum >> 6).clamp(0, 127) as i8;
        }

        let out_w = &self.out_w_i8[ob * 32..(ob + 1) * 32];
        let mut out_raw = self.out_b[ob] + l1_pre[0];
        for (&x, &w) in l2_act.iter().zip(out_w.iter()) {
            out_raw += (x as i32) * (w as i32);
        }
        ((out_raw as i64 * self.scale as i64) / (127 * self.qb as i64)) as i32
    }
}

fn read_i8_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<i8>, String> {
    let mut bytes = vec![0u8; n];
    r.read_exact(&mut bytes).map_err(|e| format!("read i8[{n}]: {e}"))?;
    Ok(bytes.into_iter().map(|b| b as i8).collect())
}

fn read_i16_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<i16>, String> {
    let mut bytes = vec![0u8; n * 2];
    r.read_exact(&mut bytes).map_err(|e| format!("read i16[{n}]: {e}"))?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

fn read_i32_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<i32>, String> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes).map_err(|e| format!("read i32[{n}]: {e}"))?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Finny Table entry: caches a previously materialized accumulator and its piece bitboards
/// for a `(perspective, king_bucket, mirror)` slot, enabling differential refreshes.
struct FinnyEntry {
    pieces: [u64; 6],
    colors: [u64; 2],
    acc: Vec<i16>,
}

/// Compact move delta describing up to 2 feature additions and 2 feature removals per move.
#[derive(Clone, Copy, Default)]
pub struct MoveDelta {
    pub moved_color: u8,
    pub moved_piece: u8,
    pub cap_piece: u8, // 255 = none
    pub m: Move,
    pub ksq: [Square; 2],
    pub prev_ksq: [Square; 2],
}

/// Cached per-ply threat codes for incremental CHNN v3 threat accumulator updates.
/// Each byte encodes: 0 = not threatened, otherwise `(mb_pack << 3) | (lesser << 2) | (defended << 1) | 1`.
#[derive(Clone, Copy)]
struct ThreatState {
    computed: bool,
    mask: u64,
    codes: [u8; 64],
}

impl Default for ThreatState {
    fn default() -> Self {
        Self { computed: false, mask: 0, codes: [0; 64] }
    }
}

#[inline]
fn compute_threat_state(b: &Board) -> ThreatState {
    let (att_pawn, att_minor, att_rook, att_all) = board_attack_maps(b);
    let mask = (b.colors[0] & att_all[1]) | (b.colors[1] & att_all[0]);
    let mut codes = [0u8; 64];
    let mut m = mask;
    while m != 0 {
        let sq = pop_lsb(&mut m);
        let v = b.mailbox[sq as usize];
        let col = mb_color(v);
        let piece = mb_piece(v);
        let ci = col.idx();
        let enemy = ci ^ 1;
        let bit = bb(sq);
        let defended = u8::from((att_all[ci] & bit) != 0);
        let lesser = u8::from(match piece {
            Piece::Knight | Piece::Bishop => (att_pawn[enemy] & bit) != 0,
            Piece::Rook => (att_minor[enemy] & bit) != 0,
            Piece::Queen => (att_rook[enemy] & bit) != 0,
            _ => false,
        });
        codes[sq as usize] = (v << 3) | (lesser << 2) | (defended << 1) | 1;
    }
    ThreatState { computed: true, mask, codes }
}

#[inline(always)]
fn apply_threat_code(net: &NnueNet, acc: &mut [i16], persp: Color, mirror: bool, sq: Square, code: u8, add: bool) {
    let v = code >> 3;
    let col = mb_color(v);
    let piece = mb_piece(v);
    let rel_c = usize::from(col != persp);
    let sq_p_raw = if persp == Color::White { sq } else { sq_flip(sq) };
    let sq_p = if mirror { sq_mirror(sq_p_raw) } else { sq_p_raw } as usize;
    let f_base = rel_c * 384 + piece.idx() * 64 + sq_p;
    let threat_off = net.king_buckets * 768;
    let defended = usize::from((code & 2) != 0);
    let f1 = threat_off + defended * 768 + f_base;
    if add {
        vec_add_assign(acc, net.ft_row(f1));
        if (code & 4) != 0 {
            vec_add_assign(acc, net.ft_row(threat_off + 2 * 768 + f_base));
        }
    } else {
        vec_sub_assign(acc, net.ft_row(f1));
        if (code & 4) != 0 {
            vec_sub_assign(acc, net.ft_row(threat_off + 2 * 768 + f_base));
        }
    }
}

/// Lazy incremental accumulator stack with Finny Table cache and incremental v3 threat maintenance.
pub struct AccStack {
    hl: usize,
    max_ply: usize,
    buf: Vec<i16>,
    computed: Vec<[bool; 2]>,
    deltas: Vec<MoveDelta>,
    boards: Vec<Board>,
    threat_states: Vec<ThreatState>,
    finny: Vec<FinnyEntry>, // indexed by ((persp * king_buckets + kb) * 2 + mirror)
    finny_kb: usize,
}

impl AccStack {
    pub fn new(hl: usize, max_ply: usize) -> Self {
        Self {
            hl,
            max_ply,
            buf: vec![0i16; max_ply * 2 * hl],
            computed: vec![[false; 2]; max_ply],
            deltas: vec![MoveDelta::default(); max_ply],
            boards: vec![Board::empty(); max_ply],
            threat_states: vec![ThreatState::default(); max_ply],
            finny: Vec::new(),
            finny_kb: 0,
        }
    }

    fn ensure_finny(&mut self, net: &NnueNet) {
        if self.finny_kb == net.king_buckets && !self.finny.is_empty() {
            return;
        }
        self.finny_kb = net.king_buckets;
        let slots = 2 * net.king_buckets * 2;
        self.finny.clear();
        self.finny.reserve(slots);
        for _ in 0..slots {
            self.finny.push(FinnyEntry { pieces: [0; 6], colors: [0; 2], acc: net.ft_b.clone() });
        }
    }

    #[inline(always)]
    fn slice(&self, ply: usize, c: usize) -> &[i16] {
        let off = (ply * 2 + c) * self.hl;
        unsafe { self.buf.get_unchecked(off..off + self.hl) }
    }

    #[inline(always)]
    fn slice_mut(&mut self, ply: usize, c: usize) -> &mut [i16] {
        let off = (ply * 2 + c) * self.hl;
        unsafe { self.buf.get_unchecked_mut(off..off + self.hl) }
    }

    #[inline(always)]
    fn split_prev_curr(&mut self, ply: usize, c: usize) -> (&[i16], &mut [i16]) {
        let prev_off = ((ply - 1) * 2 + c) * self.hl;
        let curr_off = (ply * 2 + c) * self.hl;
        let (left, right) = self.buf.split_at_mut(curr_off);
        (&left[prev_off..prev_off + self.hl], &mut right[..self.hl])
    }

    /// Initialize ply 0 from the root board (using Finny table differential refresh!).
    pub fn init_root(&mut self, net: &NnueNet, b: &Board) {
        self.ensure_finny(net);
        self.boards[0] = *b;
        self.threat_states[0].computed = false;
        self.finny_refresh(net, 0, Color::White);
        self.finny_refresh(net, 0, Color::Black);
        self.computed[0] = [true, true];
    }

    /// Differential refresh of `(ply, persp)` using the Finny Table cache (plus active threats if v3).
    fn finny_refresh(&mut self, net: &NnueNet, ply: usize, persp: Color) {
        let b = self.boards[ply];
        let ksq = b.king_sq(persp);
        let (kb, mirror) = king_bucket_and_mirror_v(net.version, net.king_buckets, persp, ksq);
        let slot = (persp.idx() * net.king_buckets + kb) * 2 + usize::from(mirror);
        let entry = &mut self.finny[slot];

        for c_idx in 0..2 {
            let color = if c_idx == 0 { Color::White } else { Color::Black };
            for p_idx in 0..6 {
                let piece = Piece::from_u8(p_idx as u8);
                let prev_bb = entry.colors[c_idx] & entry.pieces[p_idx];
                let curr_bb = b.colors[c_idx] & b.pieces[p_idx];
                let mut added = curr_bb & !prev_bb;
                let mut removed = prev_bb & !curr_bb;
                while added != 0 {
                    let sq = pop_lsb(&mut added);
                    let f = feature_index_bm(kb, mirror, persp, color, piece, sq);
                    vec_add_assign(&mut entry.acc, net.ft_row(f));
                }
                while removed != 0 {
                    let sq = pop_lsb(&mut removed);
                    let f = feature_index_bm(kb, mirror, persp, color, piece, sq);
                    vec_sub_assign(&mut entry.acc, net.ft_row(f));
                }
            }
        }
        entry.pieces = b.pieces;
        entry.colors = b.colors;

        let off = (ply * 2 + persp.idx()) * self.hl;
        self.buf[off..off + self.hl].copy_from_slice(&entry.acc);

        if net.has_threats {
            if !self.threat_states[ply].computed {
                self.threat_states[ply] = compute_threat_state(&b);
            }
            let ts = self.threat_states[ply];
            let acc = &mut self.buf[off..off + self.hl];
            let mut m = ts.mask;
            while m != 0 {
                let sq = pop_lsb(&mut m);
                apply_threat_code(net, acc, persp, mirror, sq, ts.codes[sq as usize], true);
            }
        }
    }

    #[inline(always)]
    pub fn push_move(&mut self, ply: usize, prev: &Board, next: &Board, m: Move) {
        let nxt = ply + 1;
        debug_assert!(nxt < self.max_ply);
        let mv = prev.mailbox[m.from() as usize];
        let cap_piece = if m.is_ep() {
            Piece::Pawn as u8
        } else if m.is_capture() {
            mb_piece(prev.mailbox[m.to() as usize]) as u8
        } else {
            255
        };
        self.deltas[nxt] = MoveDelta {
            moved_color: prev.side as u8,
            moved_piece: mb_piece(mv) as u8,
            cap_piece,
            m,
            ksq: [next.king_sq(Color::White), next.king_sq(Color::Black)],
            prev_ksq: [prev.king_sq(Color::White), prev.king_sq(Color::Black)],
        };
        self.boards[nxt] = *next;
        self.computed[nxt] = [false, false];
        self.threat_states[nxt].computed = false;
    }

    #[inline(always)]
    pub fn push_null(&mut self, ply: usize, next: &Board) {
        let nxt = ply + 1;
        debug_assert!(nxt < self.max_ply);
        self.deltas[nxt] = MoveDelta {
            moved_color: 0,
            moved_piece: 0,
            cap_piece: 255,
            m: Move::NONE,
            ksq: [next.king_sq(Color::White), next.king_sq(Color::Black)],
            prev_ksq: [next.king_sq(Color::White), next.king_sq(Color::Black)],
        };
        self.boards[nxt] = *next;
        self.computed[nxt] = [false, false];
        self.threat_states[nxt].computed = false;
    }

    #[inline]
    pub fn evaluate(&mut self, net: &NnueNet, ply: usize, b: &Board) -> i32 {
        self.ensure_computed(net, ply, Color::White);
        self.ensure_computed(net, ply, Color::Black);
        let ob = output_bucket(b, net.output_buckets);
        let w = self.slice(ply, 0);
        let bl = self.slice(ply, 1);
        if b.side == Color::White {
            net.eval_from_accs(w, bl, ob)
        } else {
            net.eval_from_accs(bl, w, ob)
        }
    }

    fn ensure_computed(&mut self, net: &NnueNet, ply: usize, persp: Color) {
        let ci = persp.idx();
        if self.computed[ply][ci] {
            return;
        }
        let mut cur = ply;
        let mut needs_refresh = false;
        while cur > 0 && !self.computed[cur][ci] {
            let d = self.deltas[cur];
            if !d.m.is_none()
                && net.king_buckets > 1
                && d.moved_color as usize == ci
                && d.moved_piece == Piece::King as u8
            {
                let b_prev = king_bucket_and_mirror_v(net.version, net.king_buckets, persp, d.prev_ksq[ci]);
                let b_next = king_bucket_and_mirror_v(net.version, net.king_buckets, persp, d.ksq[ci]);
                if b_prev != b_next {
                    needs_refresh = true;
                    break;
                }
            }
            cur -= 1;
        }

        if needs_refresh {
            self.finny_refresh(net, ply, persp);
            self.computed[ply][ci] = true;
            return;
        }

        for p in (cur + 1)..=ply {
            let d = self.deltas[p];
            if d.m.is_none() {
                let (src, dst) = self.split_prev_curr(p, ci);
                dst.copy_from_slice(src);
                if net.has_threats && !self.threat_states[p].computed {
                    self.threat_states[p] = self.threat_states[p - 1];
                }
            } else {
                self.apply_delta(net, p, persp, d);
                if net.has_threats {
                    if !self.threat_states[p].computed {
                        self.threat_states[p] = compute_threat_state(&self.boards[p]);
                    }
                    let prev_ts = self.threat_states[p - 1];
                    let curr_ts = self.threat_states[p];
                    let mut diff = prev_ts.mask | curr_ts.mask;
                    if diff != 0 {
                        let (_, mirror) = king_bucket_and_mirror(net.king_buckets, persp, d.ksq[ci]);
                        let acc = self.slice_mut(p, ci);
                        while diff != 0 {
                            let sq = pop_lsb(&mut diff);
                            let oc = prev_ts.codes[sq as usize];
                            let nc = curr_ts.codes[sq as usize];
                            if oc == nc {
                                continue;
                            }
                            if oc != 0 {
                                apply_threat_code(net, acc, persp, mirror, sq, oc, false);
                            }
                            if nc != 0 {
                                apply_threat_code(net, acc, persp, mirror, sq, nc, true);
                            }
                        }
                    }
                }
            }
            self.computed[p][ci] = true;
        }
    }

    fn apply_delta(&mut self, net: &NnueNet, ply: usize, persp: Color, d: MoveDelta) {
        let ci = persp.idx();
        let ksq = d.ksq[ci];
        let us = if d.moved_color == 0 { Color::White } else { Color::Black };
        let them = us.flip();
        let moved_p = Piece::from_u8(d.moved_piece);
        let from = d.m.from();
        let to = d.m.to();
        let flag = d.m.flag();
        let ver = net.version;
        let kb = net.king_buckets;

        let (src, dst) = self.split_prev_curr(ply, ci);
        match flag {
            FLAG_QUIET | FLAG_DOUBLE_PUSH => {
                let sub = feature_index_v(ver, kb, persp, ksq, us, moved_p, from);
                let add = feature_index_v(ver, kb, persp, ksq, us, moved_p, to);
                vec_add1_sub1(dst, src, net.ft_row(add), net.ft_row(sub));
            }
            FLAG_CAPTURE => {
                let cap_p = Piece::from_u8(d.cap_piece);
                let sub1 = feature_index_v(ver, kb, persp, ksq, us, moved_p, from);
                let sub2 = feature_index_v(ver, kb, persp, ksq, them, cap_p, to);
                let add1 = feature_index_v(ver, kb, persp, ksq, us, moved_p, to);
                vec_add1_sub2(dst, src, net.ft_row(add1), net.ft_row(sub1), net.ft_row(sub2));
            }
            FLAG_EP => {
                let cap_sq = if us == Color::White { to - 8 } else { to + 8 };
                let sub1 = feature_index_v(ver, kb, persp, ksq, us, Piece::Pawn, from);
                let sub2 = feature_index_v(ver, kb, persp, ksq, them, Piece::Pawn, cap_sq);
                let add1 = feature_index_v(ver, kb, persp, ksq, us, Piece::Pawn, to);
                vec_add1_sub2(dst, src, net.ft_row(add1), net.ft_row(sub1), net.ft_row(sub2));
            }
            FLAG_CASTLE_K => {
                let (r_from, r_to) = if us == Color::White { (H1, F1) } else { (H8, F8) };
                let sub1 = feature_index_v(ver, kb, persp, ksq, us, Piece::King, from);
                let sub2 = feature_index_v(ver, kb, persp, ksq, us, Piece::Rook, r_from);
                let add1 = feature_index_v(ver, kb, persp, ksq, us, Piece::King, to);
                let add2 = feature_index_v(ver, kb, persp, ksq, us, Piece::Rook, r_to);
                vec_add2_sub2(dst, src, net.ft_row(add1), net.ft_row(add2), net.ft_row(sub1), net.ft_row(sub2));
            }
            FLAG_CASTLE_Q => {
                let (r_from, r_to) = if us == Color::White { (A1, D1) } else { (A8, D8) };
                let sub1 = feature_index_v(ver, kb, persp, ksq, us, Piece::King, from);
                let sub2 = feature_index_v(ver, kb, persp, ksq, us, Piece::Rook, r_from);
                let add1 = feature_index_v(ver, kb, persp, ksq, us, Piece::King, to);
                let add2 = feature_index_v(ver, kb, persp, ksq, us, Piece::Rook, r_to);
                vec_add2_sub2(dst, src, net.ft_row(add1), net.ft_row(add2), net.ft_row(sub1), net.ft_row(sub2));
            }
            FLAG_PROMO_N | FLAG_PROMO_B | FLAG_PROMO_R | FLAG_PROMO_Q => {
                let promo = d.m.promo_piece().unwrap();
                let sub = feature_index_v(ver, kb, persp, ksq, us, Piece::Pawn, from);
                let add = feature_index_v(ver, kb, persp, ksq, us, promo, to);
                vec_add1_sub1(dst, src, net.ft_row(add), net.ft_row(sub));
            }
            FLAG_PROMO_CAP_N | FLAG_PROMO_CAP_B | FLAG_PROMO_CAP_R | FLAG_PROMO_CAP_Q => {
                let promo = d.m.promo_piece().unwrap();
                let cap_p = Piece::from_u8(d.cap_piece);
                let sub1 = feature_index_v(ver, kb, persp, ksq, us, Piece::Pawn, from);
                let sub2 = feature_index_v(ver, kb, persp, ksq, them, cap_p, to);
                let add1 = feature_index_v(ver, kb, persp, ksq, us, promo, to);
                vec_add1_sub2(dst, src, net.ft_row(add1), net.ft_row(sub1), net.ft_row(sub2));
            }
            _ => unreachable!(),
        }
    }
}

// ---------- SIMD kernels (aarch64 NEON + dotprod sdot, x86-64 AVX2, scalar fallback) ----------

#[inline]
pub fn vec_add_assign(dst: &mut [i16], src: &[i16]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let mut i = 0;
        while i + 32 <= n {
            let a0 = vld1q_s16(dp.add(i));
            let b0 = vld1q_s16(sp.add(i));
            let a1 = vld1q_s16(dp.add(i + 8));
            let b1 = vld1q_s16(sp.add(i + 8));
            let a2 = vld1q_s16(dp.add(i + 16));
            let b2 = vld1q_s16(sp.add(i + 16));
            let a3 = vld1q_s16(dp.add(i + 24));
            let b3 = vld1q_s16(sp.add(i + 24));
            vst1q_s16(dp.add(i), vaddq_s16(a0, b0));
            vst1q_s16(dp.add(i + 8), vaddq_s16(a1, b1));
            vst1q_s16(dp.add(i + 16), vaddq_s16(a2, b2));
            vst1q_s16(dp.add(i + 24), vaddq_s16(a3, b3));
            i += 32;
        }
        while i + 8 <= n {
            vst1q_s16(dp.add(i), vaddq_s16(vld1q_s16(dp.add(i)), vld1q_s16(sp.add(i))));
            i += 8;
        }
        while i < n {
            dst[i] = dst[i].wrapping_add(src[i]);
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let n = dst.len();
        let mut i = 0;
        while i + 32 <= n {
            let a0 = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
            let b0 = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            let a1 = _mm256_loadu_si256(dst.as_ptr().add(i + 16) as *const __m256i);
            let b1 = _mm256_loadu_si256(src.as_ptr().add(i + 16) as *const __m256i);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, _mm256_add_epi16(a0, b0));
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + 16) as *mut __m256i, _mm256_add_epi16(a1, b1));
            i += 32;
        }
        while i + 16 <= n {
            let a = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, _mm256_add_epi16(a, b));
            i += 16;
        }
        while i < n {
            dst[i] = dst[i].wrapping_add(src[i]);
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = d.wrapping_add(s);
    }
}

#[inline]
pub fn vec_sub_assign(dst: &mut [i16], src: &[i16]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let mut i = 0;
        while i + 32 <= n {
            let a0 = vld1q_s16(dp.add(i));
            let b0 = vld1q_s16(sp.add(i));
            let a1 = vld1q_s16(dp.add(i + 8));
            let b1 = vld1q_s16(sp.add(i + 8));
            let a2 = vld1q_s16(dp.add(i + 16));
            let b2 = vld1q_s16(sp.add(i + 16));
            let a3 = vld1q_s16(dp.add(i + 24));
            let b3 = vld1q_s16(sp.add(i + 24));
            vst1q_s16(dp.add(i), vsubq_s16(a0, b0));
            vst1q_s16(dp.add(i + 8), vsubq_s16(a1, b1));
            vst1q_s16(dp.add(i + 16), vsubq_s16(a2, b2));
            vst1q_s16(dp.add(i + 24), vsubq_s16(a3, b3));
            i += 32;
        }
        while i + 8 <= n {
            vst1q_s16(dp.add(i), vsubq_s16(vld1q_s16(dp.add(i)), vld1q_s16(sp.add(i))));
            i += 8;
        }
        while i < n {
            dst[i] = dst[i].wrapping_sub(src[i]);
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let n = dst.len();
        let mut i = 0;
        while i + 32 <= n {
            let a0 = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
            let b0 = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            let a1 = _mm256_loadu_si256(dst.as_ptr().add(i + 16) as *const __m256i);
            let b1 = _mm256_loadu_si256(src.as_ptr().add(i + 16) as *const __m256i);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, _mm256_sub_epi16(a0, b0));
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + 16) as *mut __m256i, _mm256_sub_epi16(a1, b1));
            i += 32;
        }
        while i + 16 <= n {
            let a = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, _mm256_sub_epi16(a, b));
            i += 16;
        }
        while i < n {
            dst[i] = dst[i].wrapping_sub(src[i]);
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = d.wrapping_sub(s);
    }
}

#[inline]
fn vec_add1_sub1(dst: &mut [i16], src: &[i16], add: &[i16], sub: &[i16]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let ap = add.as_ptr();
        let bp = sub.as_ptr();
        let mut i = 0;
        while i + 32 <= n {
            let r0 = vsubq_s16(vaddq_s16(vld1q_s16(sp.add(i)), vld1q_s16(ap.add(i))), vld1q_s16(bp.add(i)));
            let r1 = vsubq_s16(vaddq_s16(vld1q_s16(sp.add(i + 8)), vld1q_s16(ap.add(i + 8))), vld1q_s16(bp.add(i + 8)));
            let r2 =
                vsubq_s16(vaddq_s16(vld1q_s16(sp.add(i + 16)), vld1q_s16(ap.add(i + 16))), vld1q_s16(bp.add(i + 16)));
            let r3 =
                vsubq_s16(vaddq_s16(vld1q_s16(sp.add(i + 24)), vld1q_s16(ap.add(i + 24))), vld1q_s16(bp.add(i + 24)));
            vst1q_s16(dp.add(i), r0);
            vst1q_s16(dp.add(i + 8), r1);
            vst1q_s16(dp.add(i + 16), r2);
            vst1q_s16(dp.add(i + 24), r3);
            i += 32;
        }
        while i + 8 <= n {
            vst1q_s16(
                dp.add(i),
                vsubq_s16(vaddq_s16(vld1q_s16(sp.add(i)), vld1q_s16(ap.add(i))), vld1q_s16(bp.add(i))),
            );
            i += 8;
        }
        while i < n {
            dst[i] = src[i].wrapping_add(add[i]).wrapping_sub(sub[i]);
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let n = dst.len();
        let mut i = 0;
        while i + 32 <= n {
            let s0 = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            let a0 = _mm256_loadu_si256(add.as_ptr().add(i) as *const __m256i);
            let b0 = _mm256_loadu_si256(sub.as_ptr().add(i) as *const __m256i);
            let s1 = _mm256_loadu_si256(src.as_ptr().add(i + 16) as *const __m256i);
            let a1 = _mm256_loadu_si256(add.as_ptr().add(i + 16) as *const __m256i);
            let b1 = _mm256_loadu_si256(sub.as_ptr().add(i + 16) as *const __m256i);
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_sub_epi16(_mm256_add_epi16(s0, a0), b0),
            );
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(i + 16) as *mut __m256i,
                _mm256_sub_epi16(_mm256_add_epi16(s1, a1), b1),
            );
            i += 32;
        }
        while i + 16 <= n {
            let s = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            let a = _mm256_loadu_si256(add.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(sub.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, _mm256_sub_epi16(_mm256_add_epi16(s, a), b));
            i += 16;
        }
        while i < n {
            dst[i] = src[i].wrapping_add(add[i]).wrapping_sub(sub[i]);
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..dst.len() {
        dst[i] = src[i].wrapping_add(add[i]).wrapping_sub(sub[i]);
    }
}

#[inline]
fn vec_add1_sub2(dst: &mut [i16], src: &[i16], add1: &[i16], sub1: &[i16], sub2: &[i16]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let a1p = add1.as_ptr();
        let b1p = sub1.as_ptr();
        let b2p = sub2.as_ptr();
        let mut i = 0;
        while i + 32 <= n {
            for k in [0usize, 8, 16, 24] {
                let s = vld1q_s16(sp.add(i + k));
                let a1 = vld1q_s16(a1p.add(i + k));
                let b1 = vld1q_s16(b1p.add(i + k));
                let b2 = vld1q_s16(b2p.add(i + k));
                vst1q_s16(dp.add(i + k), vsubq_s16(vsubq_s16(vaddq_s16(s, a1), b1), b2));
            }
            i += 32;
        }
        while i + 8 <= n {
            let s = vld1q_s16(sp.add(i));
            let a1 = vld1q_s16(a1p.add(i));
            let b1 = vld1q_s16(b1p.add(i));
            let b2 = vld1q_s16(b2p.add(i));
            vst1q_s16(dp.add(i), vsubq_s16(vsubq_s16(vaddq_s16(s, a1), b1), b2));
            i += 8;
        }
        while i < n {
            dst[i] = src[i].wrapping_add(add1[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let n = dst.len();
        let mut i = 0;
        while i + 16 <= n {
            let s = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            let a1 = _mm256_loadu_si256(add1.as_ptr().add(i) as *const __m256i);
            let b1 = _mm256_loadu_si256(sub1.as_ptr().add(i) as *const __m256i);
            let b2 = _mm256_loadu_si256(sub2.as_ptr().add(i) as *const __m256i);
            let r = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(s, a1), b1), b2);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, r);
            i += 16;
        }
        while i < n {
            dst[i] = src[i].wrapping_add(add1[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..dst.len() {
        dst[i] = src[i].wrapping_add(add1[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
    }
}

#[inline]
fn vec_add2_sub2(dst: &mut [i16], src: &[i16], add1: &[i16], add2: &[i16], sub1: &[i16], sub2: &[i16]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = dst.len();
        let mut i = 0;
        while i + 8 <= n {
            let s = vld1q_s16(src.as_ptr().add(i));
            let a1 = vld1q_s16(add1.as_ptr().add(i));
            let a2 = vld1q_s16(add2.as_ptr().add(i));
            let b1 = vld1q_s16(sub1.as_ptr().add(i));
            let b2 = vld1q_s16(sub2.as_ptr().add(i));
            vst1q_s16(dst.as_mut_ptr().add(i), vsubq_s16(vsubq_s16(vaddq_s16(vaddq_s16(s, a1), a2), b1), b2));
            i += 8;
        }
        while i < n {
            dst[i] = src[i].wrapping_add(add1[i]).wrapping_add(add2[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let n = dst.len();
        let mut i = 0;
        while i + 16 <= n {
            let s = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            let a1 = _mm256_loadu_si256(add1.as_ptr().add(i) as *const __m256i);
            let a2 = _mm256_loadu_si256(add2.as_ptr().add(i) as *const __m256i);
            let b1 = _mm256_loadu_si256(sub1.as_ptr().add(i) as *const __m256i);
            let b2 = _mm256_loadu_si256(sub2.as_ptr().add(i) as *const __m256i);
            let r = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(s, a1), a2), b1), b2);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, r);
            i += 16;
        }
        while i < n {
            dst[i] = src[i].wrapping_add(add1[i]).wrapping_add(add2[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..dst.len() {
        dst[i] = src[i].wrapping_add(add1[i]).wrapping_add(add2[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
    }
}

/// Specialized zero-threat 16-neuron unrolled Pairwise-CReLU quantizer (`i16` -> `i8` in `[0, 127]`).
#[inline]
fn pairwise_quantize_simd(acc: &[i16], out: &mut [i8], qa: i16) {
    let half = out.len();
    debug_assert_eq!(acc.len(), half * 2);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let zero = vdupq_n_s16(0);
        let vmax = vdupq_n_s16(qa);
        let p_a = acc.as_ptr();
        let p_b = acc.as_ptr().add(half);
        let p_o = out.as_mut_ptr();
        let mut i = 0;
        while i + 16 <= half {
            let a0 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(p_a.add(i)), zero), vmax));
            let b0 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(p_b.add(i)), zero), vmax));
            let a1 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(p_a.add(i + 8)), zero), vmax));
            let b1 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(p_b.add(i + 8)), zero), vmax));
            let s0 = vshrn_n_u16::<8>(vmulq_u16(a0, b0));
            let s1 = vshrn_n_u16::<8>(vmulq_u16(a1, b1));
            let comb = vshr_n_u8::<1>(s0);
            let comb1 = vshr_n_u8::<1>(s1);
            vst1q_s8(p_o.add(i), vreinterpretq_s8_u8(vcombine_u8(comb, comb1)));
            i += 16;
        }
        while i < half {
            let a = acc[i].clamp(0, qa) as u32;
            let b = acc[i + half].clamp(0, qa) as u32;
            out[i] = ((a * b) >> 9) as i8;
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..half {
        let a = acc[i].clamp(0, qa) as u32;
        let b = acc[i + half].clamp(0, qa) as u32;
        out[i] = ((a * b) >> 9) as i8;
    }
}

/// Fused 4-row-at-a-time ARM64 `sdot` Layer 2 matmul (`32` i8 -> `32` i8 clamped `[0, 127]`).
/// Loads `l1_act[0..32]` into two NEON registers once and computes 4 output neurons per loop iteration.
#[inline]
fn l2_matmul_32x32_simd(l1_act: &[i8; 32], l2_w: &[i8], l2_b: &[i32], l2_act: &mut [i8; 32]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    unsafe {
        use std::arch::aarch64::*;
        #[inline(always)]
        unsafe fn sdot(mut acc: int32x4_t, x: int8x16_t, y: int8x16_t) -> int32x4_t {
            std::arch::asm!(
                "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc,
                x = in(vreg) x,
                y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags),
            );
            acc
        }
        let x0 = vld1q_s8(l1_act.as_ptr());
        let x1 = vld1q_s8(l1_act.as_ptr().add(16));
        let wp = l2_w.as_ptr();
        let bp = l2_b.as_ptr();
        let zero = vdupq_n_s32(0);
        let vmax = vdupq_n_s32(127);
        for k in (0..32).step_by(4) {
            let w0 = wp.add(k * 32);
            let w1 = w0.add(32);
            let w2 = w0.add(64);
            let w3 = w0.add(96);
            let r0 = sdot(sdot(zero, vld1q_s8(w0), x0), vld1q_s8(w0.add(16)), x1);
            let r1 = sdot(sdot(zero, vld1q_s8(w1), x0), vld1q_s8(w1.add(16)), x1);
            let r2 = sdot(sdot(zero, vld1q_s8(w2), x0), vld1q_s8(w2.add(16)), x1);
            let r3 = sdot(sdot(zero, vld1q_s8(w3), x0), vld1q_s8(w3.add(16)), x1);
            let p01 = vpaddq_s32(r0, r1);
            let p23 = vpaddq_s32(r2, r3);
            let sum4 = vaddq_s32(vpaddq_s32(p01, p23), vld1q_s32(bp.add(k)));
            let clamped = vminq_s32(vmaxq_s32(vshrq_n_s32::<6>(sum4), zero), vmax);
            l2_act[k] = vgetq_lane_s32::<0>(clamped) as i8;
            l2_act[k + 1] = vgetq_lane_s32::<1>(clamped) as i8;
            l2_act[k + 2] = vgetq_lane_s32::<2>(clamped) as i8;
            l2_act[k + 3] = vgetq_lane_s32::<3>(clamped) as i8;
        }
        return;
    }
    #[allow(unreachable_code)]
    for k in 0..32 {
        let row = &l2_w[k * 32..(k + 1) * 32];
        let pre = dot_i8_simd(l1_act, row) + l2_b[k];
        l2_act[k] = (pre >> 6).clamp(0, 127) as i8;
    }
}

/// Branchless transposed Layer 1 matmul (`hl` i8 -> `16` i32): updates all 16 output neurons in 4 NEON
/// registers with 4 independent `sdot` instructions per 4-byte chunk (zero branch mispredicts!).
#[inline]
fn l1_matmul_nnz_simd(ft_q: &[i8], l1_w_t: &[i8], l1_b: &[i32], out: &mut [i32; 16]) {
    let chunks = ft_q.len() / 4;
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    unsafe {
        use std::arch::aarch64::*;
        #[inline(always)]
        unsafe fn sdot(mut acc: int32x4_t, x: int8x16_t, y: int8x16_t) -> int32x4_t {
            std::arch::asm!(
                "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc,
                x = in(vreg) x,
                y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags),
            );
            acc
        }
        // Two-pass NNZ extraction (Stockfish trick: branchless compaction of non-zero u32 chunk indices!)
        let ft_u32 = ft_q.as_ptr() as *const u32;
        let mut nnz_idx = [std::mem::MaybeUninit::<u16>::uninit(); 512];
        let mut nnz_cnt = 0usize;
        for c in 0..chunks {
            let v = *ft_u32.add(c);
            *nnz_idx.get_unchecked_mut(nnz_cnt) = std::mem::MaybeUninit::new(c as u16);
            nnz_cnt += usize::from(v != 0);
        }

        let mut a0 = vld1q_s32(l1_b.as_ptr());
        let mut a1 = vld1q_s32(l1_b.as_ptr().add(4));
        let mut a2 = vld1q_s32(l1_b.as_ptr().add(8));
        let mut a3 = vld1q_s32(l1_b.as_ptr().add(12));
        let wp_base = l1_w_t.as_ptr();
        for k in 0..nnz_cnt {
            let c = nnz_idx.get_unchecked(k).assume_init() as usize;
            let v = *ft_u32.add(c);
            let xv = vreinterpretq_s8_u32(vdupq_n_u32(v));
            let wp = wp_base.add(c * 64);
            a0 = sdot(a0, vld1q_s8(wp), xv);
            a1 = sdot(a1, vld1q_s8(wp.add(16)), xv);
            a2 = sdot(a2, vld1q_s8(wp.add(32)), xv);
            a3 = sdot(a3, vld1q_s8(wp.add(48)), xv);
        }
        vst1q_s32(out.as_mut_ptr(), a0);
        vst1q_s32(out.as_mut_ptr().add(4), a1);
        vst1q_s32(out.as_mut_ptr().add(8), a2);
        vst1q_s32(out.as_mut_ptr().add(12), a3);
        return;
    }
    #[allow(unreachable_code)]
    {
        out.copy_from_slice(&l1_b[..16]);
        for c in 0..chunks {
            let base = c * 4;
            let (x0, x1, x2, x3) =
                (ft_q[base] as i32, ft_q[base + 1] as i32, ft_q[base + 2] as i32, ft_q[base + 3] as i32);
            if (x0 | x1 | x2 | x3) == 0 {
                continue;
            }
            let wb = c * 64;
            for j in 0..16 {
                let wj = wb + j * 4;
                out[j] += x0 * l1_w_t[wj] as i32
                    + x1 * l1_w_t[wj + 1] as i32
                    + x2 * l1_w_t[wj + 2] as i32
                    + x3 * l1_w_t[wj + 3] as i32;
            }
        }
    }
}

/// Fast `i8 * i8 -> i32` dot product using ARM64 `sdot` (16 MACs/instruction via stable inline asm).
#[inline]
fn dot_i8_simd(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    unsafe {
        use std::arch::aarch64::*;
        #[inline(always)]
        unsafe fn sdot(mut acc: int32x4_t, x: int8x16_t, y: int8x16_t) -> int32x4_t {
            std::arch::asm!(
                "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                acc = inout(vreg) acc,
                x = in(vreg) x,
                y = in(vreg) y,
                options(pure, nomem, nostack, preserves_flags),
            );
            acc
        }
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let mut s0 = vdupq_n_s32(0);
        let mut s1 = vdupq_n_s32(0);
        let mut s2 = vdupq_n_s32(0);
        let mut s3 = vdupq_n_s32(0);
        let mut i = 0;
        while i + 64 <= n {
            s0 = sdot(s0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            s1 = sdot(s1, vld1q_s8(pa.add(i + 16)), vld1q_s8(pb.add(i + 16)));
            s2 = sdot(s2, vld1q_s8(pa.add(i + 32)), vld1q_s8(pb.add(i + 32)));
            s3 = sdot(s3, vld1q_s8(pa.add(i + 48)), vld1q_s8(pb.add(i + 48)));
            i += 64;
        }
        while i + 32 <= n {
            s0 = sdot(s0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            s1 = sdot(s1, vld1q_s8(pa.add(i + 16)), vld1q_s8(pb.add(i + 16)));
            i += 32;
        }
        while i + 16 <= n {
            s0 = sdot(s0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            i += 16;
        }
        let mut sum = vaddvq_s32(vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3)));
        while i < n {
            sum += (a[i] as i32) * (b[i] as i32);
            i += 1;
        }
        return sum;
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0i32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            sum += (x as i32) * (y as i32);
        }
        sum
    }
}

/// Scalar SCReLU dot product (v1): `sum_i clamp(acc[i], 0, qa)^2 * w[i]`.
#[inline]
pub fn screlu_dot_scalar(acc: &[i16], w: &[i16], qa: i16) -> i64 {
    let mut sum = 0i64;
    for (&x, &wi) in acc.iter().zip(w.iter()) {
        let c = x.clamp(0, qa) as i64;
        sum += (c * c) * (wi as i64);
    }
    sum
}

/// SIMD SCReLU dot product (v1): bit-identical to `screlu_dot_scalar`.
#[inline]
pub fn screlu_dot_simd(acc: &[i16], w: &[i16], qa: i16) -> i64 {
    debug_assert_eq!(acc.len(), w.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = acc.len();
        let zero = vdupq_n_s16(0);
        let vmax = vdupq_n_s16(qa);
        let mut total = 0i64;
        let mut i = 0;
        while i + 64 <= n {
            let mut s0 = vdupq_n_s32(0);
            let mut s1 = vdupq_n_s32(0);
            for _ in 0..8 {
                let x = vld1q_s16(acc.as_ptr().add(i));
                let wt = vld1q_s16(w.as_ptr().add(i));
                let c = vminq_s16(vmaxq_s16(x, zero), vmax);
                let cw = vmulq_s16(c, wt);
                s0 = vmlal_s16(s0, vget_low_s16(c), vget_low_s16(cw));
                s1 = vmlal_high_s16(s1, c, cw);
                i += 8;
            }
            let sum64 = vaddq_s64(vpaddlq_s32(s0), vpaddlq_s32(s1));
            total += vgetq_lane_s64(sum64, 0) + vgetq_lane_s64(sum64, 1);
        }
        while i < n {
            let c = acc[i].clamp(0, qa) as i64;
            total += (c * c) * (w[i] as i64);
            i += 1;
        }
        return total;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let n = acc.len();
        let zero = _mm256_setzero_si256();
        let vmax = _mm256_set1_epi16(qa);
        let mut total = 0i64;
        let mut i = 0;
        while i + 64 <= n {
            let mut s = _mm256_setzero_si256();
            for _ in 0..4 {
                let x = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                let wt = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
                let c = _mm256_min_epi16(_mm256_max_epi16(x, zero), vmax);
                let cw = _mm256_mullo_epi16(c, wt);
                s = _mm256_add_epi32(s, _mm256_madd_epi16(c, cw));
                i += 16;
            }
            let sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
            total += _mm_extract_epi32::<0>(sum128) as i64
                + _mm_extract_epi32::<1>(sum128) as i64
                + _mm_extract_epi32::<2>(sum128) as i64
                + _mm_extract_epi32::<3>(sum128) as i64;
        }
        while i < n {
            let c = acc[i].clamp(0, qa) as i64;
            total += (c * c) * (w[i] as i64);
            i += 1;
        }
        return total;
    }
    #[allow(unreachable_code)]
    screlu_dot_scalar(acc, w, qa)
}

/// Scalar Pairwise-CReLU dot product (v2): `sum_{i=0..half} clamp(acc[i], 0, qa) * clamp(acc[i+half], 0, qa) * w[i]`.
#[inline]
pub fn pairwise_dot_scalar(acc: &[i16], w: &[i16], qa: i16) -> i64 {
    let half = w.len();
    debug_assert_eq!(acc.len(), half * 2);
    let mut sum = 0i64;
    for i in 0..half {
        let a = acc[i].clamp(0, qa) as i64;
        let b = acc[i + half].clamp(0, qa) as i64;
        sum += (a * b) * (w[i] as i64);
    }
    sum
}

/// SIMD Pairwise-CReLU dot product (v2): ARM64 NEON & x86-64 AVX2, bit-identical to scalar.
#[inline]
pub fn pairwise_dot_simd(acc: &[i16], w: &[i16], qa: i16) -> i64 {
    let half = w.len();
    debug_assert_eq!(acc.len(), half * 2);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let zero = vdupq_n_s16(0);
        let vmax = vdupq_n_s16(qa);
        let mut total = 0i64;
        let mut i = 0;
        let p_a = acc.as_ptr();
        let p_b = acc.as_ptr().add(half);
        let p_w = w.as_ptr();
        while i + 64 <= half {
            let mut s0 = vdupq_n_s32(0);
            let mut s1 = vdupq_n_s32(0);
            for _ in 0..8 {
                let a = vminq_s16(vmaxq_s16(vld1q_s16(p_a.add(i)), zero), vmax);
                let b = vminq_s16(vmaxq_s16(vld1q_s16(p_b.add(i)), zero), vmax);
                let wt = vld1q_s16(p_w.add(i));
                let aw = vmulq_s16(a, wt);
                s0 = vmlal_s16(s0, vget_low_s16(b), vget_low_s16(aw));
                s1 = vmlal_high_s16(s1, b, aw);
                i += 8;
            }
            let sum64 = vaddq_s64(vpaddlq_s32(s0), vpaddlq_s32(s1));
            total += vgetq_lane_s64(sum64, 0) + vgetq_lane_s64(sum64, 1);
        }
        while i < half {
            let a = acc[i].clamp(0, qa) as i64;
            let b = acc[i + half].clamp(0, qa) as i64;
            total += (a * b) * (w[i] as i64);
            i += 1;
        }
        return total;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let vmax = _mm256_set1_epi16(qa);
        let mut total = 0i64;
        let mut i = 0;
        let p_a = acc.as_ptr();
        let p_b = acc.as_ptr().add(half);
        let p_w = w.as_ptr();
        while i + 64 <= half {
            let mut s = _mm256_setzero_si256();
            for _ in 0..4 {
                let a =
                    _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p_a.add(i) as *const __m256i), zero), vmax);
                let b =
                    _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p_b.add(i) as *const __m256i), zero), vmax);
                let wt = _mm256_loadu_si256(p_w.add(i) as *const __m256i);
                let aw = _mm256_mullo_epi16(a, wt);
                s = _mm256_add_epi32(s, _mm256_madd_epi16(b, aw));
                i += 16;
            }
            let sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
            total += _mm_extract_epi32::<0>(sum128) as i64
                + _mm_extract_epi32::<1>(sum128) as i64
                + _mm_extract_epi32::<2>(sum128) as i64
                + _mm_extract_epi32::<3>(sum128) as i64;
        }
        while i < half {
            let a = acc[i].clamp(0, qa) as i64;
            let b = acc[i + half].clamp(0, qa) as i64;
            total += (a * b) * (w[i] as i64);
            i += 1;
        }
        return total;
    }
    #[allow(unreachable_code)]
    pairwise_dot_scalar(acc, w, qa)
}
