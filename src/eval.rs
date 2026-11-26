//! Classical Hand-Crafted Evaluation (Tapered PeSTO PSTs + Pawn Structure + Mobility + King Safety)
//! and Static Exchange Evaluation (SEE).

use crate::bitboard::{
    bb, bishop_attacks, file_bb, king_attacks, knight_attacks, pop_lsb, queen_attacks,
    rook_attacks, tables,
};
use crate::board::Board;
use crate::types::{sq_file, sq_flip, sq_rank, Color, Move, Piece, Square};

/// SEE piece values (centipawns).
pub const SEE_VAL: [i32; 6] = [100, 320, 330, 500, 900, 20000];

/// Tapered phase weights per piece type: P, N, B, R, Q, K.
const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];

/// PeSTO middlegame material values.
const MG_VAL: [i32; 6] = [82, 337, 365, 477, 1025, 0];
/// PeSTO endgame material values.
const EG_VAL: [i32; 6] = [94, 281, 297, 512, 936, 0];

// PeSTO Piece-Square Tables (from White's perspective, A8 = index 0 .. H1 = index 63; indexed via sq ^ 56 for White, sq for Black).
#[rustfmt::skip]
const MG_PST: [[i32; 64]; 6] = [
    // Pawn
    [
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    ],
    // Knight
    [
       -167, -89, -34, -49,  61, -97, -15, -107,
        -73, -41,  72,  36,  23,  62,   7,  -17,
        -47,  60,  37,  65,  84, 129,  73,   44,
         -9,  17,  19,  53,  37,  69,  18,   22,
        -13,   4,  16,  13,  28,  19,  21,   -8,
        -23,  -9,  12,  10,  19,  17,  25,  -16,
        -29, -53, -12,  -3,  -1,  18, -14,  -19,
       -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    // Bishop
    [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    // Rook
    [
         32,  42,  32,  51, 63,  9,  31,  43,
         27,  32,  58,  62, 80, 67,  26,  44,
         -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    ],
    // Queen
    [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    // King
    [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
];

#[rustfmt::skip]
const EG_PST: [[i32; 64]; 6] = [
    // Pawn
    [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    // Knight
    [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    // Bishop
    [
        -14, -21, -11,  -8, -7,  -9, -17, -24,
         -8,  -4,   7, -12, -3, -13,  -4, -14,
          2,  -8,   0,  -1, -2,   6,   0,   4,
         -3,   9,  12,   9, 14,  10,   3,   2,
         -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    ],
    // Rook
    [
         13, 10, 18, 15, 12,  12,   8,   5,
         11, 13, 13, 11, -3,   3,   8,   3,
          7,  7,  7,  5,  4,  -3,  -5,  -3,
          4,  3, 13,  1,  2,   1,  -1,   2,
          3,  5,  8,  4, -5,  -6,  -8, -11,
         -4,  0, -5, -1, -7, -12,  -8, -16,
         -6, -6,  0,  2, -9,  -9, -11,  -3,
         -9,  2,  3, -1, -5, -13,   4, -20,
    ],
    // Queen
    [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    // King
    [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
];

const PASSED_MG: [i32; 8] = [0, 5, 10, 20, 35, 60, 100, 0];
const PASSED_EG: [i32; 8] = [0, 10, 20, 40, 70, 120, 180, 0];

/// Classical tapered evaluation from the perspective of `board.side` (in centipawns).
pub fn evaluate_classical(b: &Board) -> i32 {
    let t = tables();
    let occ = b.occ();
    let wp = b.piece_bb(Color::White, Piece::Pawn);
    let bp = b.piece_bb(Color::Black, Piece::Pawn);

    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut phase = 0i32;

    for c_idx in 0..2 {
        let color = if c_idx == 0 {
            Color::White
        } else {
            Color::Black
        };
        let own_pawns = if c_idx == 0 { wp } else { bp };
        let opp_pawns = if c_idx == 0 { bp } else { wp };
        let own_bb = b.colors[c_idx];

        // Material + PSTs + piece-specific terms
        for p_idx in 0..6 {
            let mut bb_p = own_bb & b.pieces[p_idx];
            while bb_p != 0 {
                let sq = pop_lsb(&mut bb_p);
                let pst_sq = if c_idx == 0 {
                    sq_flip(sq) as usize
                } else {
                    sq as usize
                };
                mg[c_idx] += MG_VAL[p_idx] + MG_PST[p_idx][pst_sq];
                eg[c_idx] += EG_VAL[p_idx] + EG_PST[p_idx][pst_sq];
                phase += PHASE_INC[p_idx];

                match p_idx {
                    0 => {
                        // Pawn structure: passed, isolated, doubled
                        let f = sq_file(sq) as usize;
                        let r_rel = if c_idx == 0 {
                            sq_rank(sq)
                        } else {
                            7 - sq_rank(sq)
                        } as usize;
                        if (t.passed_mask[c_idx][sq as usize] & opp_pawns) == 0 {
                            mg[c_idx] += PASSED_MG[r_rel];
                            eg[c_idx] += PASSED_EG[r_rel];
                        }
                        if (t.adj_files[f] & own_pawns) == 0 {
                            mg[c_idx] -= 10;
                            eg[c_idx] -= 12;
                        }
                        if (file_bb(f as u8) & own_pawns).count_ones() > 1 {
                            mg[c_idx] -= 8;
                            eg[c_idx] -= 14;
                        }
                    }
                    1 => {
                        // Knight mobility
                        let mob = (knight_attacks(sq) & !own_bb).count_ones() as i32;
                        mg[c_idx] += (mob - 4) * 4;
                        eg[c_idx] += (mob - 4) * 3;
                    }
                    2 => {
                        // Bishop mobility
                        let mob = (bishop_attacks(sq, occ) & !own_bb).count_ones() as i32;
                        mg[c_idx] += (mob - 6) * 4;
                        eg[c_idx] += (mob - 6) * 4;
                    }
                    3 => {
                        // Rook on open / semi-open file + mobility
                        let fmask = file_bb(sq_file(sq));
                        if (fmask & own_pawns) == 0 {
                            if (fmask & opp_pawns) == 0 {
                                mg[c_idx] += 22;
                                eg[c_idx] += 10;
                            } else {
                                mg[c_idx] += 11;
                                eg[c_idx] += 5;
                            }
                        }
                        let mob = (rook_attacks(sq, occ) & !own_bb).count_ones() as i32;
                        mg[c_idx] += (mob - 7) * 2;
                        eg[c_idx] += (mob - 7) * 3;
                    }
                    4 => {
                        let mob = (queen_attacks(sq, occ) & !own_bb).count_ones() as i32;
                        mg[c_idx] += (mob - 14) * 1;
                        eg[c_idx] += (mob - 14) * 2;
                    }
                    5 => {
                        // Simple pawn shield in middlegame
                        let r = sq_rank(sq);
                        let shield_rank = if c_idx == 0 {
                            r + 1
                        } else {
                            r.saturating_sub(1)
                        };
                        if (c_idx == 0 && r <= 1) || (c_idx == 1 && r >= 6) {
                            let shield_mask = king_attacks(sq) & (0xFFu64 << (shield_rank * 8));
                            let shield_cnt = (shield_mask & own_pawns).count_ones() as i32;
                            mg[c_idx] += shield_cnt * 12;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Bishop pair bonus
        if b.piece_bb(color, Piece::Bishop).count_ones() >= 2 {
            mg[c_idx] += 28;
            eg[c_idx] += 42;
        }
    }

    let mg_phase = phase.min(24);
    let eg_phase = 24 - mg_phase;
    let mg_score = mg[0] - mg[1];
    let eg_score = eg[0] - eg[1];
    let white_score = (mg_score * mg_phase + eg_score * eg_phase) / 24;

    // Small tempo bonus for side to move
    const TEMPO: i32 = 12;
    if b.side == Color::White {
        white_score + TEMPO
    } else {
        -white_score + TEMPO
    }
}

/// Static Exchange Evaluation: returns true if the capture/promotion `m` has SEE >= `threshold`.
pub fn see_ge(b: &Board, m: Move, threshold: i32) -> bool {
    let from = m.from();
    let to = m.to();

    let victim_val = if m.is_ep() {
        SEE_VAL[Piece::Pawn.idx()]
    } else if let Some((_, vp)) = b.piece_at(to) {
        SEE_VAL[vp.idx()]
    } else {
        0
    };
    let promo_bonus = if let Some(pp) = m.promo_piece() {
        SEE_VAL[pp.idx()] - SEE_VAL[Piece::Pawn.idx()]
    } else {
        0
    };
    let moved_piece = b.piece_at(from).map(|(_, p)| p).unwrap_or(Piece::Pawn);
    let next_victim = m.promo_piece().unwrap_or(moved_piece);

    let mut balance = victim_val + promo_bonus - threshold;
    if balance < 0 {
        return false;
    }
    balance -= SEE_VAL[next_victim.idx()];
    if balance >= 0 {
        return true;
    }

    let mut occ = (b.occ() ^ bb(from)) | bb(to);
    if m.is_ep() {
        let cap_sq = if b.side == Color::White {
            to - 8
        } else {
            to + 8
        };
        occ ^= bb(cap_sq);
    }

    let diag = b.pieces[Piece::Bishop.idx()] | b.pieces[Piece::Queen.idx()];
    let orth = b.pieces[Piece::Rook.idx()] | b.pieces[Piece::Queen.idx()];
    let mut attackers = b.all_attackers_to(to, occ) & occ;
    let mut side = b.side.flip();
    let mut res = true;

    loop {
        let side_att = attackers & b.colors[side.idx()];
        if side_att == 0 {
            break;
        }
        // Pick least valuable attacker
        let (lva_piece, lva_sq) = match least_valuable_attacker(b, side_att) {
            Some(x) => x,
            None => break,
        };
        res = !res;
        balance = -balance - 1 - SEE_VAL[lva_piece.idx()];
        if balance >= 0 {
            // If king captures while opponent still attacks `to`, king capture is illegal -> lose.
            if lva_piece == Piece::King && (attackers & b.colors[side.flip().idx()]) != 0 {
                res = !res;
            }
            break;
        }
        occ ^= bb(lva_sq);
        if matches!(lva_piece, Piece::Pawn | Piece::Bishop | Piece::Queen) {
            attackers |= bishop_attacks(to, occ) & diag;
        }
        if matches!(lva_piece, Piece::Rook | Piece::Queen) {
            attackers |= rook_attacks(to, occ) & orth;
        }
        attackers &= occ;
        side = side.flip();
    }
    res
}

#[inline(always)]
fn least_valuable_attacker(b: &Board, att: u64) -> Option<(Piece, Square)> {
    for p in Piece::ALL {
        let subset = att & b.pieces[p.idx()];
        if subset != 0 {
            return Some((p, subset.trailing_zeros() as Square));
        }
    }
    None
}
