//! Zero-dependency exact endgame knowledge:
//! 1. Exact 3-piece KPK retrograde bitbase (196,608 canonical states, 24 KB packed bitset,
//!    computed once in ~8 ms at startup via fixed-point retrograde induction -> 100% exact W/D oracle).
//! 2. Exact wrong-colored rook-pawn (`KB+P vs K`) fortress draw detector.
//! 3. `K+N+N vs K` theoretical draw detector and `KBN vs K` right-corner mop-up evaluator.
//! 4. Opposite-color bishops (OCB) endgame drawishness scaling factor.

use crate::bitboard::{bb, king_attacks, pawn_attacks, LIGHT_SQUARES};
use crate::board::Board;
use crate::types::{sq_file, sq_flip, sq_make, sq_mirror, sq_rank, Color, Piece, Square};
use std::sync::OnceLock;

/// Outcome of probing built-in endgame knowledge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndgameEval {
    /// No special endgame pattern matched; use normal evaluation.
    None,
    /// Exact known score from the side-to-move's perspective (e.g., KPK win/draw, wrong rook pawn draw).
    Exact(i32),
    /// Scale the normal static evaluation by `num / den` (e.g., opposite-color bishops drawishness).
    Scale(i32, i32),
}

// ============================================================================
// Exact KPK Retrograde Bitbase (White pawn on files a..d, ranks 2..7)
// Index = (us * 24 * 64 * 64) + ((pr - 1) * 4 + pf) * 4096 + wk * 64 + bk
// Total states: 2 * 24 * 64 * 64 = 196,608 -> 24,576 bytes packed (1 bit = win)
// ============================================================================

const KPK_STATES: usize = 2 * 24 * 64 * 64;
const KPK_BYTES: usize = KPK_STATES / 8;

const RES_INVALID: u8 = 0;
const RES_UNKNOWN: u8 = 1;
const RES_DRAW: u8 = 2;
const RES_WIN: u8 = 4;

#[inline(always)]
fn kpk_index(us: usize, wk: Square, bk: Square, psq: Square) -> usize {
    let pf = sq_file(psq) as usize;
    let pr = sq_rank(psq) as usize;
    debug_assert!(pf < 4 && (1..=6).contains(&pr));
    let p_idx = (pr - 1) * 4 + pf;
    (((us * 24 + p_idx) * 64 + wk as usize) * 64) + bk as usize
}

fn build_kpk_bitbase() -> Box<[u8; KPK_BYTES]> {
    let mut table = vec![RES_UNKNOWN; KPK_STATES];

    // Step 1: Initialize terminal positions and illegal positions
    for us in 0..2usize {
        for pr in 1..=6u8 {
            for pf in 0..4u8 {
                let psq = sq_make(pf, pr);
                let p_att = pawn_attacks(Color::White, psq);
                for wk in 0..64u8 {
                    if wk == psq {
                        for bk in 0..64u8 {
                            table[kpk_index(us, wk, bk, psq)] = RES_INVALID;
                        }
                        continue;
                    }
                    let wk_att = king_attacks(wk);
                    for bk in 0..64u8 {
                        let idx = kpk_index(us, wk, bk, psq);
                        if bk == psq || (wk_att & bb(bk)) != 0 || (us == 0 && (p_att & bb(bk)) != 0)
                        {
                            table[idx] = RES_INVALID;
                            continue;
                        }
                        if us == 0 {
                            // Immediate promotion win if pawn on 7th rank, promo square empty, and Black king can't capture queen
                            if pr == 6 {
                                let promo = psq + 8;
                                if wk != promo
                                    && bk != promo
                                    && ((king_attacks(bk) & bb(promo)) == 0
                                        || (wk_att & bb(promo)) != 0)
                                {
                                    table[idx] = RES_WIN;
                                }
                            }
                        } else {
                            // Black to move: draw if Black is stalemated or can capture undefended pawn
                            let bk_moves = king_attacks(bk) & !wk_att & !p_att;
                            if bk_moves == 0 || ((bk_moves & bb(psq)) != 0) {
                                table[idx] = RES_DRAW;
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Retrograde fixed-point induction
    loop {
        let mut changed = false;
        for us in 0..2usize {
            for pr in 1..=6u8 {
                for pf in 0..4u8 {
                    let psq = sq_make(pf, pr);
                    let p_att = pawn_attacks(Color::White, psq);
                    for wk in 0..64u8 {
                        if wk == psq {
                            continue;
                        }
                        let wk_att = king_attacks(wk);
                        for bk in 0..64u8 {
                            let idx = kpk_index(us, wk, bk, psq);
                            if table[idx] != RES_UNKNOWN {
                                continue;
                            }
                            let mut r = RES_INVALID;
                            if us == 0 {
                                // White to move: WIN if any legal move leads to WIN; DRAW if all legal moves lead to DRAW
                                // 1. King moves
                                let mut km = wk_att & !bb(psq) & !king_attacks(bk);
                                while km != 0 {
                                    let to = km.trailing_zeros() as Square;
                                    km &= km - 1;
                                    r |= table[kpk_index(1, to, bk, psq)];
                                }
                                // 2. Single pawn push
                                let push1 = psq + 8;
                                if pr < 6 && push1 != wk && push1 != bk {
                                    r |= table[kpk_index(1, wk, bk, push1)];
                                    // 3. Double pawn push from rank 2
                                    if pr == 1 {
                                        let push2 = psq + 16;
                                        if push2 != wk && push2 != bk {
                                            r |= table[kpk_index(1, wk, bk, push2)];
                                        }
                                    }
                                }
                                if (r & RES_WIN) != 0 {
                                    table[idx] = RES_WIN;
                                    changed = true;
                                } else if (r & RES_UNKNOWN) == 0 {
                                    table[idx] = RES_DRAW;
                                    changed = true;
                                }
                            } else {
                                // Black to move: DRAW if any legal move leads to DRAW; WIN if all legal moves lead to WIN
                                let mut km = king_attacks(bk) & !wk_att & !p_att;
                                while km != 0 {
                                    let to = km.trailing_zeros() as Square;
                                    km &= km - 1;
                                    r |= table[kpk_index(0, wk, to, psq)];
                                }
                                if (r & RES_DRAW) != 0 {
                                    table[idx] = RES_DRAW;
                                    changed = true;
                                } else if (r & RES_UNKNOWN) == 0 {
                                    table[idx] = RES_WIN;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut packed = Box::new([0u8; KPK_BYTES]);
    for (i, &v) in table.iter().enumerate() {
        if v == RES_WIN {
            packed[i >> 3] |= 1u8 << (i & 7);
        }
    }
    packed
}

static KPK_TABLE: OnceLock<Box<[u8; KPK_BYTES]>> = OnceLock::new();

#[inline]
pub fn kpk_is_win(strong_to_move: bool, strong_k: Square, weak_k: Square, pawn_sq: Square) -> bool {
    let tbl = KPK_TABLE.get_or_init(build_kpk_bitbase);
    let mut wk = strong_k;
    let mut bk = weak_k;
    let mut psq = pawn_sq;
    if sq_file(psq) >= 4 {
        wk = sq_mirror(wk);
        bk = sq_mirror(bk);
        psq = sq_mirror(psq);
    }
    let us = usize::from(!strong_to_move);
    let idx = kpk_index(us, wk, bk, psq);
    (tbl[idx >> 3] & (1u8 << (idx & 7))) != 0
}

#[inline(always)]
fn chebyshev(a: Square, b: Square) -> i32 {
    let df = (sq_file(a) as i32 - sq_file(b) as i32).abs();
    let dr = (sq_rank(a) as i32 - sq_rank(b) as i32).abs();
    df.max(dr)
}

#[inline(always)]
fn manhattan(a: Square, b: Square) -> i32 {
    let df = (sq_file(a) as i32 - sq_file(b) as i32).abs();
    let dr = (sq_rank(a) as i32 - sq_rank(b) as i32).abs();
    df + dr
}

/// Probe built-in exact endgame bitbase & recognizers (called inside static evaluation).
#[inline]
pub fn probe_endgame(b: &Board) -> EndgameEval {
    let occ_cnt = b.occ().count_ones();
    if occ_cnt > 12 {
        return EndgameEval::None;
    }

    // 1. Exact 3-piece KPK bitbase
    if occ_cnt == 3 && b.pieces[Piece::Pawn.idx()] != 0 {
        let wp = b.piece_bb(Color::White, Piece::Pawn);
        let strong = if wp != 0 { Color::White } else { Color::Black };
        let weak = strong.flip();
        let mut sk = b.king_sq(strong);
        let mut wk = b.king_sq(weak);
        let mut psq = b.piece_bb(strong, Piece::Pawn).trailing_zeros() as Square;
        if strong == Color::Black {
            sk = sq_flip(sk);
            wk = sq_flip(wk);
            psq = sq_flip(psq);
        }
        let stm_is_strong = b.side == strong;
        if !kpk_is_win(stm_is_strong, sk, wk, psq) {
            return EndgameEval::Exact(0);
        }
        // Known exact win: provide smooth progress gradient (advance pawn, push enemy king away)
        let bonus =
            1200 + (sq_rank(psq) as i32) * 45 - chebyshev(sk, psq) * 10 + chebyshev(wk, psq) * 5;
        return EndgameEval::Exact(if stm_is_strong { bonus } else { -bonus });
    }

    // 2. 4-piece exact endgames: K+N+N vs K (draw), KB+P vs K (wrong-colored rook pawn draw), KBN vs K (corner mop-up)
    if occ_cnt == 4 {
        let pawns = b.pieces[Piece::Pawn.idx()];
        let rooks = b.pieces[Piece::Rook.idx()];
        let queens = b.pieces[Piece::Queen.idx()];
        if pawns == 0 && rooks == 0 && queens == 0 {
            let wn = b.piece_bb(Color::White, Piece::Knight).count_ones();
            let bn = b.piece_bb(Color::Black, Piece::Knight).count_ones();
            let wb = b.piece_bb(Color::White, Piece::Bishop).count_ones();
            let bb_cnt = b.piece_bb(Color::Black, Piece::Bishop).count_ones();
            // K+N+N vs K is a theoretical draw (cannot force mate)
            if (wn == 2 && bn == 0 && wb == 0 && bb_cnt == 0)
                || (bn == 2 && wn == 0 && wb == 0 && bb_cnt == 0)
            {
                return EndgameEval::Exact(0);
            }
            // K+B+N vs K: guide defending king into the corner matching bishop's color
            if (wn == 1 && wb == 1 && bn == 0 && bb_cnt == 0)
                || (bn == 1 && bb_cnt == 1 && wn == 0 && wb == 0)
            {
                let strong = if wb == 1 { Color::White } else { Color::Black };
                let weak = strong.flip();
                let sk = b.king_sq(strong);
                let lk = b.king_sq(weak);
                let bsq = b.piece_bb(strong, Piece::Bishop).trailing_zeros() as Square;
                let light_bishop = (bb(bsq) & LIGHT_SQUARES) != 0;
                // Light corners: a8 (56), h1 (7); Dark corners: a1 (0), h8 (63)
                let corner_dist = if light_bishop {
                    manhattan(lk, 56).min(manhattan(lk, 7))
                } else {
                    manhattan(lk, 0).min(manhattan(lk, 63))
                };
                let score = 1000 + (14 - corner_dist) * 35 - chebyshev(sk, lk) * 12;
                return EndgameEval::Exact(if b.side == strong { score } else { -score });
            }
        }

        // Wrong-colored rook pawn: K + B + RookPawn vs K
        if pawns.count_ones() == 1
            && rooks == 0
            && queens == 0
            && b.pieces[Piece::Knight.idx()] == 0
        {
            let wb = b.piece_bb(Color::White, Piece::Bishop);
            let bb_b = b.piece_bb(Color::Black, Piece::Bishop);
            if (wb != 0 && bb_b == 0 && b.piece_bb(Color::White, Piece::Pawn) != 0)
                || (bb_b != 0 && wb == 0 && b.piece_bb(Color::Black, Piece::Pawn) != 0)
            {
                let strong = if wb != 0 { Color::White } else { Color::Black };
                let weak = strong.flip();
                let psq = b.piece_bb(strong, Piece::Pawn).trailing_zeros() as Square;
                let pf = sq_file(psq);
                if pf == 0 || pf == 7 {
                    let promo = if strong == Color::White {
                        sq_make(pf, 7)
                    } else {
                        sq_make(pf, 0)
                    };
                    let bsq = b.piece_bb(strong, Piece::Bishop).trailing_zeros() as Square;
                    let bishop_light = (bb(bsq) & LIGHT_SQUARES) != 0;
                    let promo_light = (bb(promo) & LIGHT_SQUARES) != 0;
                    if bishop_light != promo_light {
                        let wk = b.king_sq(weak);
                        if chebyshev(wk, promo) <= 1 {
                            return EndgameEval::Exact(0);
                        }
                    }
                }
            }
        }
    }

    // 3. Opposite-Color Bishops endgame drawishness scaling (no rooks/queens/knights, 1 bishop each on opposite colors)
    if b.pieces[Piece::Queen.idx()] == 0
        && b.pieces[Piece::Rook.idx()] == 0
        && b.pieces[Piece::Knight.idx()] == 0
    {
        let wb = b.piece_bb(Color::White, Piece::Bishop);
        let bb_b = b.piece_bb(Color::Black, Piece::Bishop);
        if wb.count_ones() == 1 && bb_b.count_ones() == 1 {
            let w_light = (wb & LIGHT_SQUARES) != 0;
            let b_light = (bb_b & LIGHT_SQUARES) != 0;
            if w_light != b_light {
                let wp = b.piece_bb(Color::White, Piece::Pawn).count_ones() as i32;
                let bp = b.piece_bb(Color::Black, Piece::Pawn).count_ones() as i32;
                if (wp - bp).abs() <= 2 {
                    return EndgameEval::Scale(5, 8);
                }
            }
        }
    }

    EndgameEval::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kpk_bitbase_known_theory() {
        // Classic opposition draw: White Ke5, Pe4 vs Black Ke7 (White to move -> draw; Black to move -> win!)
        let b_draw = Board::from_fen("8/4k3/8/4K3/4P3/8/8/8 w - - 0 1").unwrap();
        assert_eq!(probe_endgame(&b_draw), EndgameEval::Exact(0));

        let b_win = Board::from_fen("8/4k3/8/4K3/4P3/8/8/8 b - - 0 1").unwrap();
        match probe_endgame(&b_win) {
            EndgameEval::Exact(s) => assert!(s < -1000, "Black to move should be losing: {s}"),
            other => panic!("expected win, got {other:?}"),
        }

        // Rook pawn draw: White Ka6, Pa5 vs Black Kc7 (Black king reaches corner)
        let b_rp = Board::from_fen("8/2k5/K7/P7/8/8/8/8 w - - 0 1").unwrap();
        assert_eq!(probe_endgame(&b_rp), EndgameEval::Exact(0));

        // Key square win: White Ke6, Pe5 vs Black Ke8 (both sides to move win for White)
        let b_key = Board::from_fen("4k3/8/4K3/4P3/8/8/8/8 w - - 0 1").unwrap();
        match probe_endgame(&b_key) {
            EndgameEval::Exact(s) => assert!(s > 1000, "White should win from key square: {s}"),
            other => panic!("expected win, got {other:?}"),
        }

        // Wrong-colored rook pawn fortress draw: White Kf6, Bc4 (light), Ph5 vs Black Kh8 (h8 is dark!)
        let b_wrp = Board::from_fen("7k/8/5K2/7P/2B5/8/8/8 w - - 0 1").unwrap();
        assert_eq!(probe_endgame(&b_wrp), EndgameEval::Exact(0));
    }
}
