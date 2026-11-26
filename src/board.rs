//! Copy-able bitboard chess Board, full rules, strict legal move generation, FEN, SAN, UCI, perft.

use crate::bitboard::{
    bb, between_bb, bishop_attacks, king_attacks, knight_attacks, line_bb, pawn_attacks, pop_lsb,
    queen_attacks, rook_attacks, NOT_FILE_A, NOT_FILE_H, RANK_1, RANK_2, RANK_3, RANK_6, RANK_7,
    RANK_8,
};
use crate::types::{
    parse_sq, sq_file, sq_make, sq_name, sq_rank, Color, DrawReason, Move, MoveList, Outcome,
    Piece, Square, A1, A8, B1, B8, C1, C8, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ, D1, D8, E1,
    E8, F1, F8, FLAG_CAPTURE, FLAG_CASTLE_K, FLAG_CASTLE_Q, FLAG_DOUBLE_PUSH, FLAG_EP,
    FLAG_PROMO_B, FLAG_PROMO_CAP_B, FLAG_PROMO_CAP_N, FLAG_PROMO_CAP_Q, FLAG_PROMO_CAP_R,
    FLAG_PROMO_N, FLAG_PROMO_Q, FLAG_PROMO_R, FLAG_QUIET, G1, G8, H1, H8, NO_SQ,
};
use crate::zobrist::ZOBRIST;

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub const EMPTY_MAILBOX: u8 = 255;

/// Castling mask per square: moving from or to a square clears the corresponding bits.
const CASTLING_MASK: [u8; 64] = {
    let mut m = [15u8; 64];
    m[A1 as usize] = 15 & !CASTLE_WQ;
    m[E1 as usize] = 15 & !(CASTLE_WK | CASTLE_WQ);
    m[H1 as usize] = 15 & !CASTLE_WK;
    m[A8 as usize] = 15 & !CASTLE_BQ;
    m[E8 as usize] = 15 & !(CASTLE_BK | CASTLE_BQ);
    m[H8 as usize] = 15 & !CASTLE_BK;
    m
};

#[inline(always)]
pub const fn mb_pack(c: Color, p: Piece) -> u8 {
    (c as u8) * 6 + (p as u8)
}

#[inline(always)]
pub const fn mb_color(v: u8) -> Color {
    if v < 6 {
        Color::White
    } else {
        Color::Black
    }
}

#[inline(always)]
pub const fn mb_piece(v: u8) -> Piece {
    Piece::from_u8(if v < 6 { v } else { v - 6 })
}

/// Complete chess position state, Copy (168 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    pub pieces: [u64; 6],
    pub colors: [u64; 2],
    pub mailbox: [u8; 64],
    pub side: Color,
    pub castling: u8,
    pub ep_sq: Square,
    pub halfmove: u8,
    pub fullmove: u16,
    pub hash: u64,
    pub pawn_hash: u64,
    pub non_pawn_hash: [u64; 2],
    pub minor_hash: u64,
    pub major_hash: u64,
}

impl Board {
    pub fn empty() -> Self {
        Self {
            pieces: [0; 6],
            colors: [0; 2],
            mailbox: [EMPTY_MAILBOX; 64],
            side: Color::White,
            castling: 0,
            ep_sq: NO_SQ,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            pawn_hash: 0,
            non_pawn_hash: [0; 2],
            minor_hash: 0,
            major_hash: 0,
        }
    }

    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS_FEN).expect("valid startpos FEN")
    }

    #[inline(always)]
    pub fn occ(&self) -> u64 {
        self.colors[0] | self.colors[1]
    }

    #[inline(always)]
    pub fn piece_bb(&self, c: Color, p: Piece) -> u64 {
        self.colors[c.idx()] & self.pieces[p.idx()]
    }

    #[inline(always)]
    pub fn king_sq(&self, c: Color) -> Square {
        self.piece_bb(c, Piece::King).trailing_zeros() as Square
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: Square) -> Option<(Color, Piece)> {
        let v = self.mailbox[sq as usize];
        if v == EMPTY_MAILBOX {
            None
        } else {
            Some((mb_color(v), mb_piece(v)))
        }
    }

    #[inline(always)]
    pub fn put_piece(&mut self, c: Color, p: Piece, sq: Square) {
        let b = bb(sq);
        self.pieces[p.idx()] |= b;
        self.colors[c.idx()] |= b;
        self.mailbox[sq as usize] = mb_pack(c, p);
        let k = ZOBRIST.piece[c.idx()][p.idx()][sq as usize];
        self.hash ^= k;
        match p {
            Piece::Pawn => self.pawn_hash ^= k,
            Piece::Knight | Piece::Bishop => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.minor_hash ^= k;
            }
            Piece::Rook | Piece::Queen => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.major_hash ^= k;
            }
            Piece::King => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.minor_hash ^= k;
                self.major_hash ^= k;
            }
        }
    }

    #[inline(always)]
    pub fn remove_piece(&mut self, c: Color, p: Piece, sq: Square) {
        let b = bb(sq);
        self.pieces[p.idx()] ^= b;
        self.colors[c.idx()] ^= b;
        self.mailbox[sq as usize] = EMPTY_MAILBOX;
        let k = ZOBRIST.piece[c.idx()][p.idx()][sq as usize];
        self.hash ^= k;
        match p {
            Piece::Pawn => self.pawn_hash ^= k,
            Piece::Knight | Piece::Bishop => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.minor_hash ^= k;
            }
            Piece::Rook | Piece::Queen => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.major_hash ^= k;
            }
            Piece::King => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.minor_hash ^= k;
                self.major_hash ^= k;
            }
        }
    }

    #[inline(always)]
    pub fn move_piece(&mut self, c: Color, p: Piece, from: Square, to: Square) {
        let b = bb(from) | bb(to);
        self.pieces[p.idx()] ^= b;
        self.colors[c.idx()] ^= b;
        self.mailbox[from as usize] = EMPTY_MAILBOX;
        self.mailbox[to as usize] = mb_pack(c, p);
        let k = ZOBRIST.piece[c.idx()][p.idx()][from as usize]
            ^ ZOBRIST.piece[c.idx()][p.idx()][to as usize];
        self.hash ^= k;
        match p {
            Piece::Pawn => self.pawn_hash ^= k,
            Piece::Knight | Piece::Bishop => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.minor_hash ^= k;
            }
            Piece::Rook | Piece::Queen => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.major_hash ^= k;
            }
            Piece::King => {
                self.non_pawn_hash[c.idx()] ^= k;
                self.minor_hash ^= k;
                self.major_hash ^= k;
            }
        }
    }

    pub fn compute_hash(&self) -> u64 {
        let mut h = 0u64;
        for sq in 0..64 {
            let v = self.mailbox[sq];
            if v != EMPTY_MAILBOX {
                h ^= ZOBRIST.piece[mb_color(v).idx()][mb_piece(v).idx()][sq];
            }
        }
        if self.side == Color::Black {
            h ^= ZOBRIST.side;
        }
        h ^= ZOBRIST.castling[self.castling as usize];
        if self.ep_sq < 64 {
            h ^= ZOBRIST.ep_file[sq_file(self.ep_sq) as usize];
        }
        h
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("FEN needs at least 4 fields: {fen}"));
        }
        let mut b = Self::empty();
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(format!("FEN board needs 8 ranks: {}", parts[0]));
        }
        for (ri, rstr) in ranks.iter().enumerate() {
            let rank = (7 - ri) as u8;
            let mut file = 0u8;
            for ch in rstr.chars() {
                if let Some(d) = ch.to_digit(10) {
                    file += d as u8;
                } else if let Some((c, p)) = Piece::from_char(ch) {
                    if file >= 8 {
                        return Err(format!("rank overflow in FEN: {rstr}"));
                    }
                    b.put_piece(c, p, sq_make(file, rank));
                    file += 1;
                } else {
                    return Err(format!("invalid FEN char: {ch}"));
                }
            }
            if file != 8 {
                return Err(format!("rank {rank} has {file} files in FEN"));
            }
        }
        b.side = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            x => return Err(format!("invalid side: {x}")),
        };
        if parts[2] != "-" {
            for ch in parts[2].chars() {
                match ch {
                    'K' => b.castling |= CASTLE_WK,
                    'Q' => b.castling |= CASTLE_WQ,
                    'k' => b.castling |= CASTLE_BK,
                    'q' => b.castling |= CASTLE_BQ,
                    _ => return Err(format!("invalid castling char: {ch}")),
                }
            }
        }
        b.ep_sq = if parts[3] == "-" {
            NO_SQ
        } else {
            parse_sq(parts[3]).ok_or_else(|| format!("invalid ep sq: {}", parts[3]))?
        };
        b.halfmove = if parts.len() >= 5 {
            parts[4].parse().unwrap_or(0)
        } else {
            0
        };
        b.fullmove = if parts.len() >= 6 {
            parts[5].parse().unwrap_or(1)
        } else {
            1
        };
        b.hash = b.compute_hash();
        Ok(b)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::with_capacity(80);
        for ri in (0..8).rev() {
            let mut empty = 0;
            for fi in 0..8 {
                let sq = sq_make(fi, ri);
                if let Some((c, p)) = self.piece_at(sq) {
                    if empty > 0 {
                        s.push(char::from_digit(empty, 10).unwrap());
                        empty = 0;
                    }
                    s.push(if c == Color::White {
                        p.char_upper()
                    } else {
                        p.char_lower()
                    });
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                s.push(char::from_digit(empty, 10).unwrap());
            }
            if ri > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.side == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            if self.castling & CASTLE_WK != 0 {
                s.push('K');
            }
            if self.castling & CASTLE_WQ != 0 {
                s.push('Q');
            }
            if self.castling & CASTLE_BK != 0 {
                s.push('k');
            }
            if self.castling & CASTLE_BQ != 0 {
                s.push('q');
            }
        }
        s.push(' ');
        s.push_str(&sq_name(self.ep_sq));
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    /// Returns bitboard of all pieces of `attacker` attacking `sq` with given `occ`.
    #[inline(always)]
    pub fn attackers_to_occ(&self, sq: Square, occ: u64, attacker: Color) -> u64 {
        let c = self.colors[attacker.idx()];
        let pawns = c & self.pieces[Piece::Pawn.idx()];
        let knights = c & self.pieces[Piece::Knight.idx()];
        let kings = c & self.pieces[Piece::King.idx()];
        let bq = c & (self.pieces[Piece::Bishop.idx()] | self.pieces[Piece::Queen.idx()]);
        let rq = c & (self.pieces[Piece::Rook.idx()] | self.pieces[Piece::Queen.idx()]);
        (pawn_attacks(attacker.flip(), sq) & pawns)
            | (knight_attacks(sq) & knights)
            | (king_attacks(sq) & kings)
            | (bishop_attacks(sq, occ) & bq)
            | (rook_attacks(sq, occ) & rq)
    }

    /// Returns bitboard of ALL pieces (both colors) attacking `sq` with given `occ`.
    #[inline(always)]
    pub fn all_attackers_to(&self, sq: Square, occ: u64) -> u64 {
        let wp = self.piece_bb(Color::White, Piece::Pawn);
        let bp = self.piece_bb(Color::Black, Piece::Pawn);
        let bq = self.pieces[Piece::Bishop.idx()] | self.pieces[Piece::Queen.idx()];
        let rq = self.pieces[Piece::Rook.idx()] | self.pieces[Piece::Queen.idx()];
        (pawn_attacks(Color::Black, sq) & wp)
            | (pawn_attacks(Color::White, sq) & bp)
            | (knight_attacks(sq) & self.pieces[Piece::Knight.idx()])
            | (king_attacks(sq) & self.pieces[Piece::King.idx()])
            | (bishop_attacks(sq, occ) & bq)
            | (rook_attacks(sq, occ) & rq)
    }

    #[inline(always)]
    pub fn is_sq_attacked(&self, sq: Square, occ: u64, by: Color) -> bool {
        let c = self.colors[by.idx()];
        if pawn_attacks(by.flip(), sq) & c & self.pieces[Piece::Pawn.idx()] != 0 {
            return true;
        }
        if knight_attacks(sq) & c & self.pieces[Piece::Knight.idx()] != 0 {
            return true;
        }
        if king_attacks(sq) & c & self.pieces[Piece::King.idx()] != 0 {
            return true;
        }
        let bq = c & (self.pieces[Piece::Bishop.idx()] | self.pieces[Piece::Queen.idx()]);
        if bq != 0 && bishop_attacks(sq, occ) & bq != 0 {
            return true;
        }
        let rq = c & (self.pieces[Piece::Rook.idx()] | self.pieces[Piece::Queen.idx()]);
        if rq != 0 && rook_attacks(sq, occ) & rq != 0 {
            return true;
        }
        false
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.is_sq_attacked(self.king_sq(self.side), self.occ(), self.side.flip())
    }

    /// Bitboard of all squares attacked by `by` (fast threat map for threat-aware history & pruning).
    #[inline]
    pub fn attacks_by(&self, by: Color) -> u64 {
        let c = self.colors[by.idx()];
        let occ = self.occ();
        let pawns = c & self.pieces[Piece::Pawn.idx()];
        let mut att = if by == Color::White {
            ((pawns & NOT_FILE_A) << 7) | ((pawns & NOT_FILE_H) << 9)
        } else {
            ((pawns & NOT_FILE_A) >> 9) | ((pawns & NOT_FILE_H) >> 7)
        };
        let kbb = c & self.pieces[Piece::King.idx()];
        if kbb != 0 {
            att |= king_attacks(kbb.trailing_zeros() as Square);
        }
        let mut knights = c & self.pieces[Piece::Knight.idx()];
        while knights != 0 {
            att |= knight_attacks(pop_lsb(&mut knights));
        }
        let mut bq = c & (self.pieces[Piece::Bishop.idx()] | self.pieces[Piece::Queen.idx()]);
        while bq != 0 {
            att |= bishop_attacks(pop_lsb(&mut bq), occ);
        }
        let mut rq = c & (self.pieces[Piece::Rook.idx()] | self.pieces[Piece::Queen.idx()]);
        while rq != 0 {
            att |= rook_attacks(pop_lsb(&mut rq), occ);
        }
        att
    }

    /// Bitboard of all squares attacked by the opponent (`self.side.flip()`).
    #[inline(always)]
    pub fn enemy_attacks(&self) -> u64 {
        self.attacks_by(self.side.flip())
    }

    /// Precompute direct-check target squares for each piece type (P, N, B, R, Q) against opponent's king.
    #[inline]
    pub fn check_squares(&self) -> [u64; 5] {
        let them = self.side.flip();
        let opp_k = self.king_sq(them);
        let occ = self.occ();
        let b_att = bishop_attacks(opp_k, occ);
        let r_att = rook_attacks(opp_k, occ);
        [
            pawn_attacks(them, opp_k),
            knight_attacks(opp_k),
            b_att,
            r_att,
            b_att | r_att,
        ]
    }

    /// Fast check whether a legal move `m` delivers check (handles direct checks fast, falls back for discovered/special).
    #[inline]
    pub fn gives_check_fast(&self, m: Move, chk_sq: &[u64; 5], dc_candidates: u64) -> bool {
        let from = m.from();
        let to = m.to();
        let flag = m.flag();
        if flag == FLAG_QUIET || flag == FLAG_DOUBLE_PUSH || flag == FLAG_CAPTURE {
            let p = mb_piece(self.mailbox[from as usize]).idx();
            if p < 5 && (chk_sq[p] & bb(to)) != 0 {
                return true;
            }
            if (dc_candidates & bb(from)) == 0 {
                return false;
            }
        }
        self.make_move(m).in_check()
    }

    /// Bitboard of our pieces that could reveal a discovered check on the opponent's king if moved.
    #[inline]
    pub fn discovered_check_candidates(&self) -> u64 {
        let us = self.side;
        let opp_k = self.king_sq(us.flip());
        let our_bb = self.colors[us.idx()];
        let their_bb = self.colors[us.flip().idx()];
        let bq = our_bb & (self.pieces[Piece::Bishop.idx()] | self.pieces[Piece::Queen.idx()]);
        let rq = our_bb & (self.pieces[Piece::Rook.idx()] | self.pieces[Piece::Queen.idx()]);
        let mut dc = 0u64;
        let mut snipers =
            (bishop_attacks(opp_k, their_bb) & bq) | (rook_attacks(opp_k, their_bb) & rq);
        while snipers != 0 {
            let s = pop_lsb(&mut snipers);
            let between = between_bb(opp_k, s) & self.occ();
            if between.count_ones() == 1 && (between & our_bb) != 0 {
                dc |= between;
            }
        }
        dc
    }

    /// Verify if a TT move `m` is strictly legal in `self` without generating the full move list.
    pub fn is_legal_tt_move(&self, m: Move) -> bool {
        if m.is_none() {
            return false;
        }
        let from = m.from();
        let to = m.to();
        let v = self.mailbox[from as usize];
        if v == EMPTY_MAILBOX || mb_color(v) != self.side {
            return false;
        }
        let us = self.side;
        let them = us.flip();
        let piece = mb_piece(v);
        let flag = m.flag();
        let occ = self.occ();
        let target_v = self.mailbox[to as usize];

        // Basic destination & capture consistency
        match flag {
            FLAG_QUIET => {
                if target_v != EMPTY_MAILBOX {
                    return false;
                }
                if piece == Piece::Pawn {
                    let expected = if us == Color::White {
                        from.wrapping_add(8)
                    } else {
                        from.wrapping_sub(8)
                    };
                    if to != expected || (bb(to) & (RANK_1 | RANK_8)) != 0 {
                        return false;
                    }
                }
            }
            FLAG_DOUBLE_PUSH => {
                if piece != Piece::Pawn || target_v != EMPTY_MAILBOX {
                    return false;
                }
                let mid = if us == Color::White {
                    from + 8
                } else {
                    from.wrapping_sub(8)
                };
                let exp_to = if us == Color::White {
                    from + 16
                } else {
                    from.wrapping_sub(16)
                };
                let start_rank = if us == Color::White { RANK_2 } else { RANK_7 };
                if (bb(from) & start_rank) == 0 || to != exp_to || (occ & bb(mid)) != 0 {
                    return false;
                }
            }
            FLAG_CAPTURE => {
                if target_v == EMPTY_MAILBOX
                    || mb_color(target_v) != them
                    || mb_piece(target_v) == Piece::King
                {
                    return false;
                }
                if piece == Piece::Pawn {
                    if (pawn_attacks(us, from) & bb(to)) == 0 || (bb(to) & (RANK_1 | RANK_8)) != 0 {
                        return false;
                    }
                }
            }
            FLAG_CASTLE_K | FLAG_CASTLE_Q => {
                if piece != Piece::King || self.in_check() {
                    return false;
                }
                if us == Color::White {
                    if flag == FLAG_CASTLE_K {
                        return from == E1
                            && to == G1
                            && (self.castling & CASTLE_WK) != 0
                            && (occ & (bb(F1) | bb(G1))) == 0
                            && !self.is_sq_attacked(F1, occ, them)
                            && !self.is_sq_attacked(G1, occ, them);
                    } else {
                        return from == E1
                            && to == C1
                            && (self.castling & CASTLE_WQ) != 0
                            && (occ & (bb(B1) | bb(C1) | bb(D1))) == 0
                            && !self.is_sq_attacked(D1, occ, them)
                            && !self.is_sq_attacked(C1, occ, them);
                    }
                } else if flag == FLAG_CASTLE_K {
                    return from == E8
                        && to == G8
                        && (self.castling & CASTLE_BK) != 0
                        && (occ & (bb(F8) | bb(G8))) == 0
                        && !self.is_sq_attacked(F8, occ, them)
                        && !self.is_sq_attacked(G8, occ, them);
                } else {
                    return from == E8
                        && to == C8
                        && (self.castling & CASTLE_BQ) != 0
                        && (occ & (bb(B8) | bb(C8) | bb(D8))) == 0
                        && !self.is_sq_attacked(D8, occ, them)
                        && !self.is_sq_attacked(C8, occ, them);
                }
            }
            FLAG_EP => {
                if piece != Piece::Pawn
                    || to != self.ep_sq
                    || (pawn_attacks(us, from) & bb(to)) == 0
                {
                    return false;
                }
            }
            FLAG_PROMO_N | FLAG_PROMO_B | FLAG_PROMO_R | FLAG_PROMO_Q => {
                if piece != Piece::Pawn || target_v != EMPTY_MAILBOX {
                    return false;
                }
                let expected = if us == Color::White {
                    from.wrapping_add(8)
                } else {
                    from.wrapping_sub(8)
                };
                if to != expected || (bb(to) & (RANK_1 | RANK_8)) == 0 {
                    return false;
                }
            }
            FLAG_PROMO_CAP_N | FLAG_PROMO_CAP_B | FLAG_PROMO_CAP_R | FLAG_PROMO_CAP_Q => {
                if piece != Piece::Pawn
                    || target_v == EMPTY_MAILBOX
                    || mb_color(target_v) != them
                    || mb_piece(target_v) == Piece::King
                {
                    return false;
                }
                if (pawn_attacks(us, from) & bb(to)) == 0 || (bb(to) & (RANK_1 | RANK_8)) == 0 {
                    return false;
                }
            }
            _ => return false,
        }

        // Piece movement geometry for non-pawns (castling already returned above)
        match piece {
            Piece::Pawn => {}
            Piece::Knight => {
                if (knight_attacks(from) & bb(to)) == 0 {
                    return false;
                }
            }
            Piece::Bishop => {
                if (bishop_attacks(from, occ) & bb(to)) == 0 {
                    return false;
                }
            }
            Piece::Rook => {
                if (rook_attacks(from, occ) & bb(to)) == 0 {
                    return false;
                }
            }
            Piece::Queen => {
                if (queen_attacks(from, occ) & bb(to)) == 0 {
                    return false;
                }
            }
            Piece::King => {
                if (king_attacks(from) & bb(to)) == 0 {
                    return false;
                }
            }
        }

        // Final king-safety legality check
        let next = self.make_move(m);
        !next.is_sq_attacked(next.king_sq(us), next.occ(), them)
    }

    /// Returns `(checkers, pinned)` for `self.side`.
    #[inline]
    pub fn checkers_and_pinned(&self) -> (u64, u64) {
        let us = self.side;
        let them = us.flip();
        let ksq = self.king_sq(us);
        let occ = self.occ();
        let our_bb = self.colors[us.idx()];
        let their_bb = self.colors[them.idx()];

        let bq = their_bb & (self.pieces[Piece::Bishop.idx()] | self.pieces[Piece::Queen.idx()]);
        let rq = their_bb & (self.pieces[Piece::Rook.idx()] | self.pieces[Piece::Queen.idx()]);

        let checkers = (pawn_attacks(us, ksq) & their_bb & self.pieces[Piece::Pawn.idx()])
            | (knight_attacks(ksq) & their_bb & self.pieces[Piece::Knight.idx()])
            | (bishop_attacks(ksq, occ) & bq)
            | (rook_attacks(ksq, occ) & rq);

        let mut pinned = 0u64;
        let mut snipers = (bishop_attacks(ksq, their_bb) & bq) | (rook_attacks(ksq, their_bb) & rq);
        while snipers != 0 {
            let s = pop_lsb(&mut snipers);
            let between = between_bb(ksq, s) & our_bb;
            if between.count_ones() == 1 {
                pinned |= between;
            }
        }
        (checkers, pinned)
    }

    /// Generate all strictly legal moves.
    pub fn legal_moves(&self, ml: &mut MoveList) {
        ml.clear();
        self.gen_moves::<0>(ml);
    }

    /// Generate strictly legal tactical moves (captures + promotions). If in check, generates all legal evasions.
    pub fn tactical_moves(&self, ml: &mut MoveList) {
        ml.clear();
        self.gen_moves::<1>(ml);
    }

    /// Generate strictly legal quiet moves (non-captures, non-promotions; assumes not in check).
    pub fn quiet_moves(&self, ml: &mut MoveList) {
        ml.clear();
        self.gen_moves::<2>(ml);
    }

    /// MODE: 0 = ALL, 1 = TACTICAL (all evasions if in check), 2 = QUIET_ONLY (not in check).
    #[inline]
    fn gen_moves<const MODE: u8>(&self, ml: &mut MoveList) {
        let us = self.side;
        let them = us.flip();
        let ksq = self.king_sq(us);
        let our_bb = self.colors[us.idx()];
        let their_bb = self.colors[them.idx()];
        let occ = our_bb | their_bb;
        let (checkers, pinned) = self.checkers_and_pinned();
        let num_checks = checkers.count_ones();
        let in_check = num_checks > 0;
        let tactical_allowed = MODE != 2;
        let quiet_allowed = MODE != 1 || in_check;

        // 1. King moves (remove our king from occ so sliders see through it)
        let occ_no_king = occ ^ bb(ksq);
        let mut k_targets = king_attacks(ksq) & !our_bb;
        if !quiet_allowed {
            k_targets &= their_bb;
        } else if !tactical_allowed {
            k_targets &= !their_bb;
        }
        while k_targets != 0 {
            let to = pop_lsb(&mut k_targets);
            if !self.is_sq_attacked(to, occ_no_king, them) {
                let flag = if bb(to) & their_bb != 0 {
                    FLAG_CAPTURE
                } else {
                    FLAG_QUIET
                };
                ml.push(Move::new(ksq, to, flag));
            }
        }

        // Castling (only when not in check and quiet moves allowed)
        if !in_check && quiet_allowed {
            if us == Color::White {
                if self.castling & CASTLE_WK != 0
                    && (occ & (bb(F1) | bb(G1))) == 0
                    && !self.is_sq_attacked(F1, occ, them)
                    && !self.is_sq_attacked(G1, occ, them)
                {
                    ml.push(Move::new(E1, G1, FLAG_CASTLE_K));
                }
                if self.castling & CASTLE_WQ != 0
                    && (occ & (bb(B1) | bb(C1) | bb(D1))) == 0
                    && !self.is_sq_attacked(D1, occ, them)
                    && !self.is_sq_attacked(C1, occ, them)
                {
                    ml.push(Move::new(E1, C1, FLAG_CASTLE_Q));
                }
            } else {
                if self.castling & CASTLE_BK != 0
                    && (occ & (bb(F8) | bb(G8))) == 0
                    && !self.is_sq_attacked(F8, occ, them)
                    && !self.is_sq_attacked(G8, occ, them)
                {
                    ml.push(Move::new(E8, G8, FLAG_CASTLE_K));
                }
                if self.castling & CASTLE_BQ != 0
                    && (occ & (bb(B8) | bb(C8) | bb(D8))) == 0
                    && !self.is_sq_attacked(D8, occ, them)
                    && !self.is_sq_attacked(C8, occ, them)
                {
                    ml.push(Move::new(E8, C8, FLAG_CASTLE_Q));
                }
            }
        }

        // Double check: only king moves are legal.
        if num_checks >= 2 {
            return;
        }

        // Target mask for non-king pieces: all squares if not in check, else checker + block ray.
        let check_mask = if num_checks == 1 {
            let csq = checkers.trailing_zeros() as Square;
            checkers | between_bb(ksq, csq)
        } else {
            !0u64
        };

        // 2. Pawns
        let pawns = our_bb & self.pieces[Piece::Pawn.idx()];
        let empty = !occ;
        let promo_rank = if us == Color::White { RANK_7 } else { RANK_2 };
        let non_promo_pawns = pawns & !promo_rank;
        let promo_pawns = pawns & promo_rank;

        // 2a. Non-promoting pawn pushes (only when quiet allowed)
        if quiet_allowed {
            let mut single = if us == Color::White {
                (non_promo_pawns << 8) & empty
            } else {
                (non_promo_pawns >> 8) & empty
            };
            let mut double = if us == Color::White {
                ((single & RANK_3) << 8) & empty & check_mask
            } else {
                ((single & RANK_6) >> 8) & empty & check_mask
            };
            single &= check_mask;

            while single != 0 {
                let to = pop_lsb(&mut single);
                let from = if us == Color::White { to - 8 } else { to + 8 };
                if bb(from) & pinned == 0 || bb(to) & line_bb(ksq, from) != 0 {
                    ml.push(Move::new(from, to, FLAG_QUIET));
                }
            }
            while double != 0 {
                let to = pop_lsb(&mut double);
                let from = if us == Color::White { to - 16 } else { to + 16 };
                if bb(from) & pinned == 0 || bb(to) & line_bb(ksq, from) != 0 {
                    ml.push(Move::new(from, to, FLAG_DOUBLE_PUSH));
                }
            }
        }

        if tactical_allowed {
            // 2b. Non-promoting pawn captures
            let cap_mask = their_bb & check_mask;
            let (mut left_caps, mut right_caps) = if us == Color::White {
                (
                    ((non_promo_pawns & NOT_FILE_A) << 7) & cap_mask,
                    ((non_promo_pawns & NOT_FILE_H) << 9) & cap_mask,
                )
            } else {
                (
                    ((non_promo_pawns & NOT_FILE_A) >> 9) & cap_mask,
                    ((non_promo_pawns & NOT_FILE_H) >> 7) & cap_mask,
                )
            };
            while left_caps != 0 {
                let to = pop_lsb(&mut left_caps);
                let from = if us == Color::White { to - 7 } else { to + 9 };
                if bb(from) & pinned == 0 || bb(to) & line_bb(ksq, from) != 0 {
                    ml.push(Move::new(from, to, FLAG_CAPTURE));
                }
            }
            while right_caps != 0 {
                let to = pop_lsb(&mut right_caps);
                let from = if us == Color::White { to - 9 } else { to + 7 };
                if bb(from) & pinned == 0 || bb(to) & line_bb(ksq, from) != 0 {
                    ml.push(Move::new(from, to, FLAG_CAPTURE));
                }
            }

            // 2c. Promoting pawns (all 4 promotion pieces so tactical ∪ quiet == legal)
            if promo_pawns != 0 {
                let mut push = if us == Color::White {
                    (promo_pawns << 8) & empty & check_mask
                } else {
                    (promo_pawns >> 8) & empty & check_mask
                };
                while push != 0 {
                    let to = pop_lsb(&mut push);
                    let from = if us == Color::White { to - 8 } else { to + 8 };
                    if bb(from) & pinned == 0 || bb(to) & line_bb(ksq, from) != 0 {
                        ml.push(Move::new(from, to, FLAG_PROMO_Q));
                        ml.push(Move::new(from, to, FLAG_PROMO_R));
                        ml.push(Move::new(from, to, FLAG_PROMO_B));
                        ml.push(Move::new(from, to, FLAG_PROMO_N));
                    }
                }
                let (mut lcap, mut rcap) = if us == Color::White {
                    (
                        ((promo_pawns & NOT_FILE_A) << 7) & cap_mask,
                        ((promo_pawns & NOT_FILE_H) << 9) & cap_mask,
                    )
                } else {
                    (
                        ((promo_pawns & NOT_FILE_A) >> 9) & cap_mask,
                        ((promo_pawns & NOT_FILE_H) >> 7) & cap_mask,
                    )
                };
                while lcap != 0 {
                    let to = pop_lsb(&mut lcap);
                    let from = if us == Color::White { to - 7 } else { to + 9 };
                    if bb(from) & pinned == 0 || bb(to) & line_bb(ksq, from) != 0 {
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_Q));
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_N));
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_R));
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_B));
                    }
                }
                while rcap != 0 {
                    let to = pop_lsb(&mut rcap);
                    let from = if us == Color::White { to - 9 } else { to + 7 };
                    if bb(from) & pinned == 0 || bb(to) & line_bb(ksq, from) != 0 {
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_Q));
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_N));
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_R));
                        ml.push(Move::new(from, to, FLAG_PROMO_CAP_B));
                    }
                }
            }

            // 2d. En passant (always checked with explicit occupancy update for horizontal pins)
            if self.ep_sq < 64 {
                let ep_to = self.ep_sq;
                let cap_sq = if us == Color::White {
                    ep_to - 8
                } else {
                    ep_to + 8
                };
                let mut ep_pawns = pawn_attacks(them, ep_to) & pawns;
                while ep_pawns != 0 {
                    let from = pop_lsb(&mut ep_pawns);
                    let new_occ = (occ ^ bb(from) ^ bb(cap_sq)) | bb(ep_to);
                    let bq = their_bb
                        & (self.pieces[Piece::Bishop.idx()] | self.pieces[Piece::Queen.idx()]);
                    let rq = their_bb
                        & (self.pieces[Piece::Rook.idx()] | self.pieces[Piece::Queen.idx()]);
                    let p_att = pawn_attacks(us, ksq)
                        & ((their_bb & self.pieces[Piece::Pawn.idx()]) ^ bb(cap_sq));
                    let n_att = knight_attacks(ksq) & their_bb & self.pieces[Piece::Knight.idx()];
                    let in_chk = (bishop_attacks(ksq, new_occ) & bq) != 0
                        || (rook_attacks(ksq, new_occ) & rq) != 0
                        || p_att != 0
                        || n_att != 0;
                    if !in_chk {
                        ml.push(Move::new(from, ep_to, FLAG_EP));
                    }
                }
            }
        }

        let target_mask = if !tactical_allowed {
            empty & check_mask
        } else if quiet_allowed {
            !our_bb & check_mask
        } else {
            their_bb & check_mask
        };

        // 3. Knights (pinned knights can never move)
        let mut knights = our_bb & self.pieces[Piece::Knight.idx()] & !pinned;
        while knights != 0 {
            let from = pop_lsb(&mut knights);
            let mut att = knight_attacks(from) & target_mask;
            while att != 0 {
                let to = pop_lsb(&mut att);
                let flag = if bb(to) & their_bb != 0 {
                    FLAG_CAPTURE
                } else {
                    FLAG_QUIET
                };
                ml.push(Move::new(from, to, flag));
            }
        }

        // 4. Bishops
        let mut bishops = our_bb & self.pieces[Piece::Bishop.idx()];
        while bishops != 0 {
            let from = pop_lsb(&mut bishops);
            let mut att = bishop_attacks(from, occ) & target_mask;
            if bb(from) & pinned != 0 {
                att &= line_bb(ksq, from);
            }
            while att != 0 {
                let to = pop_lsb(&mut att);
                let flag = if bb(to) & their_bb != 0 {
                    FLAG_CAPTURE
                } else {
                    FLAG_QUIET
                };
                ml.push(Move::new(from, to, flag));
            }
        }

        // 5. Rooks
        let mut rooks = our_bb & self.pieces[Piece::Rook.idx()];
        while rooks != 0 {
            let from = pop_lsb(&mut rooks);
            let mut att = rook_attacks(from, occ) & target_mask;
            if bb(from) & pinned != 0 {
                att &= line_bb(ksq, from);
            }
            while att != 0 {
                let to = pop_lsb(&mut att);
                let flag = if bb(to) & their_bb != 0 {
                    FLAG_CAPTURE
                } else {
                    FLAG_QUIET
                };
                ml.push(Move::new(from, to, flag));
            }
        }

        // 6. Queens
        let mut queens = our_bb & self.pieces[Piece::Queen.idx()];
        while queens != 0 {
            let from = pop_lsb(&mut queens);
            let mut att = queen_attacks(from, occ) & target_mask;
            if bb(from) & pinned != 0 {
                att &= line_bb(ksq, from);
            }
            while att != 0 {
                let to = pop_lsb(&mut att);
                let flag = if bb(to) & their_bb != 0 {
                    FLAG_CAPTURE
                } else {
                    FLAG_QUIET
                };
                ml.push(Move::new(from, to, flag));
            }
        }
    }

    /// Apply a legal move to a copy of the board (copy-make).
    #[inline]
    pub fn make_move(&self, m: Move) -> Board {
        let mut b = *self;
        b.apply_move(m);
        b
    }

    /// Apply a legal move in place.
    #[inline]
    pub fn apply_move(&mut self, m: Move) {
        let us = self.side;
        let them = us.flip();
        let from = m.from();
        let to = m.to();
        let flag = m.flag();

        // Clear old ep hash
        if self.ep_sq < 64 {
            self.hash ^= ZOBRIST.ep_file[sq_file(self.ep_sq) as usize];
            self.ep_sq = NO_SQ;
        }
        // Remove old castling hash
        self.hash ^= ZOBRIST.castling[self.castling as usize];

        let moved_v = self.mailbox[from as usize];
        debug_assert!(moved_v != EMPTY_MAILBOX);
        let moved_piece = mb_piece(moved_v);

        self.halfmove = if moved_piece == Piece::Pawn || m.is_capture() {
            0
        } else {
            self.halfmove.saturating_add(1)
        };

        match flag {
            FLAG_QUIET => {
                self.move_piece(us, moved_piece, from, to);
            }
            FLAG_DOUBLE_PUSH => {
                self.move_piece(us, Piece::Pawn, from, to);
                let ep = if us == Color::White {
                    from + 8
                } else {
                    from - 8
                };
                self.ep_sq = ep;
                self.hash ^= ZOBRIST.ep_file[sq_file(ep) as usize];
            }
            FLAG_CASTLE_K => {
                self.move_piece(us, Piece::King, from, to);
                if us == Color::White {
                    self.move_piece(us, Piece::Rook, H1, F1);
                } else {
                    self.move_piece(us, Piece::Rook, H8, F8);
                }
            }
            FLAG_CASTLE_Q => {
                self.move_piece(us, Piece::King, from, to);
                if us == Color::White {
                    self.move_piece(us, Piece::Rook, A1, D1);
                } else {
                    self.move_piece(us, Piece::Rook, A8, D8);
                }
            }
            FLAG_CAPTURE => {
                let cap_v = self.mailbox[to as usize];
                debug_assert!(cap_v != EMPTY_MAILBOX);
                self.remove_piece(them, mb_piece(cap_v), to);
                self.move_piece(us, moved_piece, from, to);
            }
            FLAG_EP => {
                let cap_sq = if us == Color::White { to - 8 } else { to + 8 };
                self.remove_piece(them, Piece::Pawn, cap_sq);
                self.move_piece(us, Piece::Pawn, from, to);
            }
            FLAG_PROMO_N | FLAG_PROMO_B | FLAG_PROMO_R | FLAG_PROMO_Q => {
                let promo = m.promo_piece().unwrap();
                self.remove_piece(us, Piece::Pawn, from);
                self.put_piece(us, promo, to);
            }
            FLAG_PROMO_CAP_N | FLAG_PROMO_CAP_B | FLAG_PROMO_CAP_R | FLAG_PROMO_CAP_Q => {
                let cap_v = self.mailbox[to as usize];
                debug_assert!(cap_v != EMPTY_MAILBOX);
                let promo = m.promo_piece().unwrap();
                self.remove_piece(them, mb_piece(cap_v), to);
                self.remove_piece(us, Piece::Pawn, from);
                self.put_piece(us, promo, to);
            }
            _ => unreachable!(),
        }

        self.castling &= CASTLING_MASK[from as usize] & CASTLING_MASK[to as usize];
        self.hash ^= ZOBRIST.castling[self.castling as usize];
        self.side = them;
        self.hash ^= ZOBRIST.side;
        if us == Color::Black {
            self.fullmove += 1;
        }
    }

    /// Make a null move (for Null Move Pruning in search).
    #[inline]
    pub fn make_null_move(&self) -> Board {
        let mut b = *self;
        if b.ep_sq < 64 {
            b.hash ^= ZOBRIST.ep_file[sq_file(b.ep_sq) as usize];
            b.ep_sq = NO_SQ;
        }
        b.side = b.side.flip();
        b.hash ^= ZOBRIST.side;
        b.halfmove = b.halfmove.saturating_add(1);
        b
    }

    /// True if side has any non-pawn, non-king material (used for NMP zugzwang guard).
    #[inline(always)]
    pub fn has_non_pawn_material(&self, c: Color) -> bool {
        let non_pk = self.pieces[Piece::Knight.idx()]
            | self.pieces[Piece::Bishop.idx()]
            | self.pieces[Piece::Rook.idx()]
            | self.pieces[Piece::Queen.idx()];
        (self.colors[c.idx()] & non_pk) != 0
    }

    /// Check insufficient mating material (K vs K, K+B vs K, K+N vs K, K+B vs K+B same color).
    pub fn is_insufficient_material(&self) -> bool {
        if (self.pieces[Piece::Pawn.idx()]
            | self.pieces[Piece::Rook.idx()]
            | self.pieces[Piece::Queen.idx()])
            != 0
        {
            return false;
        }
        let minors = self.pieces[Piece::Knight.idx()] | self.pieces[Piece::Bishop.idx()];
        let n = minors.count_ones();
        if n <= 1 {
            return true;
        }
        if n == 2 && self.pieces[Piece::Knight.idx()] == 0 {
            let wb = self.piece_bb(Color::White, Piece::Bishop);
            let bb_b = self.piece_bb(Color::Black, Piece::Bishop);
            if wb.count_ones() == 1 && bb_b.count_ones() == 1 {
                const LIGHT_SQ: u64 = 0x55AA_55AA_55AA_55AA;
                return ((wb & LIGHT_SQ) != 0) == ((bb_b & LIGHT_SQ) != 0);
            }
        }
        false
    }

    /// Parse a UCI move string (e.g. "e2e4", "e7e8q") against legal moves.
    pub fn parse_uci_move(&self, s: &str) -> Option<Move> {
        let s = s.trim();
        let mut ml = MoveList::new();
        self.legal_moves(&mut ml);
        for &m in ml.as_slice() {
            if m.to_uci() == s {
                return Some(m);
            }
        }
        None
    }

    /// Format a legal move as Standard Algebraic Notation (SAN), e.g., "Nf3", "exd5", "O-O", "e8=Q+".
    pub fn move_to_san(&self, m: Move) -> String {
        if m.flag() == FLAG_CASTLE_K {
            let next = self.make_move(m);
            return format!("O-O{}", next.check_suffix());
        }
        if m.flag() == FLAG_CASTLE_Q {
            let next = self.make_move(m);
            return format!("O-O-O{}", next.check_suffix());
        }
        let from = m.from();
        let to = m.to();
        let (_, piece) = self.piece_at(from).expect("valid move");
        let mut s = String::new();
        if piece == Piece::Pawn {
            if m.is_capture() {
                s.push((b'a' + sq_file(from)) as char);
                s.push('x');
            }
            s.push_str(&sq_name(to));
            if let Some(p) = m.promo_piece() {
                s.push('=');
                s.push(p.char_upper());
            }
        } else {
            s.push(piece.char_upper());
            // Disambiguation among other legal moves of the same piece to the same square.
            let mut ml = MoveList::new();
            self.legal_moves(&mut ml);
            let mut same_dest = false;
            let mut same_file = false;
            let mut same_rank = false;
            for &other in ml.as_slice() {
                if other == m || other.to() != to {
                    continue;
                }
                if let Some((_, op)) = self.piece_at(other.from()) {
                    if op == piece {
                        same_dest = true;
                        if sq_file(other.from()) == sq_file(from) {
                            same_file = true;
                        }
                        if sq_rank(other.from()) == sq_rank(from) {
                            same_rank = true;
                        }
                    }
                }
            }
            if same_dest {
                if !same_file {
                    s.push((b'a' + sq_file(from)) as char);
                } else if !same_rank {
                    s.push((b'1' + sq_rank(from)) as char);
                } else {
                    s.push_str(&sq_name(from));
                }
            }
            if m.is_capture() {
                s.push('x');
            }
            s.push_str(&sq_name(to));
        }
        let next = self.make_move(m);
        s.push_str(next.check_suffix());
        s
    }

    fn check_suffix(&self) -> &'static str {
        if !self.in_check() {
            return "";
        }
        let mut ml = MoveList::new();
        self.legal_moves(&mut ml);
        if ml.is_empty() {
            "#"
        } else {
            "+"
        }
    }

    /// Recursive perft (bulk-counting at depth 1).
    pub fn perft(&self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut ml = MoveList::new();
        self.legal_moves(&mut ml);
        if depth == 1 {
            return ml.len() as u64;
        }
        let mut nodes = 0u64;
        for &m in ml.as_slice() {
            nodes += self.make_move(m).perft(depth - 1);
        }
        nodes
    }

    /// Perft divide: returns per-move node counts at the root.
    pub fn perft_divide(&self, depth: u32) -> Vec<(Move, u64)> {
        let mut ml = MoveList::new();
        self.legal_moves(&mut ml);
        let mut out = Vec::with_capacity(ml.len());
        for &m in ml.as_slice() {
            let n = if depth <= 1 {
                1
            } else {
                self.make_move(m).perft(depth - 1)
            };
            out.push((m, n));
        }
        out
    }

    /// Pretty ASCII board display.
    pub fn pretty(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            s.push_str(&format!("{} | ", r + 1));
            for f in 0..8 {
                let sq = sq_make(f, r);
                let ch = match self.piece_at(sq) {
                    Some((Color::White, p)) => p.char_upper(),
                    Some((Color::Black, p)) => p.char_lower(),
                    None => '.',
                };
                s.push(ch);
                s.push(' ');
            }
            s.push('\n');
        }
        s.push_str("  +-----------------\n    a b c d e f g h\n");
        s.push_str(&format!("FEN: {}\n", self.to_fen()));
        s
    }
}

/// Full game state including position history for repetition detection.
#[derive(Clone, Debug)]
pub struct Game {
    pub board: Board,
    pub history: Vec<u64>,
}

impl Game {
    pub fn new() -> Self {
        Self::from_board(Board::startpos())
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        Ok(Self::from_board(Board::from_fen(fen)?))
    }

    pub fn from_board(board: Board) -> Self {
        let mut history = Vec::with_capacity(256);
        history.push(board.hash);
        Self { board, history }
    }

    pub fn apply_move(&mut self, m: Move) {
        self.board.apply_move(m);
        self.history.push(self.board.hash);
    }

    /// Count how many times the current position hash has appeared in the reversible window.
    pub fn repetition_count(&self) -> usize {
        let cur = self.board.hash;
        let len = self.history.len();
        let limit = (self.board.halfmove as usize).min(len - 1);
        let start = len - 1 - limit;
        let mut count = 0;
        let mut i = len - 1;
        loop {
            if self.history[i] == cur {
                count += 1;
            }
            if i < start + 2 {
                break;
            }
            i -= 2;
        }
        count
    }

    pub fn outcome(&self) -> Option<Outcome> {
        let mut ml = MoveList::new();
        self.board.legal_moves(&mut ml);
        if ml.is_empty() {
            if self.board.in_check() {
                return Some(Outcome::Win(self.board.side.flip()));
            } else {
                return Some(Outcome::Draw(DrawReason::Stalemate));
            }
        }
        if self.board.halfmove >= 100 {
            return Some(Outcome::Draw(DrawReason::FiftyMove));
        }
        if self.repetition_count() >= 3 {
            return Some(Outcome::Draw(DrawReason::Repetition));
        }
        if self.board.is_insufficient_material() {
            return Some(Outcome::Draw(DrawReason::InsufficientMaterial));
        }
        None
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

/// Rank constants exposed for other modules.
pub const PROMO_RANKS: u64 = RANK_1 | RANK_8;
