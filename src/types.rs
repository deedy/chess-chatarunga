//! Core chess primitives: Color, Piece, Square, Move (16-bit packed), MoveList, Outcome.

use std::fmt;

pub const MAX_MOVES: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    pub const ALL: [Piece; 6] = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn from_u8(v: u8) -> Piece {
        match v {
            0 => Piece::Pawn,
            1 => Piece::Knight,
            2 => Piece::Bishop,
            3 => Piece::Rook,
            4 => Piece::Queen,
            _ => Piece::King,
        }
    }

    pub const fn char_upper(self) -> char {
        match self {
            Piece::Pawn => 'P',
            Piece::Knight => 'N',
            Piece::Bishop => 'B',
            Piece::Rook => 'R',
            Piece::Queen => 'Q',
            Piece::King => 'K',
        }
    }

    pub const fn char_lower(self) -> char {
        match self {
            Piece::Pawn => 'p',
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            Piece::King => 'k',
        }
    }

    pub const fn from_char(c: char) -> Option<(Color, Piece)> {
        match c {
            'P' => Some((Color::White, Piece::Pawn)),
            'N' => Some((Color::White, Piece::Knight)),
            'B' => Some((Color::White, Piece::Bishop)),
            'R' => Some((Color::White, Piece::Rook)),
            'Q' => Some((Color::White, Piece::Queen)),
            'K' => Some((Color::White, Piece::King)),
            'p' => Some((Color::Black, Piece::Pawn)),
            'n' => Some((Color::Black, Piece::Knight)),
            'b' => Some((Color::Black, Piece::Bishop)),
            'r' => Some((Color::Black, Piece::Rook)),
            'q' => Some((Color::Black, Piece::Queen)),
            'k' => Some((Color::Black, Piece::King)),
            _ => None,
        }
    }
}

/// Square index 0..64: A1 = 0, B1 = 1, ..., H1 = 7, A2 = 8, ..., H8 = 63.
pub type Square = u8;

pub const A1: Square = 0;
pub const B1: Square = 1;
pub const C1: Square = 2;
pub const D1: Square = 3;
pub const E1: Square = 4;
pub const F1: Square = 5;
pub const G1: Square = 6;
pub const H1: Square = 7;
pub const A8: Square = 56;
pub const B8: Square = 57;
pub const C8: Square = 58;
pub const D8: Square = 59;
pub const E8: Square = 60;
pub const F8: Square = 61;
pub const G8: Square = 62;
pub const H8: Square = 63;
pub const NO_SQ: Square = 64;

#[inline(always)]
pub const fn sq_make(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

#[inline(always)]
pub const fn sq_file(sq: Square) -> u8 {
    sq & 7
}

#[inline(always)]
pub const fn sq_rank(sq: Square) -> u8 {
    sq >> 3
}

#[inline(always)]
pub const fn sq_flip(sq: Square) -> Square {
    sq ^ 56
}

#[inline(always)]
pub const fn sq_mirror(sq: Square) -> Square {
    sq ^ 7
}

pub fn sq_name(sq: Square) -> String {
    if sq >= 64 {
        return "-".to_string();
    }
    let f = (b'a' + sq_file(sq)) as char;
    let r = (b'1' + sq_rank(sq)) as char;
    format!("{f}{r}")
}

pub fn parse_sq(s: &str) -> Option<Square> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    if !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some(sq_make(b[0] - b'a', b[1] - b'1'))
}

/// Castling rights bitmask.
pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

/// Move flags (4 bits, matching standard chess move encoding):
/// bit 3 = promotion, bit 2 = capture, bits 0..2 = subtype.
pub const FLAG_QUIET: u16 = 0;
pub const FLAG_DOUBLE_PUSH: u16 = 1;
pub const FLAG_CASTLE_K: u16 = 2;
pub const FLAG_CASTLE_Q: u16 = 3;
pub const FLAG_CAPTURE: u16 = 4;
pub const FLAG_EP: u16 = 5;
pub const FLAG_PROMO_N: u16 = 8;
pub const FLAG_PROMO_B: u16 = 9;
pub const FLAG_PROMO_R: u16 = 10;
pub const FLAG_PROMO_Q: u16 = 11;
pub const FLAG_PROMO_CAP_N: u16 = 12;
pub const FLAG_PROMO_CAP_B: u16 = 13;
pub const FLAG_PROMO_CAP_R: u16 = 14;
pub const FLAG_PROMO_CAP_Q: u16 = 15;

/// 16-bit packed chess move: bits 0..6 = from, bits 6..12 = to, bits 12..16 = flags.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);

    #[inline(always)]
    pub const fn new(from: Square, to: Square, flag: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (flag << 12))
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        (self.0 & 63) as Square
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        ((self.0 >> 6) & 63) as Square
    }

    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        (self.flag() & 4) != 0
    }

    #[inline(always)]
    pub const fn is_promo(self) -> bool {
        (self.flag() & 8) != 0
    }

    #[inline(always)]
    pub const fn is_tactical(self) -> bool {
        (self.flag() & 12) != 0
    }

    #[inline(always)]
    pub const fn is_ep(self) -> bool {
        self.flag() == FLAG_EP
    }

    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        let f = self.flag();
        f == FLAG_CASTLE_K || f == FLAG_CASTLE_Q
    }

    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        self.flag() == FLAG_DOUBLE_PUSH
    }

    #[inline(always)]
    pub const fn promo_piece(self) -> Option<Piece> {
        if !self.is_promo() {
            return None;
        }
        Some(match self.flag() & 3 {
            0 => Piece::Knight,
            1 => Piece::Bishop,
            2 => Piece::Rook,
            _ => Piece::Queen,
        })
    }

    pub fn to_uci(self) -> String {
        if self.is_none() {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", sq_name(self.from()), sq_name(self.to()));
        if let Some(p) = self.promo_piece() {
            s.push(p.char_lower());
        }
        s
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// Stack-allocated move list (zero-init overhead via `MaybeUninit`, no heap allocation on hot path).
pub struct MoveList {
    moves: [std::mem::MaybeUninit<Move>; MAX_MOVES],
    len: usize,
}

impl Clone for MoveList {
    fn clone(&self) -> Self {
        let mut out = Self::new();
        out.len = self.len;
        for i in 0..self.len {
            out.moves[i] = self.moves[i];
        }
        out
    }
}

impl MoveList {
    #[inline(always)]
    pub const fn new() -> Self {
        Self { moves: [const { std::mem::MaybeUninit::uninit() }; MAX_MOVES], len: 0 }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        // SAFETY: `self.len < MAX_MOVES` (max legal chess moves <= 218 < 256).
        unsafe {
            self.moves.get_unchecked_mut(self.len).write(m);
        }
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: elements `0..self.len` are initialized by `push`, and `MaybeUninit<Move>` has identical layout to `Move`.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr() as *const Move, self.len) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        // SAFETY: elements `0..self.len` are initialized by `push`.
        unsafe { std::slice::from_raw_parts_mut(self.moves.as_mut_ptr() as *mut Move, self.len) }
    }

    #[inline(always)]
    pub fn swap(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.len && b < self.len);
        self.moves.swap(a, b);
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Index<usize> for MoveList {
    type Output = Move;
    #[inline(always)]
    fn index(&self, i: usize) -> &Move {
        debug_assert!(i < self.len);
        // SAFETY: caller indexes `i < self.len`, which is initialized.
        unsafe { self.moves.get_unchecked(i).assume_init_ref() }
    }
}

impl std::ops::IndexMut<usize> for MoveList {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Move {
        debug_assert!(i < self.len);
        // SAFETY: caller indexes `i < self.len`, which is initialized.
        unsafe { self.moves.get_unchecked_mut(i).assume_init_mut() }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    Win(Color),
    Draw(DrawReason),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DrawReason {
    Stalemate,
    FiftyMove,
    Repetition,
    InsufficientMaterial,
}
