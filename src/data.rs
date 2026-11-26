//! Compact 32-byte binary training record format (Bullet-compatible layout).
//! Each record stores a board position (occupancy + 4-bit piece nibbles), side to move,
//! White-POV centipawn score, and game WDL result (0 = Black win, 1 = Draw, 2 = White win).

use crate::bitboard::pop_lsb;
use crate::board::{mb_color, mb_piece, Board};
use crate::types::{Color, Piece, Square};

/// 32-byte packed training position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PackedPos {
    pub occ: u64,
    /// Up to 32 pieces in LSB-to-MSB order of `occ`, packed as 4-bit nibbles:
    /// bits 0..3 = piece (0..6), bit 3 = color (0 = White, 1 = Black).
    pub pcs: [u8; 16],
    /// White-POV search score in centipawns.
    pub score: i16,
    /// Game outcome from White's perspective: 0 = Black win, 1 = Draw, 2 = White win.
    pub result: u8,
    /// Side to move: 0 = White, 1 = Black.
    pub stm: u8,
    pub _pad: [u8; 4],
}

const _: () = assert!(std::mem::size_of::<PackedPos>() == 32);

impl PackedPos {
    pub fn pack(b: &Board, white_score: i16, white_result: u8) -> Self {
        let mut occ = b.occ();
        let orig_occ = occ;
        let mut pcs = [0u8; 16];
        let mut idx = 0usize;
        while occ != 0 {
            let sq = pop_lsb(&mut occ);
            let v = b.mailbox[sq as usize];
            let nibble = (mb_piece(v) as u8) | ((mb_color(v) as u8) << 3);
            if idx < 32 {
                if idx & 1 == 0 {
                    pcs[idx >> 1] |= nibble;
                } else {
                    pcs[idx >> 1] |= nibble << 4;
                }
                idx += 1;
            }
        }
        Self {
            occ: orig_occ,
            pcs,
            score: white_score,
            result: white_result,
            stm: b.side as u8,
            _pad: [0; 4],
        }
    }

    #[inline]
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&self.occ.to_le_bytes());
        out[8..24].copy_from_slice(&self.pcs);
        out[24..26].copy_from_slice(&self.score.to_le_bytes());
        out[26] = self.result;
        out[27] = self.stm;
        out
    }

    #[inline]
    pub fn from_bytes(b: &[u8; 32]) -> Self {
        let occ = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let mut pcs = [0u8; 16];
        pcs.copy_from_slice(&b[8..24]);
        let score = i16::from_le_bytes(b[24..26].try_into().unwrap());
        let result = b[26];
        let stm = b[27];
        Self {
            occ,
            pcs,
            score,
            result,
            stm,
            _pad: [0; 4],
        }
    }

    /// Extract piece list `(sq, color, piece)`, king squares `[w_ksq, b_ksq]`, and piece count.
    #[inline]
    pub fn decode_pieces(&self, out: &mut [(Square, Color, Piece); 32]) -> (usize, [Square; 2]) {
        let mut occ = self.occ;
        let mut idx = 0usize;
        let mut kings = [0u8; 2];
        while occ != 0 && idx < 32 {
            let sq = pop_lsb(&mut occ);
            let byte = self.pcs[idx >> 1];
            let nibble = if idx & 1 == 0 { byte & 0xF } else { byte >> 4 };
            let piece = Piece::from_u8(nibble & 7);
            let color = if (nibble >> 3) == 0 {
                Color::White
            } else {
                Color::Black
            };
            if piece == Piece::King {
                kings[color.idx()] = sq;
            }
            out[idx] = (sq, color, piece);
            idx += 1;
        }
        (idx, kings)
    }
}
