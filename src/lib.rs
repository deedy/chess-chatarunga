//! # chess_ai — Fast bitboard chess engine with Alpha-Beta search and SIMD NNUE.

pub mod arena;
pub mod bitboard;
pub mod board;
pub mod book;
pub mod data;
pub mod elo;
pub mod endgame;
pub mod eval;
pub mod nnue;
pub mod rng;
pub mod search;
pub mod types;
pub mod uci;
pub mod zobrist;

pub use board::{Board, Game, STARTPOS_FEN};
pub use eval::{evaluate_classical, see_ge};
pub use nnue::NnueNet;
pub use rng::Rng;
pub use search::{EvalMode, SearchLimits, SearchResult, Searcher};
pub use types::{Color, DrawReason, Move, MoveList, Outcome, Piece, Square, MAX_MOVES};
