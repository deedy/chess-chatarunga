//! Curated balanced opening positions for reproducible engine matches & benchmarks.

use crate::board::Board;
use crate::eval::evaluate_classical;
use crate::rng::Rng;
use crate::types::MoveList;

use std::fs::File;
use std::io::{BufRead, BufReader};

/// Load opening FENs from an EPD/FEN file (one position per line; accepts 4-field EPDs or 6-field FENs).
pub fn load_epd_book(path: &str) -> Result<Vec<String>, String> {
    let f = File::open(path).map_err(|e| format!("open epd {path}: {e}"))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.map_err(|e| format!("read epd {path}: {e}"))?;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        // Strip EPD operations after ';' if present
        let fen_part = s.split(';').next().unwrap_or(s).trim();
        let toks: Vec<&str> = fen_part.split_whitespace().collect();
        if toks.len() >= 6 {
            out.push(toks[..6].join(" "));
        } else if toks.len() >= 4 {
            out.push(format!(
                "{} {} {} {} 0 1",
                toks[0], toks[1], toks[2], toks[3]
            ));
        }
    }
    if out.is_empty() {
        return Err(format!("no valid FENs found in {path}"));
    }
    Ok(out)
}

/// 60 classic balanced ECO openings (each played as both White and Black = 120 game pairs).
pub const BOOK_FENS: &[&str] = &[
    // Ruy Lopez, Italian, Scotch, Four Knights, Petrov, Philidor
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 1 5",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 4",
    // Sicilian (Najdorf, Classical, Dragon, Sveshnikov, Kan, Taimanov, Alapin, Closed)
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 1 6",
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "r1bqkb1r/pp1p1ppp/2n2n2/4p3/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "rnbqkbnr/1p1p1ppp/p3p3/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",
    "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/2P5/PP1P1PPP/RNBQKBNR b KQkq - 0 2",
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQkq - 1 6",
    "r1bqkb1r/pp1p1ppp/2n1pn2/8/3NP3/2N5/PPP1BPPP/R1BQK2R b KQkq - 3 6",
    // French (Winawer, Classical, Tarrasch, Advance)
    "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq - 1 3",
    "rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/2P5/PP3PPP/RNBQKBNR b KQkq - 0 4",
    // Caro-Kann (Classical, Advance, Exchange, Panov)
    "rnbqkbnr/pp2pppp/2p5/8/3PN3/8/PPP2PPP/R1BQKBNR b KQkq - 0 4",
    "rn1qkbnr/pp2pppp/2p5/3pPb2/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 1 4",
    "rnbqkbnr/pp2pppp/2p5/3P4/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    "rnbqkbnr/pp2pppp/8/3p4/2PP4/8/PP3PPP/RNBQKBNR b KQkq - 0 4",
    // Scandinavian, Pirc, Modern, Alekhine
    "rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 3",
    "rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",
    "rnbqkbnr/ppp1pp1p/3p2p1/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3",
    "rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 4",
    // Queen's Gambit Declined, Slav, Semi-Slav, QGA, Exchange QGD
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 4",
    "rnbqkb1r/ppp1pppp/5n2/8/2pPP3/5N2/PP3PPP/RNBQKB1R b KQkq - 0 4",
    "rnbqkb1r/ppp2ppp/5n2/3p4/3P4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    "r1bqkb1r/pppn1ppp/4pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R w KQkq - 4 6",
    // Nimzo-Indian, Queen's Indian, Bogo-Indian, Catalan
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR b KQkq - 0 4",
    "rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R b KQkq - 0 4",
    "rnbqk2r/ppp2ppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R b KQkq - 1 5",
    "rnbqk2r/ppp2ppp/4pn2/8/2pP4/5NP1/PP2PPBP/RNBQK2R w KQkq - 0 6",
    // King's Indian, Grunfeld, Benoni, Benko, Dutch
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 1 5",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 3 6",
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    "rnbqkb1r/ppp1pp1p/6p1/3n4/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq - 0 5",
    "rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 4",
    "rnbqkb1r/p2ppppp/5n2/1ppP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 4",
    "rnbqkb1r/ppppp1pp/5n2/5p2/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3",
    // English, Reti, London, Trompowsky, Bird
    "rnbqkbnr/pppp1ppp/8/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq - 1 2",
    "rnbqkb1r/pp2pppp/3p1n2/2p5/2P5/2N3P1/PP1PPP1P/R1BQKBNR w KQkq - 0 4",
    "rnbqkb1r/pppp1ppp/4pn2/8/2P5/5NP1/PP1PPP1P/RNBQKB1R b KQkq - 0 3",
    "rnbqkb1r/ppp1pppp/4pn2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
    "rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 2 2",
    "rnbqkb1r/ppp1pppp/5n2/3p4/5P2/5N2/PPPPP1PP/RNBQKB1R b KQkq - 1 2",
    // More varied middlegame-ready openings
    "r1bq1rk1/ppp1bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 7",
    "r2q1rk1/pp2bppp/2n1bn2/3pp3/4P3/1NN5/PPP1BPPP/R1BQ1RK1 w - - 2 9",
    "r1b2rk1/pp1nqppp/2pbpn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 4 8",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 8",
    "r2qk2r/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 b kq - 3 7",
    "r1bqk2r/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQK2R w KQkq - 4 7",
];

/// Generate a random balanced opening position by playing `plies` random moves and rejecting
/// terminal positions or positions with |classical eval| > `max_cp`.
pub fn random_opening(rng: &mut Rng, plies: usize, max_cp: i32) -> Board {
    let mut ml = MoveList::new();
    loop {
        let mut b = Board::startpos();
        let mut ok = true;
        for _ in 0..plies {
            b.legal_moves(&mut ml);
            if ml.is_empty() {
                ok = false;
                break;
            }
            let m = ml[rng.next_usize(ml.len())];
            b.apply_move(m);
        }
        if !ok {
            continue;
        }
        b.legal_moves(&mut ml);
        if ml.is_empty() {
            continue;
        }
        if evaluate_classical(&b).abs() <= max_cp {
            return b;
        }
    }
}
