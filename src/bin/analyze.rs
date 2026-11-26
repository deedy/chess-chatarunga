//! Quantitative NNUE Interpretability & Special Position Analyzer:
//! 1. Implied Chess Piece Values via:
//!    (a) Counterfactual single-piece removal across Opening / Middlegame / Endgame
//!        (P, N, B, R, Q, phase-controlled Bishop Pair, Knight center-vs-rim, Advanced Pawn,
//!         and Hanging-Piece Static Threat Sensitivity).
//!    (b) 7-variable Ordinary Least Squares (OLS) linear material regression on balanced
//!        positions (|eval| < 350 cp) fitting [1, dP, dN, dB, dR, dQ, dBishopPair], eliminating
//!        large-imbalance sigmoid saturation and phase confounding.
//! 2. Deep side-by-side static & 1000ms search analysis of 6 iconic chess positions across
//!    Classical PeSTO, NNUE Gen 3, CHNN v2 (1B), CHNN v3 Threats (1B), and 4-thread Lazy SMP.

use chatarunga::bitboard::{bb, pop_lsb};
use chatarunga::board::{mb_color, mb_piece, Board, Game};
use chatarunga::book::load_epd_book;
use chatarunga::eval::evaluate_classical;
use chatarunga::nnue::{board_attack_maps, NnueNet};
use chatarunga::search::{EvalMode, SearchLimits, Searcher};
use chatarunga::types::{sq_rank, Color, Piece};
use std::sync::Arc;

struct EvalBackend {
    name: &'static str,
    mode: EvalMode,
    /// Effective centipawn scale multiplier (matching SearchParams::default() for external Bullet nets)
    scale_pct: i32,
}

impl EvalBackend {
    fn static_eval_white(&self, b: &Board) -> i32 {
        let stm = match &self.mode {
            EvalMode::Classical => evaluate_classical(b),
            EvalMode::Nnue(net) | EvalMode::DualNnue { big: net, .. } => {
                let raw = net.evaluate_board(b);
                if self.scale_pct == 100 {
                    raw
                } else {
                    raw * self.scale_pct / 100
                }
            }
        };
        if b.side == Color::White {
            stm
        } else {
            -stm
        }
    }
}

#[derive(Default)]
struct Accum {
    sum: f64,
    cnt: u64,
}
impl Accum {
    fn add(&mut self, v: f64) {
        self.sum += v;
        self.cnt += 1;
    }
    fn mean(&self) -> f64 {
        if self.cnt == 0 {
            0.0
        } else {
            self.sum / self.cnt as f64
        }
    }
}

/// Pure-Rust 7x7 Ordinary Least Squares solver (Gaussian elimination with partial pivoting)
/// Features: [1, dP, dN, dB, dR, dQ, dBishopPair]
struct Ols7 {
    xtx: [[f64; 7]; 7],
    xty: [f64; 7],
    y_sum: f64,
    y2_sum: f64,
    n: u64,
}

impl Default for Ols7 {
    fn default() -> Self {
        Self {
            xtx: [[0.0; 7]; 7],
            xty: [0.0; 7],
            y_sum: 0.0,
            y2_sum: 0.0,
            n: 0,
        }
    }
}

impl Ols7 {
    fn add(&mut self, x: [f64; 7], y: f64) {
        for i in 0..7 {
            self.xty[i] += x[i] * y;
            for j in 0..7 {
                self.xtx[i][j] += x[i] * x[j];
            }
        }
        self.y_sum += y;
        self.y2_sum += y * y;
        self.n += 1;
    }

    fn solve(&self) -> Option<[f64; 7]> {
        if self.n < 20 {
            return None;
        }
        let mut a = [[0.0f64; 8]; 7];
        for i in 0..7 {
            for j in 0..7 {
                a[i][j] = self.xtx[i][j];
            }
            a[i][7] = self.xty[i];
        }
        for col in 0..7 {
            let mut piv = col;
            let mut best = a[col][col].abs();
            for row in (col + 1)..7 {
                if a[row][col].abs() > best {
                    best = a[row][col].abs();
                    piv = row;
                }
            }
            if best < 1e-9 {
                return None;
            }
            a.swap(col, piv);
            let diag = a[col][col];
            for j in col..8 {
                a[col][j] /= diag;
            }
            for row in 0..7 {
                if row == col {
                    continue;
                }
                let f = a[row][col];
                for j in col..8 {
                    a[row][j] -= f * a[col][j];
                }
            }
        }
        let mut beta = [0.0; 7];
        for i in 0..7 {
            beta[i] = a[i][7];
        }
        Some(beta)
    }
}

fn collect_diverse_positions(epd_path: &str, net: &Arc<NnueNet>, n_games: usize) -> Vec<Board> {
    let book = load_epd_book(epd_path).expect("load epd");
    let mut s = Searcher::new(EvalMode::Nnue(Arc::clone(net)), 16);
    let mut boards = Vec::with_capacity(n_games * 55);
    let limits = SearchLimits {
        max_nodes: 1500,
        ..Default::default()
    };

    for gi in 0..n_games {
        let fen = &book[(gi * 1237) % book.len()];
        let Ok(mut game) = Game::from_fen(fen) else {
            continue;
        };
        s.clear();
        for ply in 0..150 {
            if game.outcome().is_some() {
                break;
            }
            let b = game.board;
            if ply % 2 == 0 && !b.in_check() {
                let occ_cnt = b.occ().count_ones();
                if (6..=32).contains(&occ_cnt) {
                    boards.push(b);
                }
            }
            let res = s.search(&b, &game.history, limits);
            if res.best_move.is_none() || res.score.abs() > 2500 {
                break;
            }
            game.apply_move(res.best_move);
        }
    }
    boards
}

fn material_features(b: &Board) -> [f64; 7] {
    let dp = b.piece_bb(Color::White, Piece::Pawn).count_ones() as f64
        - b.piece_bb(Color::Black, Piece::Pawn).count_ones() as f64;
    let dn = b.piece_bb(Color::White, Piece::Knight).count_ones() as f64
        - b.piece_bb(Color::Black, Piece::Knight).count_ones() as f64;
    let wb = b.piece_bb(Color::White, Piece::Bishop).count_ones();
    let bb_cnt = b.piece_bb(Color::Black, Piece::Bishop).count_ones();
    let db = wb as f64 - bb_cnt as f64;
    let dr = b.piece_bb(Color::White, Piece::Rook).count_ones() as f64
        - b.piece_bb(Color::Black, Piece::Rook).count_ones() as f64;
    let dq = b.piece_bb(Color::White, Piece::Queen).count_ones() as f64
        - b.piece_bb(Color::Black, Piece::Queen).count_ones() as f64;
    let dbp = (if wb >= 2 { 1.0 } else { 0.0 }) - (if bb_cnt >= 2 { 1.0 } else { 0.0 });
    [1.0, dp, dn, db, dr, dq, dbp]
}

fn run_piece_value_probe(boards: &[Board], backends: &[EvalBackend]) {
    println!(
        "=== IMPLIED CHESS PIECE VALUE ANALYSIS ({} quiet game positions) ===",
        boards.len()
    );
    let mut phase_counts = [0usize; 3];
    for b in boards {
        let cnt = b.occ().count_ones();
        let ph = if cnt >= 24 {
            0
        } else if cnt >= 14 {
            1
        } else {
            2
        };
        phase_counts[ph] += 1;
    }
    println!(
        "Phase distribution: Opening (24-32 pcs) = {}, Middlegame (14-23 pcs) = {}, Endgame (6-13 pcs) = {}\n",
        phase_counts[0], phase_counts[1], phase_counts[2]
    );

    const CENTER_SQ: u64 = (1u64 << 27) | (1u64 << 28) | (1u64 << 35) | (1u64 << 36); // d4, e4, d5, e5
    const RIM_FILES: u64 = 0x8181_8181_8181_8181; // files a and h

    for eb in backends {
        let mut vals: [[Accum; 5]; 4] = Default::default();
        // Phase-controlled Bishop Pair (Middlegame only, 14..=23 pieces)
        let mut mg_bp2 = Accum::default();
        let mut mg_bp1 = Accum::default();
        let mut knight_center = Accum::default();
        let mut knight_rim = Accum::default();
        let mut pawn_rank67 = Accum::default();
        let mut pawn_rank23 = Accum::default();
        // Hanging-piece probe (opponent piece attacked & undefended when STM to move)
        let mut hanging_minor = Accum::default();
        let mut safe_minor = Accum::default();
        let mut hanging_major = Accum::default();
        let mut safe_major = Accum::default();
        // OLS regression on balanced positions (|eval| < 350 cp) overall and by phase
        let mut ols_all = Ols7::default();
        let mut ols_phase: [Ols7; 3] = Default::default();

        for b in boards {
            let cnt = b.occ().count_ones();
            let ph = if cnt >= 24 {
                0
            } else if cnt >= 14 {
                1
            } else {
                2
            };
            let base_w = eb.static_eval_white(b) as f64;
            if base_w.abs() < 350.0 {
                let x = material_features(b);
                ols_all.add(x, base_w);
                ols_phase[ph].add(x, base_w);
            }
            let (_, _, _, att_all) = board_attack_maps(b);

            let mut occ = b.occ();
            while occ != 0 {
                let sq = pop_lsb(&mut occ);
                let v = b.mailbox[sq as usize];
                let col = mb_color(v);
                let piece = mb_piece(v);
                if piece == Piece::King {
                    continue;
                }
                let mut b2 = *b;
                b2.remove_piece(col, piece, sq);
                let occ2 = b2.occ();
                if b2.is_sq_attacked(b2.king_sq(Color::White), occ2, Color::Black)
                    || b2.is_sq_attacked(b2.king_sq(Color::Black), occ2, Color::White)
                {
                    continue;
                }
                let rem_w = eb.static_eval_white(&b2) as f64;
                let delta = if col == Color::White {
                    base_w - rem_w
                } else {
                    rem_w - base_w
                };
                let pi = piece.idx();
                vals[ph][pi].add(delta);
                vals[3][pi].add(delta);

                if piece == Piece::Bishop && ph == 1 {
                    let nb = b.piece_bb(col, Piece::Bishop).count_ones();
                    if nb >= 2 {
                        mg_bp2.add(delta);
                    } else {
                        mg_bp1.add(delta);
                    }
                }
                if piece == Piece::Knight {
                    if (bb(sq) & CENTER_SQ) != 0 {
                        knight_center.add(delta);
                    } else if (bb(sq) & RIM_FILES) != 0 {
                        knight_rim.add(delta);
                    }
                }
                if piece == Piece::Pawn {
                    let rel_r = if col == Color::White {
                        sq_rank(sq)
                    } else {
                        7 - sq_rank(sq)
                    };
                    if rel_r >= 5 {
                        pawn_rank67.add(delta);
                    } else if rel_r <= 2 {
                        pawn_rank23.add(delta);
                    }
                }

                if col != b.side && piece != Piece::Pawn {
                    let ci = col.idx();
                    let enemy = ci ^ 1;
                    let is_att = (att_all[enemy] & bb(sq)) != 0;
                    let is_def = (att_all[ci] & bb(sq)) != 0;
                    let is_minor = matches!(piece, Piece::Knight | Piece::Bishop);
                    if is_att && !is_def {
                        if is_minor {
                            hanging_minor.add(delta);
                        } else {
                            hanging_major.add(delta);
                        }
                    } else if !is_att {
                        if is_minor {
                            safe_minor.add(delta);
                        } else {
                            safe_major.add(delta);
                        }
                    }
                }
            }
        }

        println!("--- Model: {} ---", eb.name);
        println!("  [A] Counterfactual Single-Piece Removal:");
        let phase_names = ["Opening   ", "Middlegame", "Endgame   ", "Overall   "];
        for ph in 0..4 {
            let p = vals[ph][0].mean();
            let n = vals[ph][1].mean();
            let bi = vals[ph][2].mean();
            let r = vals[ph][3].mean();
            let q = vals[ph][4].mean();
            let norm = p.max(1.0);
            println!(
                "    {} | cp: P={:5.1} N={:5.1} B={:5.1} R={:5.1} Q={:6.1} | norm(P=1): P=1.00 N={:.2} B={:.2} R={:.2} Q={:.2} (B-N={:+.2}, R-B={:.2})",
                phase_names[ph],
                p, n, bi, r, q,
                n / norm, bi / norm, r / norm, q / norm,
                (bi - n) / norm, (r - bi) / norm
            );
        }
        println!(
            "  [B] OLS Material Regression (balanced |eval|<350 cp, n={}):",
            ols_all.n
        );
        if let Some(b) = ols_all.solve() {
            let norm = b[1].max(1.0);
            println!(
                "    Overall    | cp: P={:5.1} N={:5.1} B={:5.1} R={:5.1} Q={:6.1} BP={:+5.1} | norm(P=1): P=1.00 N={:.2} B={:.2} R={:.2} Q={:.2} BP={:+.2} (B-N={:+.2}, R-B={:.2})",
                b[1], b[2], b[3], b[4], b[5], b[6],
                b[2] / norm, b[3] / norm, b[4] / norm, b[5] / norm, b[6] / norm,
                (b[3] - b[2]) / norm, (b[4] - b[3]) / norm
            );
        }
        for ph in 0..3 {
            if let Some(b) = ols_phase[ph].solve() {
                let norm = b[1].max(1.0);
                println!(
                    "    {} | cp: P={:5.1} N={:5.1} B={:5.1} R={:5.1} Q={:6.1} BP={:+5.1} | norm(P=1): P=1.00 N={:.2} B={:.2} R={:.2} Q={:.2} BP={:+.2} [n={}]",
                    phase_names[ph],
                    b[1], b[2], b[3], b[4], b[5], b[6],
                    b[2] / norm, b[3] / norm, b[4] / norm, b[5] / norm, b[6] / norm,
                    ols_phase[ph].n
                );
            }
        }
        let p_all = vals[3][0].mean().max(1.0);
        let mg_bp_cp = mg_bp2.mean() - mg_bp1.mean();
        println!("  [C] Positional & Tactical Concept Probes:");
        println!(
            "    Middlegame Bishop Pair (B2 - B1): {:+.1} cp ({:+.2}P) [n2={}, n1={}]",
            mg_bp_cp,
            mg_bp_cp / vals[1][0].mean().max(1.0),
            mg_bp2.cnt,
            mg_bp1.cnt
        );
        println!(
            "    Knight Center vs Rim: Center={:.1} cp ({:.2}P) vs Rim={:.1} cp ({:.2}P) -> diff {:+.1} cp ({:+.2}P)",
            knight_center.mean(),
            knight_center.mean() / p_all,
            knight_rim.mean(),
            knight_rim.mean() / p_all,
            knight_center.mean() - knight_rim.mean(),
            (knight_center.mean() - knight_rim.mean()) / p_all
        );
        println!(
            "    Advanced Pawn (6th/7th) vs Base (2nd/3rd): Adv={:.1} cp ({:.2}P) vs Base={:.1} cp ({:.2}P) -> ratio {:.2}x",
            pawn_rank67.mean(), pawn_rank67.mean() / p_all,
            pawn_rank23.mean(), pawn_rank23.mean() / p_all,
            pawn_rank67.mean() / pawn_rank23.mean().max(1.0)
        );
        println!(
            "    Hanging Minors (N/B): safe={:.1} cp vs hanging={:.1} cp -> static discount {:+.1} cp ({:.1}% retained) [n_hang={}]",
            safe_minor.mean(), hanging_minor.mean(),
            safe_minor.mean() - hanging_minor.mean(),
            100.0 * hanging_minor.mean() / safe_minor.mean().max(1.0),
            hanging_minor.cnt
        );
        println!(
            "    Hanging Majors (R/Q): safe={:.1} cp vs hanging={:.1} cp -> static discount {:+.1} cp ({:.1}% retained) [n_hang={}]\n",
            safe_major.mean(), hanging_major.mean(),
            safe_major.mean() - hanging_major.mean(),
            100.0 * hanging_major.mean() / safe_major.mean().max(1.0),
            hanging_major.cnt
        );
    }
}

fn pv_to_san(b: &Board, pv: &[chatarunga::Move]) -> String {
    let mut cur = *b;
    let mut out = Vec::new();
    for &m in pv.iter().take(6) {
        if m.is_none() || !cur.is_legal_tt_move(m) {
            break;
        }
        out.push(cur.move_to_san(m));
        cur.apply_move(m);
    }
    out.join(" ")
}

fn run_special_positions(backends: &[EvalBackend], net_champ: &Arc<NnueNet>) {
    let final_positions: &[(&str, &str, u64)] = &[
        (
            "P1: Kasparov's Immortal (Kasparov-Topalov, Wijk aan Zee 1999, 24. Rxd4!!)",
            "b2r3r/k4p1p/p2q1np1/NppP4/3p1Q2/P4PPB/1PP4P/1K1RR3 w - - 0 24",
            1000,
        ),
        (
            "P2: Game of the Century (D. Byrne-R. Fischer, New York 1956, 17... Be6!!)",
            "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3R1K1R b - - 0 17",
            1000,
        ),
        (
            "P3: Topalov-Shirov (Linares 1998, 47... Bh3!! — Famous Endgame Horizon Test)",
            "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 b - - 0 47",
            3000,
        ),
        (
            "P4: Opposite-Color Bishops Fortress (White +2 Pawns, Dead Theoretical Draw)",
            "8/4b3/p2k4/P2Pp3/2P1P3/4K3/8/5B2 w - - 0 60",
            1000,
        ),
        (
            "P5: Good Knight vs. Trapped Bad Bishop (Equal Material, Strategic Win)",
            "3k4/1p1b1p2/p1p1p1p1/P1PpP2p/1P1P3P/3N1KP1/8/8 w - - 0 45",
            1000,
        ),
        (
            "P6: Kiwipete Tactical Tension (Complex Pins, Captures, & Static Threats)",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            1000,
        ),
    ];

    println!("=== SPECIAL CHESS POSITIONS MULTI-ENGINE ANALYSIS (Static Eval + Deep Search) ===\n");

    for &(title, fen, ms) in final_positions {
        let limits = SearchLimits {
            soft_ms: Some(ms),
            hard_ms: Some(ms),
            ..Default::default()
        };
        let b = Board::from_fen(fen).expect("valid fen");
        println!("----------------------------------------------------------------");
        println!("{title} [{ms}ms]");
        println!("FEN: {fen}");
        for eb in backends {
            let st_w = eb.static_eval_white(&b);
            let mut searcher = Searcher::new(eb.mode.clone(), 64);
            let res = searcher.search(&b, &[b.hash], limits);
            let sc_w = if b.side == Color::White {
                res.score
            } else {
                -res.score
            };
            println!(
                "  {:<22} | static={:+5} cp | {:4}ms d={:2} score={:+5} cp | best={:<6} | nodes={:8} | PV: {}",
                eb.name,
                st_w,
                ms,
                res.depth,
                sc_w,
                b.move_to_san(res.best_move),
                res.nodes,
                pv_to_san(&b, &res.pv)
            );
        }
        let mut s4 = Searcher::new(EvalMode::Nnue(Arc::clone(net_champ)), 128);
        s4.set_threads(4);
        let res4 = s4.search(&b, &[b.hash], limits);
        let sc4_w = if b.side == Color::White {
            res4.score
        } else {
            -res4.score
        };
        println!(
            "  {:<22} | static=  ---   | {:4}ms d={:2} score={:+5} cp | best={:<6} | nodes={:8} | PV: {}",
            "CHNN v2 1B (4T SMP)",
            ms,
            res4.depth,
            sc4_w,
            b.move_to_san(res4.best_move),
            res4.nodes,
            pv_to_san(&b, &res4.pv)
        );
        println!();
    }
}

fn main() {
    let epd_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/UHO_4060_v4.epd".to_string());
    if epd_path == "--replay" {
        let start_fen = std::env::args().nth(2).expect("need start fen");
        let moves_str = std::env::args()
            .nth(3)
            .expect("need space-separated SAN moves");
        let mut b = Board::from_fen(&start_fen).expect("valid start fen");
        let mut ml = chatarunga::MoveList::new();
        for (i, san) in moves_str.split_whitespace().enumerate() {
            let pre_fen = b.to_fen();
            let move_no = b.fullmove;
            let side_ch = if b.side == Color::White { "." } else { "..." };
            b.legal_moves(&mut ml);
            let m = ml
                .as_slice()
                .iter()
                .copied()
                .find(|&mv| b.move_to_san(mv) == san)
                .unwrap_or_else(|| panic!("could not match SAN '{san}' at ply {i} fen={pre_fen}"));
            b.apply_move(m);
            println!(
                "PLY {:3} | {:2}{:<3} {:<7} | uci={:<5} | PRE: {} | POST: {}",
                i + 1,
                move_no,
                side_ch,
                san,
                m.to_uci(),
                pre_fen,
                b.to_fen()
            );
        }
        return;
    }
    let n_games: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let net_gen3 = Arc::new(NnueNet::load("nets/gen3.nnue").expect("load gen3"));
    let net_v2_1b = Arc::new(NnueNet::load("nets/h100_v5_v2_1b.nnue").expect("load h100_v5_v2_1b"));
    let net_v4 = Arc::new(NnueNet::load("nets/h100_v4.nnue").expect("load h100_v4"));

    let backends = vec![
        EvalBackend {
            name: "Classical PeSTO",
            mode: EvalMode::Classical,
            scale_pct: 100,
        },
        EvalBackend {
            name: "NNUE Gen 3 (Self-Play)",
            mode: EvalMode::Nnue(net_gen3),
            scale_pct: 100,
        },
        EvalBackend {
            name: "CHNN v2 1B (Pairwise)",
            mode: EvalMode::Nnue(Arc::clone(&net_v2_1b)),
            scale_pct: 74,
        },
        EvalBackend {
            name: "CHNN v3 1B (Threats)",
            mode: EvalMode::Nnue(net_v4),
            scale_pct: 74,
        },
    ];

    let boards = collect_diverse_positions(&epd_path, &net_v2_1b, n_games);
    run_piece_value_probe(&boards, &backends);
    run_special_positions(&backends, &net_v2_1b);
}
