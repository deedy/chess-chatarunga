//! Standard UCI (Universal Chess Interface) protocol handler.

use crate::board::{Board, Game};
use crate::eval::evaluate_classical;
use crate::nnue::NnueNet;
use crate::search::{EvalMode, SearchLimits, Searcher};
use crate::types::Color;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

pub fn run_uci(initial_eval: EvalMode, name: &str) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut game = Game::new();
    let mut tt_mb = 64usize;
    let mut smp_threads = 1usize;
    let mut move_overhead_ms = 10u64;
    let mut searcher: Option<Searcher> = Some(Searcher::new(initial_eval, tt_mb));
    if let Some(s) = searcher.as_mut() {
        s.set_uci_info(true);
    }
    let mut running: Option<(Arc<AtomicBool>, JoinHandle<Searcher>)> = None;

    let finish_search = |running: &mut Option<(Arc<AtomicBool>, JoinHandle<Searcher>)>,
                         searcher: &mut Option<Searcher>,
                         force_stop: bool| {
        if let Some((stop, handle)) = running.take() {
            if force_stop {
                stop.store(true, Ordering::Relaxed);
            }
            if let Ok(s) = handle.join() {
                *searcher = Some(s);
            }
        }
    };

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "uci" => {
                writeln!(stdout, "id name {name}").ok();
                writeln!(stdout, "id author Debarghya Das").ok();
                writeln!(
                    stdout,
                    "option name Hash type spin default 64 min 1 max 4096"
                )
                .ok();
                writeln!(
                    stdout,
                    "option name Threads type spin default 1 min 1 max 64"
                )
                .ok();
                writeln!(
                    stdout,
                    "option name Move Overhead type spin default 10 min 0 max 2000"
                )
                .ok();
                writeln!(
                    stdout,
                    "option name EvalFile type string default <embedded>"
                )
                .ok();
                writeln!(stdout, "uciok").ok();
                stdout.flush().ok();
            }
            "isready" => {
                if let Some((_, ref h)) = running {
                    if h.is_finished() {
                        finish_search(&mut running, &mut searcher, false);
                    }
                }
                writeln!(stdout, "readyok").ok();
                stdout.flush().ok();
            }
            "stop" => {
                finish_search(&mut running, &mut searcher, true);
            }
            "setoption" => {
                finish_search(&mut running, &mut searcher, true);
                if let Some(ni) = tokens.iter().position(|&t| t == "name") {
                    let vi = tokens.iter().position(|&t| t == "value");
                    let opt_name = if let Some(vi) = vi {
                        tokens[ni + 1..vi].join(" ")
                    } else {
                        tokens[ni + 1..].join(" ")
                    };
                    let opt_val = vi.map(|vi| tokens[vi + 1..].join(" ")).unwrap_or_default();
                    if opt_name.eq_ignore_ascii_case("Hash") {
                        if let Ok(mb) = opt_val.parse::<usize>() {
                            tt_mb = mb.clamp(1, 4096);
                            if let Some(s) = searcher.as_mut() {
                                let eval = s.eval.clone();
                                let mut ns = Searcher::new(eval, tt_mb);
                                ns.set_threads(smp_threads);
                                ns.set_uci_info(true);
                                *s = ns;
                            }
                        }
                    } else if opt_name.eq_ignore_ascii_case("Threads") {
                        if let Ok(t) = opt_val.parse::<usize>() {
                            smp_threads = t.clamp(1, 64);
                            if let Some(s) = searcher.as_mut() {
                                s.set_threads(smp_threads);
                            }
                        }
                    } else if opt_name.eq_ignore_ascii_case("Move Overhead") {
                        if let Ok(ms) = opt_val.parse::<u64>() {
                            move_overhead_ms = ms.clamp(0, 2000);
                        }
                    } else if opt_name.eq_ignore_ascii_case("EvalFile") {
                        if opt_val.is_empty() || opt_val == "<empty>" || opt_val == "classical" {
                            let mut ns = Searcher::new(EvalMode::Classical, tt_mb);
                            ns.set_threads(smp_threads);
                            ns.set_uci_info(true);
                            searcher = Some(ns);
                        } else if opt_val != "<embedded>" {
                            if let Ok(net) = NnueNet::load(&opt_val) {
                                let mut ns = Searcher::new(EvalMode::Nnue(Arc::new(net)), tt_mb);
                                ns.set_threads(smp_threads);
                                ns.set_uci_info(true);
                                searcher = Some(ns);
                            }
                        }
                    }
                }
            }
            "ucinewgame" => {
                finish_search(&mut running, &mut searcher, true);
                if let Some(s) = searcher.as_mut() {
                    s.clear();
                }
                game = Game::new();
            }
            "position" => {
                finish_search(&mut running, &mut searcher, true);
                let mut i = 1;
                if i < tokens.len() && tokens[i] == "startpos" {
                    game = Game::new();
                    i += 1;
                } else if i < tokens.len() && tokens[i] == "fen" {
                    i += 1;
                    let mut fen_parts = Vec::new();
                    while i < tokens.len() && tokens[i] != "moves" {
                        fen_parts.push(tokens[i]);
                        i += 1;
                    }
                    if let Ok(g) = Game::from_fen(&fen_parts.join(" ")) {
                        game = g;
                    }
                }
                if i < tokens.len() && tokens[i] == "moves" {
                    for &mv_str in &tokens[i + 1..] {
                        if let Some(m) = game.board.parse_uci_move(mv_str) {
                            game.apply_move(m);
                        }
                    }
                }
            }
            "go" => {
                finish_search(&mut running, &mut searcher, true);
                let limits = parse_go(&tokens[1..], game.board.side, move_overhead_ms);
                if let Some(mut s) = searcher.take() {
                    let board = game.board;
                    let history = game.history.clone();
                    let stop = Arc::new(AtomicBool::new(false));
                    let stop_c = Arc::clone(&stop);
                    let handle = std::thread::spawn(move || {
                        let res = s.search_with_stop(&board, &history, limits, Some(stop_c));
                        let mut out = io::stdout();
                        writeln!(out, "bestmove {}", res.best_move.to_uci()).ok();
                        out.flush().ok();
                        s
                    });
                    running = Some((stop, handle));
                }
            }
            "perft" => {
                finish_search(&mut running, &mut searcher, true);
                let d: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
                let t0 = Instant::now();
                let div = game.board.perft_divide(d);
                let mut total = 0u64;
                for (m, n) in div {
                    writeln!(stdout, "{}: {n}", m.to_uci()).ok();
                    total += n;
                }
                let secs = t0.elapsed().as_secs_f64();
                writeln!(
                    stdout,
                    "\nNodes: {total} ({:.2}M nps)",
                    total as f64 / secs / 1e6
                )
                .ok();
                stdout.flush().ok();
            }
            "eval" => {
                finish_search(&mut running, &mut searcher, true);
                let hce = evaluate_classical(&game.board);
                let nn = searcher.as_ref().and_then(|s| match &s.eval {
                    EvalMode::Classical => None,
                    EvalMode::Nnue(net) => Some(net.evaluate_board(&game.board)),
                    EvalMode::DualNnue { big, .. } => Some(big.evaluate_board(&game.board)),
                });
                writeln!(stdout, "classical: {hce} cp, nnue: {nn:?}").ok();
                stdout.flush().ok();
            }
            "d" => {
                writeln!(stdout, "{}", game.board.pretty()).ok();
                stdout.flush().ok();
            }
            "quit" => {
                finish_search(&mut running, &mut searcher, true);
                break;
            }
            _ => {}
        }
    }
    finish_search(&mut running, &mut searcher, true);
}

fn parse_go(args: &[&str], side: Color, move_overhead_ms: u64) -> SearchLimits {
    let mut limits = SearchLimits::default();
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut movetime: Option<u64> = None;
    let mut movestogo: Option<u64> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "depth" if i + 1 < args.len() => {
                limits.max_depth = args[i + 1].parse().unwrap_or(64);
                i += 2;
            }
            "nodes" if i + 1 < args.len() => {
                limits.max_nodes = args[i + 1].parse().unwrap_or(u64::MAX);
                i += 2;
            }
            "movetime" if i + 1 < args.len() => {
                movetime = args[i + 1].parse().ok();
                i += 2;
            }
            "wtime" if i + 1 < args.len() => {
                wtime = args[i + 1].parse().ok();
                i += 2;
            }
            "btime" if i + 1 < args.len() => {
                btime = args[i + 1].parse().ok();
                i += 2;
            }
            "winc" if i + 1 < args.len() => {
                winc = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            "binc" if i + 1 < args.len() => {
                binc = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            "movestogo" if i + 1 < args.len() => {
                movestogo = args[i + 1].parse().ok();
                i += 2;
            }
            "infinite" => {
                limits.max_depth = 64;
                limits.soft_ms = None;
                limits.hard_ms = None;
                i += 1;
            }
            _ => i += 1,
        }
    }

    if let Some(mt) = movetime {
        let adj = mt.saturating_sub(move_overhead_ms).max(1);
        limits.soft_ms = Some(adj);
        limits.hard_ms = Some(adj);
    } else if let Some(rem) = if side == Color::White { wtime } else { btime } {
        let inc = if side == Color::White { winc } else { binc };
        let safe_rem = rem.saturating_sub(move_overhead_ms).max(5);
        let mtg = movestogo.unwrap_or(28).max(1);
        let base = (safe_rem / mtg + inc * 3 / 4).max(5);
        let hard = (safe_rem / 3)
            .max(base * 2)
            .min(safe_rem.saturating_sub(5).max(5));
        limits.soft_ms = Some((base * 3 / 4).max(5).min(hard));
        limits.hard_ms = Some(hard);
    }
    limits
}

/// Quick bench helper for standard benchmark FENs.
pub fn run_bench(eval: EvalMode, depth: i32) -> (u64, f64) {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    ];
    let mut s = Searcher::new(eval, 32);
    let t0 = Instant::now();
    let mut total_nodes = 0u64;
    for fen in fens {
        s.clear();
        let b = Board::from_fen(fen).unwrap();
        let res = s.search(
            &b,
            &[b.hash],
            SearchLimits {
                max_depth: depth,
                ..Default::default()
            },
        );
        total_nodes += res.nodes;
    }
    (total_nodes, t0.elapsed().as_secs_f64())
}
