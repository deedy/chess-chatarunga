//! Engine throughput benchmark: perft, legal movegen, classical vs NNUE eval, search NPS.

use chatarunga::nnue::NnueNet;
use chatarunga::search::EvalMode;
use chatarunga::uci::run_bench;
use chatarunga::{evaluate_classical, Board, MoveList};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut nnue_path: Option<String> = None;
    let mut dual_spec: Option<String> = None;
    let mut depth = 9i32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nnue" if i + 1 < args.len() => {
                nnue_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--dual" if i + 1 < args.len() => {
                dual_spec = Some(args[i + 1].clone());
                i += 2;
            }
            "--depth" if i + 1 < args.len() => {
                depth = args[i + 1].parse().unwrap_or(9);
                i += 2;
            }
            _ => i += 1,
        }
    }

    // 1. Perft speed
    let b = Board::startpos();
    let t0 = Instant::now();
    let p5 = b.perft(5);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "Perft(5) startpos: {p5} nodes in {:.3}s ({:.1}M nps)",
        dt,
        p5 as f64 / dt / 1e6
    );

    let kiwi =
        Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .unwrap();
    let t0 = Instant::now();
    let k4 = kiwi.perft(4);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "Perft(4) kiwipete: {k4} nodes in {:.3}s ({:.1}M nps)",
        dt,
        k4 as f64 / dt / 1e6
    );

    // 2. Legal movegen throughput
    let mut ml = MoveList::new();
    let iters = 1_000_000u64;
    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..iters {
        kiwi.legal_moves(&mut ml);
        acc += ml.len();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "Legal movegen (kiwipete): {:.2}M pos/s (acc={acc})",
        iters as f64 / dt / 1e6
    );

    // 3. Classical eval throughput
    let t0 = Instant::now();
    let mut ev_sum = 0i64;
    for _ in 0..iters {
        ev_sum += evaluate_classical(&kiwi) as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "Classical eval (kiwipete): {:.2}M evals/s (sum={ev_sum})",
        iters as f64 / dt / 1e6
    );

    // 4. Classical search bench
    let (nodes, secs) = run_bench(EvalMode::Classical, depth);
    println!(
        "Classical search (6 pos, depth {depth}): {nodes} nodes in {secs:.3}s ({:.2}M nps)",
        nodes as f64 / secs / 1e6
    );

    // 5. NNUE search bench (if provided)
    if let Some(p) = nnue_path {
        let net = Arc::new(NnueNet::load(&p).expect("load nnue"));
        let (nodes, secs) = run_bench(EvalMode::Nnue(net), depth);
        println!(
            "NNUE search ({p}, 6 pos, depth {depth}): {nodes} nodes in {secs:.3}s ({:.2}M nps)",
            nodes as f64 / secs / 1e6
        );
    }

    // 6. Dual-Net search bench (if provided as small+big)
    if let Some(spec) = dual_spec {
        let (sp, bp) = spec.split_once('+').expect("--dual requires <small>+<big>");
        let small = Arc::new(NnueNet::load(sp).expect("load small nnue"));
        let big = Arc::new(NnueNet::load(bp).expect("load big nnue"));
        let (nodes, secs) = run_bench(EvalMode::DualNnue { small, big }, depth);
        println!(
            "Dual-Net search ({spec}, 6 pos, depth {depth}): {nodes} nodes in {secs:.3}s ({:.2}M nps)",
            nodes as f64 / secs / 1e6
        );
    }
}
