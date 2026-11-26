//! Parallel SPSA (Simultaneous Perturbation Stochastic Approximation) tuner for SearchParams.
//!
//! Runs multi-threaded color-swapped UHO mini-matches between `theta + c*delta` and `theta - c*delta`
//! and updates all 25 SearchParams simultaneously.
//!
//! Example:
//!   spsa --nnue nets/h100_v3.nnue --epd data/UHO_4060_v4.epd --iters 350 --pairs 14 --nodes 15000 --threads 14

use chatarunga::arena::{play_internal_game, TimeControl};
use chatarunga::book::{load_epd_book, BOOK_FENS};
use chatarunga::nnue::NnueNet;
use chatarunga::rng::Rng;
use chatarunga::search::{EvalMode, SearchParams, Searcher, NUM_PARAMS, PARAM_META};
use chatarunga::types::{Color, Outcome};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut nnue_path: Option<String> = None;
    let mut dual_spec: Option<String> = None;
    let mut epd_path: Option<String> = None;
    let mut iters = 300usize;
    let mut pairs_per_iter = 14usize; // 28 games per iter = 2 games per thread on 14 threads
    let mut tc = TimeControl::FixedNodes(15_000);
    let mut threads = 14usize;
    let mut seed = 2026_1126u64;
    let mut init_kv: Option<String> = None;

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
            "--epd" if i + 1 < args.len() => {
                epd_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--iters" if i + 1 < args.len() => {
                iters = args[i + 1].parse().unwrap_or(300);
                i += 2;
            }
            "--pairs" if i + 1 < args.len() => {
                pairs_per_iter = args[i + 1].parse().unwrap_or(14);
                i += 2;
            }
            "--nodes" if i + 1 < args.len() => {
                tc = TimeControl::FixedNodes(args[i + 1].parse().unwrap_or(15_000));
                i += 2;
            }
            "--ms" if i + 1 < args.len() => {
                tc = TimeControl::MoveTimeMs(args[i + 1].parse().unwrap_or(15));
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                threads = args[i + 1].parse().unwrap_or(14);
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or(seed);
                i += 2;
            }
            "--init" if i + 1 < args.len() => {
                init_kv = Some(args[i + 1].clone());
                i += 2;
            }
            _ => i += 1,
        }
    }

    let eval_mode = if let Some(spec) = &dual_spec {
        let (sp, bp) = spec.split_once('+').expect("--dual requires <small>+<big>");
        EvalMode::DualNnue {
            small: Arc::new(NnueNet::load(sp).expect("load small")),
            big: Arc::new(NnueNet::load(bp).expect("load big")),
        }
    } else if let Some(p) = &nnue_path {
        EvalMode::Nnue(Arc::new(NnueNet::load(p).expect("load nnue")))
    } else {
        EvalMode::Classical
    };
    let book: Arc<Vec<String>> = match &epd_path {
        Some(p) => Arc::new(load_epd_book(p).expect("load epd")),
        None => Arc::new(BOOK_FENS.iter().map(|s| s.to_string()).collect()),
    };

    let mut init_params = SearchParams::default();
    if let Some(kv) = &init_kv {
        init_params.apply_kv_string(kv).expect("parse --init");
    }
    let mut theta = init_params.to_vec();
    let mut rng = Rng::new(seed);

    // Standard SPSA schedule (Spall 1998 / Stockfish Fishtest style)
    let alpha = 0.602f64;
    let gamma = 0.101f64;
    let big_a = (iters as f64) * 0.10;
    let r_end = 0.002f64; // base learning rate scaling

    println!(
        "SPSA Tuner | {} params | iters={iters} pairs/iter={pairs_per_iter} ({} games/iter) | tc={tc:?} | threads={threads} | book={} FENs",
        NUM_PARAMS,
        pairs_per_iter * 2,
        book.len()
    );
    println!(
        "Start params: {}",
        SearchParams::from_vec(&theta).to_kv_string()
    );

    let t0 = Instant::now();
    let n_threads = threads.max(1).min(pairs_per_iter.max(1));
    let mut workers: Vec<(Searcher, Searcher)> = (0..n_threads)
        .map(|_| {
            (
                Searcher::new(eval_mode.clone(), 8),
                Searcher::new(eval_mode.clone(), 8),
            )
        })
        .collect();

    for k in 0..iters {
        let kf = (k + 1) as f64;
        let c_scale = ((iters as f64) / kf).powf(gamma);
        let a_scale = ((big_a + iters as f64) / (big_a + kf)).powf(alpha);

        let mut delta = [0.0f64; NUM_PARAMS];
        let mut theta_p = [0.0f64; NUM_PARAMS];
        let mut theta_m = [0.0f64; NUM_PARAMS];
        let mut c_k = [0.0f64; NUM_PARAMS];

        for j in 0..NUM_PARAMS {
            let sign = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
            delta[j] = sign;
            let cj = PARAM_META[j].c_end * c_scale;
            c_k[j] = cj;
            theta_p[j] = (theta[j] + cj * sign).clamp(PARAM_META[j].min, PARAM_META[j].max);
            theta_m[j] = (theta[j] - cj * sign).clamp(PARAM_META[j].min, PARAM_META[j].max);
        }

        let p_plus = SearchParams::from_vec(&theta_p);
        let p_minus = SearchParams::from_vec(&theta_m);

        // Pick unique openings for this iteration
        let mut fens = Vec::with_capacity(pairs_per_iter);
        for _ in 0..pairs_per_iter {
            let idx = rng.next_usize(book.len());
            fens.push(book[idx].clone());
        }

        // Run color-swapped pairs in parallel reusing pre-allocated Searchers (zero allocs per iter!)
        let mut wins = 0i64;
        let mut draws = 0i64;
        let mut losses = 0i64;
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for (tid, (sp, sm)) in workers.iter_mut().enumerate() {
                let fens_ref = &fens;
                handles.push(s.spawn(move || -> (i64, i64, i64) {
                    sp.set_params(p_plus);
                    sm.set_params(p_minus);
                    let mut w = 0i64;
                    let mut d = 0i64;
                    let mut l = 0i64;
                    let mut idx = tid;
                    while idx < fens_ref.len() {
                        let fen = &fens_ref[idx];
                        if let Ok(o1) = play_internal_game(sp, sm, fen, tc) {
                            match o1 {
                                Outcome::Win(Color::White) => w += 1,
                                Outcome::Win(Color::Black) => l += 1,
                                Outcome::Draw(_) => d += 1,
                            }
                        }
                        if let Ok(o2) = play_internal_game(sm, sp, fen, tc) {
                            match o2 {
                                Outcome::Win(Color::Black) => w += 1,
                                Outcome::Win(Color::White) => l += 1,
                                Outcome::Draw(_) => d += 1,
                            }
                        }
                        idx += n_threads;
                    }
                    (w, d, l)
                }));
            }
            for h in handles {
                let (w, d, l) = h.join().expect("spsa worker panicked");
                wins += w;
                draws += d;
                losses += l;
            }
        });
        let diff = (wins - losses) as f64; // range [-2*pairs, +2*pairs]
        let n_games = (wins + draws + losses).max(1) as f64;
        let score_diff = diff / n_games; // in [-1.0, +1.0]

        // SPSA step: theta_j += a_j * (score_diff / (2 * c_j * delta_j))
        // Note: a_end_j = r_end * (2 * c_end_j)^2 so each param moves smoothly proportional to its c_end!
        for j in 0..NUM_PARAMS {
            let a_end = r_end * (2.0 * PARAM_META[j].c_end).powi(2) * (n_games / 2.0);
            let a_k = a_end * a_scale;
            let step = (a_k * score_diff / (c_k[j] * delta[j]))
                .clamp(-PARAM_META[j].c_end * 1.5, PARAM_META[j].c_end * 1.5);
            theta[j] = (theta[j] + step).clamp(PARAM_META[j].min, PARAM_META[j].max);
        }

        if (k + 1) % 25 == 0 || k == 0 || k + 1 == iters {
            let elapsed = t0.elapsed().as_secs_f64();
            let cur = SearchParams::from_vec(&theta);
            println!(
                "iter {:4}/{iters} ({:5.1}s) | +{:2} ={:2} -{:2} (diff {:+3}) | {}",
                k + 1,
                elapsed,
                wins,
                draws,
                losses,
                wins - losses,
                cur.to_kv_string()
            );
        }
    }

    let final_p = SearchParams::from_vec(&theta);
    println!("\nFINAL TUNED PARAMS:\n{}", final_p.to_kv_string());
    println!("\nRust struct:\n{:#?}", final_p);
}
