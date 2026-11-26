//! Fast multi-threaded self-play NNUE data generator.
//!
//! Writes compact 32-byte `PackedPos` records (Bullet-compatible) with quiet-position filtering,
//! win/draw adjudication, and randomized balanced openings.
//!
//! Example:
//!   datagen --eval classical --positions 2000000 --nodes 5000 --threads 14 --out data/gen0.bin

use chatarunga::book::{random_opening, BOOK_FENS};
use chatarunga::data::PackedPos;
use chatarunga::nnue::NnueNet;
use chatarunga::search::{EvalMode, SearchLimits, Searcher};
use chatarunga::{Board, Color, Game, Outcome, Rng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut eval_spec = "classical".to_string();
    let mut target_pos: u64 = 500_000;
    let mut nodes_per_move: u64 = 5_000;
    let mut threads: usize = 12;
    let mut out_path = "data/gen0.bin".to_string();
    let mut seed: u64 = 1;
    let mut rand_plies: usize = 8;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--eval" if i + 1 < args.len() => {
                eval_spec = args[i + 1].clone();
                i += 2;
            }
            "--positions" if i + 1 < args.len() => {
                target_pos = args[i + 1].parse().unwrap_or(500_000);
                i += 2;
            }
            "--nodes" if i + 1 < args.len() => {
                nodes_per_move = args[i + 1].parse().unwrap_or(5_000);
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                threads = args[i + 1].parse().unwrap_or(12);
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = args[i + 1].clone();
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or(1);
                i += 2;
            }
            "--rand-plies" if i + 1 < args.len() => {
                rand_plies = args[i + 1].parse().unwrap_or(8);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let eval_mode = if eval_spec == "classical" || eval_spec == "hce" {
        EvalMode::Classical
    } else if let Some(p) = eval_spec.strip_prefix("nnue:") {
        EvalMode::Nnue(Arc::new(NnueNet::load(p).expect("load nnue")))
    } else {
        EvalMode::Nnue(Arc::new(NnueNet::load(&eval_spec).expect("load nnue")))
    };

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    let file = File::create(&out_path).expect("create output file");
    let writer = Arc::new(Mutex::new(BufWriter::with_capacity(1 << 20, file)));
    let total_pos = Arc::new(AtomicU64::new(0));
    let total_games = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();

    println!(
        "Datagen: eval={eval_spec} target={target_pos} nodes={nodes_per_move} threads={threads} seed={seed} out={out_path}"
    );

    let mut handles = Vec::with_capacity(threads);
    for tid in 0..threads {
        let eval_mode = eval_mode.clone();
        let writer = Arc::clone(&writer);
        let total_pos = Arc::clone(&total_pos);
        let total_games = Arc::clone(&total_games);
        handles.push(thread::spawn(move || {
            let mut rng = Rng::new(
                seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(tid as u64 + 1),
            );
            let mut searcher = Searcher::new(eval_mode, 16);
            let mut local_buf: Vec<u8> = Vec::with_capacity(32 * 4096);
            let mut game_samples: Vec<(Board, i16)> = Vec::with_capacity(200);

            while total_pos.load(Ordering::Relaxed) < target_pos {
                searcher.clear();
                // 50% pure random opening (8 plies), 50% ECO book + 3 random plies
                let start_board = if rng.next_u64() & 1 == 0 {
                    random_opening(&mut rng, rand_plies, 350)
                } else {
                    let fen = BOOK_FENS[rng.next_usize(BOOK_FENS.len())];
                    let mut b = Board::from_fen(fen).unwrap();
                    let mut ml = chatarunga::MoveList::new();
                    for _ in 0..3 {
                        b.legal_moves(&mut ml);
                        if ml.is_empty() {
                            break;
                        }
                        let m = ml[rng.next_usize(ml.len())];
                        b.apply_move(m);
                    }
                    b
                };

                // Skip unbalanced start positions using a quick probe
                let mut game = Game::from_board(start_board);
                if game.outcome().is_some() {
                    continue;
                }
                let probe = searcher.search(
                    &game.board,
                    &game.history,
                    SearchLimits {
                        max_nodes: 2000,
                        ..Default::default()
                    },
                );
                if probe.score.abs() > 700 || probe.best_move.is_none() {
                    continue;
                }

                game_samples.clear();
                let mut win_streak = 0i32; // >0 white winning, <0 black winning
                let mut draw_streak = 0u32;
                let mut white_result: Option<u8> = None;

                for ply in 0..320 {
                    if let Some(o) = game.outcome() {
                        white_result = Some(match o {
                            Outcome::Win(Color::White) => 2,
                            Outcome::Win(Color::Black) => 0,
                            Outcome::Draw(_) => 1,
                        });
                        break;
                    }

                    let res = searcher.search(
                        &game.board,
                        &game.history,
                        SearchLimits {
                            max_nodes: nodes_per_move,
                            ..Default::default()
                        },
                    );
                    if res.best_move.is_none() {
                        white_result = Some(if game.board.side == Color::White {
                            0
                        } else {
                            2
                        });
                        break;
                    }

                    let white_score = if game.board.side == Color::White {
                        res.score
                    } else {
                        -res.score
                    };

                    // Filter: skip checks, tactical best moves, and extreme mate scores
                    if !game.board.in_check()
                        && !res.best_move.is_tactical()
                        && white_score.abs() < 3000
                    {
                        game_samples.push((game.board, white_score as i16));
                    }

                    // Win adjudication
                    if white_score >= 1500 {
                        win_streak = if win_streak > 0 { win_streak + 1 } else { 1 };
                    } else if white_score <= -1500 {
                        win_streak = if win_streak < 0 { win_streak - 1 } else { -1 };
                    } else {
                        win_streak = 0;
                    }
                    if win_streak >= 4 {
                        white_result = Some(2);
                        break;
                    }
                    if win_streak <= -4 {
                        white_result = Some(0);
                        break;
                    }

                    // Draw adjudication after ply 40
                    if ply >= 40 && white_score.abs() <= 12 && game.board.occ().count_ones() <= 22 {
                        draw_streak += 1;
                        if draw_streak >= 10 {
                            white_result = Some(1);
                            break;
                        }
                    } else {
                        draw_streak = 0;
                    }

                    game.apply_move(res.best_move);
                }

                let wdl = white_result.unwrap_or(1);
                if game_samples.is_empty() {
                    continue;
                }

                local_buf.clear();
                for &(b, sc) in &game_samples {
                    let rec = PackedPos::pack(&b, sc, wdl);
                    local_buf.extend_from_slice(&rec.to_bytes());
                }
                let added = game_samples.len() as u64;
                {
                    let mut w = writer.lock().unwrap();
                    w.write_all(&local_buf).expect("write records");
                }
                let prev = total_pos.fetch_add(added, Ordering::Relaxed);
                let gnum = total_games.fetch_add(1, Ordering::Relaxed) + 1;

                if (prev / 100_000) != ((prev + added) / 100_000) {
                    let cur = prev + added;
                    let secs = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "  [datagen] {cur}/{target_pos} pos ({gnum} games) | {:.0} pos/s | {:.1}s",
                        cur as f64 / secs,
                        secs
                    );
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("datagen worker panicked");
    }
    writer.lock().unwrap().flush().ok();
    let final_pos = total_pos.load(Ordering::Relaxed);
    let final_games = total_games.load(Ordering::Relaxed);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "Done: {final_pos} positions from {final_games} games in {secs:.1}s ({:.0} pos/s) -> {out_path}",
        final_pos as f64 / secs
    );
}
