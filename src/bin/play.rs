//! Play interactive chess in the terminal against Classical or NNUE engine.

use chatarunga::nnue::NnueNet;
use chatarunga::search::{EvalMode, SearchLimits, Searcher};
use chatarunga::{Color, Game};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut human = Color::White;
    let mut ms = 200u64;
    let mut nnue_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--black" => {
                human = Color::Black;
                i += 1;
            }
            "--ms" if i + 1 < args.len() => {
                ms = args[i + 1].parse().unwrap_or(200);
                i += 2;
            }
            "--nnue" if i + 1 < args.len() => {
                nnue_path = Some(args[i + 1].clone());
                i += 2;
            }
            _ => i += 1,
        }
    }
    let eval = match nnue_path {
        Some(p) => EvalMode::Nnue(Arc::new(NnueNet::load(&p).expect("load nnue"))),
        None => EvalMode::Classical,
    };
    let mut searcher = Searcher::new(eval, 64);
    let mut game = Game::new();
    let stdin = io::stdin();

    loop {
        println!("\n{}", game.board.pretty());
        if let Some(o) = game.outcome() {
            println!("Game over: {o:?}");
            break;
        }
        if game.board.side == human {
            print!("Your move (UCI, e.g. e2e4, or quit): ");
            io::stdout().flush().ok();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            let s = line.trim();
            if s == "quit" || s == "exit" {
                break;
            }
            match game.board.parse_uci_move(s) {
                Some(m) => game.apply_move(m),
                None => println!("Illegal move: {s}"),
            }
        } else {
            let res = searcher.search(
                &game.board,
                &game.history,
                SearchLimits {
                    soft_ms: Some(ms),
                    hard_ms: Some(ms),
                    ..Default::default()
                },
            );
            let san = game.board.move_to_san(res.best_move);
            println!(
                "AI plays {} ({}) | depth {} | score {} cp | {} nodes | {} ms",
                san,
                res.best_move.to_uci(),
                res.depth,
                res.score,
                res.nodes,
                res.time_ms
            );
            game.apply_move(res.best_move);
        }
    }
}
