//! Chatarunga — Zero-dependency Rust UCI chess engine with embedded 6.0B-step 16x1024 SCReLU NNUE.
//!
//! Usage:
//!   `chatarunga`                          Run UCI protocol with embedded NNUE (default)
//!   `chatarunga bench [depth]`            Run standard 6-position OpenBench-style node/NPS benchmark
//!   `chatarunga --nnue <path>`            Run UCI with an external CHNN `.nnue` network file
//!   `chatarunga --classical`              Run UCI with classical tapered PeSTO evaluation

use chatarunga::nnue::NnueNet;
use chatarunga::search::EvalMode;
use chatarunga::uci::{run_bench, run_uci};
use std::sync::Arc;

const EMBEDDED_NET: &[u8] = include_bytes!("../nets/h100_v7_screlu_1024_6b.nnue");
const ENGINE_NAME: &str = "Chatarunga 2.0";

fn load_embedded() -> EvalMode {
    let net = NnueNet::from_bytes(EMBEDDED_NET).unwrap_or_else(|e| {
        eprintln!("Failed to load embedded NNUE: {e}");
        std::process::exit(1);
    });
    EvalMode::Nnue(Arc::new(net))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(String::as_str) == Some("bench") {
        let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(11);
        let (nodes, secs) = run_bench(load_embedded(), depth);
        println!("{nodes} nodes {:.0} nps", nodes as f64 / secs.max(1e-6));
        return;
    }

    let mut nnue_path: Option<String> = None;
    let mut use_classical = false;
    let mut name = ENGINE_NAME.to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nnue" if i + 1 < args.len() => {
                nnue_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--classical" => {
                use_classical = true;
                i += 1;
            }
            "--name" if i + 1 < args.len() => {
                name = args[i + 1].clone();
                i += 2;
            }
            "--version" | "-v" => {
                println!("{ENGINE_NAME}");
                return;
            }
            _ => i += 1,
        }
    }

    let eval = if use_classical {
        EvalMode::Classical
    } else if let Some(p) = nnue_path {
        let net = NnueNet::load(&p).unwrap_or_else(|e| {
            eprintln!("Failed to load NNUE {p}: {e}");
            std::process::exit(1);
        });
        EvalMode::Nnue(Arc::new(net))
    } else {
        load_embedded()
    };

    run_uci(eval, &name);
}
