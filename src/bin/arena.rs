//! Multi-threaded chess arena: match internal engines against each other or against Stockfish UCI_Elo.
//!
//! Example:
//!   arena classical sf:2000 --games 60 --ms 50 --threads 8
//!   arena nnue:nets/h100_v3.nnue nnue:nets/h100_v3.nnue --params1 "rfp_mult=75" --epd data/UHO_4060_v4.epd --sprt 0,10 --games 1000 --ms 20 --threads 14

use chatarunga::arena::{run_match_with_book_pgn, PlayerSpec, TimeControl};
use chatarunga::book::load_epd_book;
use chatarunga::elo::SprtConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: arena <p1> <p2> [--games N] [--ms MS | --tc BASE+INC | --nodes N | --depth D] [--ms2 MS | --tc2 BASE+INC | --nodes2 N] [--threads T] [--epd PATH] [--offset N] [--sprt E0,E1] [--params1 K=V,...] [--params2 K=V,...] [--pgn PATH]"
        );
        eprintln!("Specs: classical | nnue:<path> | dual:<small>+<big> | sf:<elo> | sf | uci:<binary>");
        std::process::exit(1);
    }
    let p1_str = &args[1];
    let p2_str = &args[2];
    let mut games = 60usize;
    let mut tc1 = TimeControl::MoveTimeMs(50);
    let mut tc2: Option<TimeControl> = None;
    let mut threads = 8usize;
    let mut epd_path: Option<String> = None;
    let mut book_offset = 0usize;
    let mut sprt: Option<SprtConfig> = None;
    let mut params1: Option<String> = None;
    let mut params2: Option<String> = None;
    let mut pgn_path: Option<String> = None;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--games" if i + 1 < args.len() => {
                games = args[i + 1].parse().unwrap_or(60);
                i += 2;
            }
            "--ms" if i + 1 < args.len() => {
                tc1 = TimeControl::MoveTimeMs(args[i + 1].parse().unwrap_or(50));
                i += 2;
            }
            "--tc" if i + 1 < args.len() => {
                tc1 = TimeControl::parse_clock(&args[i + 1]).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--nodes" if i + 1 < args.len() => {
                tc1 = TimeControl::FixedNodes(args[i + 1].parse().unwrap_or(20000));
                i += 2;
            }
            "--depth" if i + 1 < args.len() => {
                tc1 = TimeControl::FixedDepth(args[i + 1].parse().unwrap_or(6));
                i += 2;
            }
            "--ms2" if i + 1 < args.len() => {
                tc2 = Some(TimeControl::MoveTimeMs(args[i + 1].parse().unwrap_or(50)));
                i += 2;
            }
            "--tc2" if i + 1 < args.len() => {
                tc2 = Some(TimeControl::parse_clock(&args[i + 1]).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                }));
                i += 2;
            }
            "--nodes2" if i + 1 < args.len() => {
                tc2 = Some(TimeControl::FixedNodes(args[i + 1].parse().unwrap_or(20000)));
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                threads = args[i + 1].parse().unwrap_or(8);
                i += 2;
            }
            "--epd" if i + 1 < args.len() => {
                epd_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--offset" if i + 1 < args.len() => {
                book_offset = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            "--sprt" if i + 1 < args.len() => {
                if let Some((a, b)) = args[i + 1].split_once(',') {
                    if let (Ok(e0), Ok(e1)) = (a.parse::<f64>(), b.parse::<f64>()) {
                        sprt = Some(SprtConfig::new(e0, e1));
                    }
                }
                i += 2;
            }
            "--params1" if i + 1 < args.len() => {
                params1 = Some(args[i + 1].clone());
                i += 2;
            }
            "--params2" if i + 1 < args.len() => {
                params2 = Some(args[i + 1].clone());
                i += 2;
            }
            "--pgn" if i + 1 < args.len() => {
                pgn_path = Some(args[i + 1].clone());
                i += 2;
            }
            _ => i += 1,
        }
    }

    let tc2 = tc2.unwrap_or(tc1);
    let mut p1 = PlayerSpec::parse(p1_str).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if let Some(kv) = &params1 {
        p1 = p1.with_params(kv).unwrap_or_else(|e| {
            eprintln!("params1 error: {e}");
            std::process::exit(1);
        });
    }
    let mut p2 = PlayerSpec::parse(p2_str).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if let Some(kv) = &params2 {
        p2 = p2.with_params(kv).unwrap_or_else(|e| {
            eprintln!("params2 error: {e}");
            std::process::exit(1);
        });
    }

    let book = epd_path.as_ref().map(|p| {
        load_epd_book(p).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        })
    });

    println!(
        "Match: {p1_str} vs {p2_str} | {games} games | tc1={tc1:?} tc2={tc2:?} | {threads} threads | book={}",
        book.as_ref().map(|b| format!("{} FENs", b.len())).unwrap_or_else(|| "built-in 60 ECO".into())
    );
    let st = run_match_with_book_pgn(
        &p1,
        &p2,
        p1_str,
        p2_str,
        games,
        tc1,
        tc2,
        threads,
        book.as_deref(),
        book_offset,
        sprt,
        pgn_path.as_deref(),
        true,
    )
    .unwrap_or_else(|e| {
        eprintln!("Match error: {e}");
        std::process::exit(1);
    });
    let (elo, ci, los) = st.elo_ci();
    let sprt_str = if let Some(cfg) = sprt {
        let llr = st.sprt_llr(cfg.elo0, cfg.elo1);
        let verdict = if llr >= cfg.upper_bound() {
            "H1 ACCEPTED"
        } else if llr <= cfg.lower_bound() {
            "H0 ACCEPTED"
        } else {
            "INCONCLUSIVE"
        };
        format!(" | SPRT({},{}) LLR={llr:+.2} -> {verdict}", cfg.elo0, cfg.elo1)
    } else {
        String::new()
    };
    println!(
        "RESULT {p1_str} vs {p2_str}: +{} ={} -{} (penta {:?}) | score {:.2}% | Elo {:+.1} ± {:.1} | LOS {:.1}%{sprt_str}",
        st.wins,
        st.draws,
        st.losses,
        st.penta,
        st.score() * 100.0,
        elo,
        ci,
        los
    );
}
