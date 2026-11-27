//! Multi-threaded match runner: internal engines vs internal engines OR vs external UCI engines (e.g., Stockfish).
//! Supports EPD opening books, pentanomial pair statistics, SPRT early stopping, and custom SearchParams.

use crate::board::{Board, Game};
use crate::book::BOOK_FENS;
use crate::elo::{MatchStats, SprtConfig};
use crate::nnue::NnueNet;
use crate::search::{EvalMode, SearchLimits, SearchParams, Searcher};
use crate::types::{Color, Move, Outcome};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone)]
pub enum PlayerSpec {
    Classical { tt_mb: usize, smp_threads: usize, params: SearchParams },
    Nnue { net: Arc<NnueNet>, tt_mb: usize, smp_threads: usize, params: SearchParams },
    DualNnue { small: Arc<NnueNet>, big: Arc<NnueNet>, tt_mb: usize, smp_threads: usize, params: SearchParams },
    ExternalUci { cmd: String, options: Vec<(String, String)> },
}

impl PlayerSpec {
    /// Parse a player spec, supporting optional `:t=N` Lazy SMP thread suffix
    /// (e.g. `nnue:nets/h100_v3.nnue:t=4`, `dual:nets/small_128_1b.nnue+nets/h100_v4.nnue:t=4`, `sf:3190:t=4`).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (base, smp_threads) = if let Some((head, tail)) = spec.rsplit_once(":t=") {
            let t: usize = tail.parse().map_err(|_| format!("invalid thread count in {spec}"))?;
            (head, t.clamp(1, 64))
        } else {
            (spec, 1usize)
        };
        let default_tt = if smp_threads > 1 { 64 * smp_threads.min(4) } else { 32 };
        if base == "classical" || base == "hce" {
            return Ok(PlayerSpec::Classical { tt_mb: default_tt, smp_threads, params: SearchParams::default() });
        }
        if let Some(rest) = base.strip_prefix("dual:") {
            let (small_p, big_p) = rest
                .split_once('+')
                .ok_or_else(|| format!("dual spec requires dual:<small>+<big>[:t=N], got {spec}"))?;
            let small = Arc::new(NnueNet::load(small_p)?);
            let big = Arc::new(NnueNet::load(big_p)?);
            return Ok(PlayerSpec::DualNnue {
                small,
                big,
                tt_mb: default_tt,
                smp_threads,
                params: SearchParams::default(),
            });
        }
        if let Some(path) = base.strip_prefix("nnue:") {
            let net = NnueNet::load(path)?;
            return Ok(PlayerSpec::Nnue {
                net: Arc::new(net),
                tt_mb: default_tt,
                smp_threads,
                params: SearchParams::default(),
            });
        }
        if let Some(rest) = base.strip_prefix("sf:") {
            let elo: u32 = rest.parse().map_err(|_| format!("invalid sf elo: {rest}"))?;
            return Ok(PlayerSpec::ExternalUci {
                cmd: "stockfish".to_string(),
                options: vec![
                    ("Threads".into(), smp_threads.to_string()),
                    ("Hash".into(), default_tt.to_string()),
                    ("UCI_LimitStrength".into(), "true".into()),
                    ("UCI_Elo".into(), elo.to_string()),
                ],
            });
        }
        if base == "sf" || base == "stockfish" {
            return Ok(PlayerSpec::ExternalUci {
                cmd: "stockfish".to_string(),
                options: vec![("Threads".into(), smp_threads.to_string()), ("Hash".into(), default_tt.to_string())],
            });
        }
        if let Some(path) = base.strip_prefix("uci:") {
            let mut options = vec![("Hash".into(), default_tt.to_string())];
            if smp_threads > 1 {
                options.push(("Threads".into(), smp_threads.to_string()));
            }
            return Ok(PlayerSpec::ExternalUci { cmd: path.to_string(), options });
        }
        Err(format!("unknown player spec: {spec} (use classical | nnue:<path>[:t=N] | sf[:<elo>][:t=N] | uci:<bin>)"))
    }

    pub fn with_params(mut self, kv: &str) -> Result<Self, String> {
        match &mut self {
            PlayerSpec::Classical { params, .. }
            | PlayerSpec::Nnue { params, .. }
            | PlayerSpec::DualNnue { params, .. } => {
                params.apply_kv_string(kv)?;
            }
            PlayerSpec::ExternalUci { .. } => {}
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TimeControl {
    MoveTimeMs(u64),
    FixedNodes(u64),
    FixedDepth(i32),
    Clock { base_ms: u64, inc_ms: u64 },
}

impl TimeControl {
    /// Parse `BASE+INC` in seconds (e.g. `10+0.1` -> 10000ms + 100ms, `6+0.06` -> 6000ms + 60ms).
    pub fn parse_clock(s: &str) -> Result<Self, String> {
        let (b, inc) = s.split_once('+').ok_or_else(|| format!("invalid clock tc '{s}' (expected BASE+INC)"))?;
        let base_s: f64 = b.parse().map_err(|_| format!("invalid base seconds '{b}'"))?;
        let inc_s: f64 = inc.parse().map_err(|_| format!("invalid inc seconds '{inc}'"))?;
        if base_s <= 0.0 || inc_s < 0.0 {
            return Err(format!("invalid clock tc '{s}'"));
        }
        Ok(TimeControl::Clock {
            base_ms: (base_s * 1000.0).round().max(10.0) as u64,
            inc_ms: (inc_s * 1000.0).round().max(0.0) as u64,
        })
    }
}

struct UciProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl UciProcess {
    fn spawn(cmd: &str, options: &[(String, String)]) -> Result<Self, String> {
        let mut child = Command::new(cmd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn {cmd}: {e}"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut p = Self { child, stdin, stdout };
        p.send("uci")?;
        p.wait_for("uciok")?;
        for (k, v) in options {
            p.send(&format!("setoption name {k} value {v}"))?;
        }
        p.send("isready")?;
        p.wait_for("readyok")?;
        Ok(p)
    }

    fn send(&mut self, line: &str) -> Result<(), String> {
        writeln!(self.stdin, "{line}").map_err(|e| format!("uci write: {e}"))?;
        self.stdin.flush().map_err(|e| format!("uci flush: {e}"))
    }

    fn wait_for(&mut self, token: &str) -> Result<String, String> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line).map_err(|e| format!("uci read: {e}"))?;
            if n == 0 {
                return Err(format!("uci EOF waiting for {token}"));
            }
            if line.starts_with(token) {
                return Ok(line);
            }
        }
    }

    fn new_game(&mut self) -> Result<(), String> {
        self.send("ucinewgame")?;
        self.send("isready")?;
        self.wait_for("readyok")?;
        Ok(())
    }

    fn choose(
        &mut self,
        start_fen: &str,
        uci_moves: &[String],
        tc: TimeControl,
        clock: (u64, u64, u64, u64),
    ) -> Result<String, String> {
        if uci_moves.is_empty() {
            self.send(&format!("position fen {start_fen}"))?;
        } else {
            self.send(&format!("position fen {start_fen} moves {}", uci_moves.join(" ")))?;
        }
        match tc {
            TimeControl::MoveTimeMs(ms) => self.send(&format!("go movetime {ms}"))?,
            TimeControl::FixedNodes(n) => self.send(&format!("go nodes {n}"))?,
            TimeControl::FixedDepth(d) => self.send(&format!("go depth {d}"))?,
            TimeControl::Clock { .. } => {
                let (wt, bt, wi, bi) = clock;
                self.send(&format!("go wtime {wt} btime {bt} winc {wi} binc {bi}"))?;
            }
        }
        let line = self.wait_for("bestmove")?;
        let mv = line.split_whitespace().nth(1).unwrap_or("0000").to_string();
        Ok(mv)
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

enum EngineSlot {
    Internal(Searcher),
    External(UciProcess),
}

impl EngineSlot {
    fn from_spec(spec: &PlayerSpec) -> Result<Self, String> {
        match spec {
            PlayerSpec::Classical { tt_mb, smp_threads, params } => {
                let mut s = Searcher::with_params(EvalMode::Classical, *tt_mb, *params);
                s.set_threads(*smp_threads);
                Ok(EngineSlot::Internal(s))
            }
            PlayerSpec::Nnue { net, tt_mb, smp_threads, params } => {
                let mut s = Searcher::with_params(EvalMode::Nnue(Arc::clone(net)), *tt_mb, *params);
                s.set_threads(*smp_threads);
                Ok(EngineSlot::Internal(s))
            }
            PlayerSpec::DualNnue { small, big, tt_mb, smp_threads, params } => {
                let mut s = Searcher::with_params(
                    EvalMode::DualNnue { small: Arc::clone(small), big: Arc::clone(big) },
                    *tt_mb,
                    *params,
                );
                s.set_threads(*smp_threads);
                Ok(EngineSlot::Internal(s))
            }
            PlayerSpec::ExternalUci { cmd, options } => Ok(EngineSlot::External(UciProcess::spawn(cmd, options)?)),
        }
    }

    fn new_game(&mut self) -> Result<(), String> {
        match self {
            EngineSlot::Internal(s) => {
                s.clear();
                Ok(())
            }
            EngineSlot::External(p) => p.new_game(),
        }
    }

    fn choose(
        &mut self,
        game: &Game,
        start_fen: &str,
        uci_moves: &[String],
        tc: TimeControl,
        clock: (u64, u64, u64, u64),
    ) -> Result<Move, String> {
        match self {
            EngineSlot::Internal(s) => {
                let limits = match tc {
                    TimeControl::MoveTimeMs(ms) => {
                        SearchLimits { soft_ms: Some(ms), hard_ms: Some(ms), ..Default::default() }
                    }
                    TimeControl::FixedNodes(n) => SearchLimits { max_nodes: n, ..Default::default() },
                    TimeControl::FixedDepth(d) => SearchLimits { max_depth: d, ..Default::default() },
                    TimeControl::Clock { .. } => {
                        let (wt, bt, wi, bi) = clock;
                        let (rem, inc) = if game.board.side == Color::White { (wt, wi) } else { (bt, bi) };
                        let safe_rem = rem.saturating_sub(15).max(5);
                        let mtg = if inc == 0 { 44 } else { 40 };
                        let base = (safe_rem / mtg + inc * 3 / 4).max(5);
                        let hard = (base * 5 / 2).min(safe_rem / 6).max(5).min(safe_rem.saturating_sub(5).max(5));
                        let soft = (base * 3 / 4).max(5).min(hard);
                        SearchLimits { soft_ms: Some(soft), hard_ms: Some(hard), ..Default::default() }
                    }
                };
                Ok(s.search(&game.board, &game.history, limits).best_move)
            }
            EngineSlot::External(p) => {
                let s = p.choose(start_fen, uci_moves, tc, clock)?;
                game.board
                    .parse_uci_move(&s)
                    .ok_or_else(|| format!("illegal UCI move '{s}' at {}", game.board.to_fen()))
            }
        }
    }
}

pub fn play_internal_game(
    white: &mut Searcher,
    black: &mut Searcher,
    start_fen: &str,
    tc: TimeControl,
) -> Result<Outcome, String> {
    white.clear();
    black.clear();
    let mut game = Game::from_board(Board::from_fen(start_fen)?);
    let limits = match tc {
        TimeControl::MoveTimeMs(ms) => SearchLimits { soft_ms: Some(ms), hard_ms: Some(ms), ..Default::default() },
        TimeControl::FixedNodes(n) => SearchLimits { max_nodes: n, ..Default::default() },
        TimeControl::FixedDepth(d) => SearchLimits { max_depth: d, ..Default::default() },
        TimeControl::Clock { base_ms, inc_ms } => {
            let ms = (base_ms / 40 + inc_ms * 3 / 4).max(5);
            SearchLimits { soft_ms: Some(ms), hard_ms: Some(ms * 2), ..Default::default() }
        }
    };
    let mut adjudicate_win = 0i32;
    let mut adjudicate_draw = 0i32;

    for ply in 0..320 {
        if let Some(o) = game.outcome() {
            return Ok(o);
        }
        let res = if game.board.side == Color::White {
            white.search(&game.board, &game.history, limits)
        } else {
            black.search(&game.board, &game.history, limits)
        };
        if res.best_move.is_none() {
            return Ok(Outcome::Win(game.board.side.flip()));
        }
        // Fast win/draw adjudication for SPSA/internal games (standard Fishtest thresholds)
        let white_score = if game.board.side == Color::White { res.score } else { -res.score };
        if white_score >= 1000 {
            adjudicate_win = (adjudicate_win + 1).max(1);
            if adjudicate_win >= 6 {
                return Ok(Outcome::Win(Color::White));
            }
        } else if white_score <= -1000 {
            adjudicate_win = (adjudicate_win - 1).min(-1);
            if adjudicate_win <= -6 {
                return Ok(Outcome::Win(Color::Black));
            }
        } else {
            adjudicate_win = 0;
        }
        if ply >= 70 && white_score.abs() <= 12 {
            adjudicate_draw += 1;
            if adjudicate_draw >= 12 {
                return Ok(Outcome::Draw(crate::types::DrawReason::FiftyMove));
            }
        } else {
            adjudicate_draw = 0;
        }
        game.apply_move(res.best_move);
    }
    Ok(Outcome::Draw(crate::types::DrawReason::FiftyMove))
}

fn play_one_game(
    white: &mut EngineSlot,
    black: &mut EngineSlot,
    start_fen: &str,
    tc_white: TimeControl,
    tc_black: TimeControl,
) -> Result<(Outcome, Vec<String>), String> {
    white.new_game()?;
    black.new_game()?;
    let mut game = Game::from_board(Board::from_fen(start_fen)?);
    let mut uci_moves: Vec<String> = Vec::with_capacity(200);
    let mut san_moves: Vec<String> = Vec::with_capacity(200);

    let mut wtime_ms: i64 = match tc_white {
        TimeControl::Clock { base_ms, .. } => base_ms as i64,
        _ => 10_000,
    };
    let mut btime_ms: i64 = match tc_black {
        TimeControl::Clock { base_ms, .. } => base_ms as i64,
        _ => 10_000,
    };
    let winc_ms: u64 = match tc_white {
        TimeControl::Clock { inc_ms, .. } => inc_ms,
        _ => 0,
    };
    let binc_ms: u64 = match tc_black {
        TimeControl::Clock { inc_ms, .. } => inc_ms,
        _ => 0,
    };

    for _ in 0..400 {
        if let Some(o) = game.outcome() {
            return Ok((o, san_moves));
        }
        let clock = (wtime_ms.max(1) as u64, btime_ms.max(1) as u64, winc_ms, binc_ms);
        let t0 = std::time::Instant::now();
        let is_white = game.board.side == Color::White;
        let m = if is_white {
            white.choose(&game, start_fen, &uci_moves, tc_white, clock)?
        } else {
            black.choose(&game, start_fen, &uci_moves, tc_black, clock)?
        };
        let spent = t0.elapsed().as_millis() as i64;
        if is_white {
            if matches!(tc_white, TimeControl::Clock { .. }) {
                wtime_ms -= spent;
                if wtime_ms < -150 {
                    return Ok((Outcome::Win(Color::Black), san_moves));
                }
                wtime_ms = (wtime_ms + winc_ms as i64).max(1);
            }
        } else if matches!(tc_black, TimeControl::Clock { .. }) {
            btime_ms -= spent;
            if btime_ms < -150 {
                return Ok((Outcome::Win(Color::White), san_moves));
            }
            btime_ms = (btime_ms + binc_ms as i64).max(1);
        }
        if m.is_none() {
            return Ok((Outcome::Win(game.board.side.flip()), san_moves));
        }
        san_moves.push(game.board.move_to_san(m));
        uci_moves.push(m.to_uci());
        game.apply_move(m);
    }
    Ok((Outcome::Draw(crate::types::DrawReason::FiftyMove), san_moves))
}

fn format_pgn_game(
    round: usize,
    white_name: &str,
    black_name: &str,
    start_fen: &str,
    outcome: Outcome,
    san_moves: &[String],
) -> String {
    let res = match outcome {
        Outcome::Win(Color::White) => "1-0",
        Outcome::Win(Color::Black) => "0-1",
        Outcome::Draw(_) => "1/2-1/2",
    };
    let mut out = String::with_capacity(512);
    out.push_str(&format!("[Event \"chess_ai Arena\"]\n[Round \"{round}\"]\n"));
    out.push_str(&format!("[White \"{white_name}\"]\n[Black \"{black_name}\"]\n"));
    out.push_str(&format!("[Result \"{res}\"]\n[SetUp \"1\"]\n[FEN \"{start_fen}\"]\n\n"));
    let start_board = Board::from_fen(start_fen).unwrap_or_else(|_| Board::startpos());
    let mut move_no = start_board.fullmove as usize;
    let mut white_to_move = start_board.side == Color::White;
    if !white_to_move && !san_moves.is_empty() {
        out.push_str(&format!("{move_no}... "));
    }
    for (i, san) in san_moves.iter().enumerate() {
        if white_to_move {
            out.push_str(&format!("{move_no}. {san} "));
        } else {
            out.push_str(&format!("{san} "));
            move_no += 1;
        }
        if (i + 1) % 12 == 0 {
            out.push('\n');
        }
        white_to_move = !white_to_move;
    }
    out.push_str(&format!("{res}\n\n"));
    out
}

fn outcome_score_for_p1(o: Outcome, p1_is_white: bool) -> u8 {
    match o {
        Outcome::Win(c) => {
            if (c == Color::White) == p1_is_white {
                2
            } else {
                0
            }
        }
        Outcome::Draw(_) => 1,
    }
}

/// Run `games` games (in color-swapped pairs) between `p1` and `p2` using `threads` threads.
/// Optionally takes a custom opening book slice, SPRT early-stopping config, and PGN output file.
#[allow(clippy::too_many_arguments)]
pub fn run_match_with_book_pgn(
    p1: &PlayerSpec,
    p2: &PlayerSpec,
    p1_name: &str,
    p2_name: &str,
    games: usize,
    tc1: TimeControl,
    tc2: TimeControl,
    threads: usize,
    book: Option<&[String]>,
    book_offset: usize,
    sprt: Option<SprtConfig>,
    pgn_path: Option<&str>,
    verbose: bool,
) -> Result<MatchStats, String> {
    let pairs = (games + 1) / 2;
    let next_pair = Arc::new(Mutex::new(0usize));
    let stats = Arc::new(Mutex::new(MatchStats::default()));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let book_arc: Option<Arc<Vec<String>>> = book.map(|b| Arc::new(b.to_vec()));
    let pgn_file: Option<Arc<Mutex<std::fs::File>>> = match pgn_path {
        Some(path) => {
            Some(Arc::new(Mutex::new(std::fs::File::create(path).map_err(|e| format!("create pgn {path}: {e}"))?)))
        }
        None => None,
    };
    let p1_label = Arc::new(p1_name.to_string());
    let p2_label = Arc::new(p2_name.to_string());
    let n_threads = threads.max(1).min(pairs.max(1));

    let mut handles = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let p1 = p1.clone();
        let p2 = p2.clone();
        let p1_label = Arc::clone(&p1_label);
        let p2_label = Arc::clone(&p2_label);
        let next_pair = Arc::clone(&next_pair);
        let stats = Arc::clone(&stats);
        let stop_flag = Arc::clone(&stop_flag);
        let book_arc = book_arc.clone();
        let pgn_file = pgn_file.clone();
        handles.push(thread::spawn(move || -> Result<(), String> {
            let mut e1 = EngineSlot::from_spec(&p1)?;
            let mut e2 = EngineSlot::from_spec(&p2)?;
            loop {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                let pi = {
                    let mut g = next_pair.lock().unwrap();
                    if *g >= pairs {
                        break;
                    }
                    let cur = *g;
                    *g += 1;
                    cur
                };
                let fen_str: String = if let Some(b) = &book_arc {
                    b[(book_offset + pi * 997) % b.len()].clone()
                } else {
                    BOOK_FENS[(book_offset + pi) % BOOK_FENS.len()].to_string()
                };
                let (o1, san1) = play_one_game(&mut e1, &mut e2, &fen_str, tc1, tc2)?;
                let g2 =
                    if pi * 2 + 1 < games { Some(play_one_game(&mut e2, &mut e1, &fen_str, tc2, tc1)?) } else { None };

                if let Some(pf) = &pgn_file {
                    let mut f = pf.lock().unwrap();
                    let _ =
                        f.write_all(format_pgn_game(pi * 2 + 1, &p1_label, &p2_label, &fen_str, o1, &san1).as_bytes());
                    if let Some((o2, ref san2)) = g2 {
                        let _ = f.write_all(
                            format_pgn_game(pi * 2 + 2, &p2_label, &p1_label, &fen_str, o2, san2).as_bytes(),
                        );
                    }
                    let _ = f.flush();
                }

                let mut st = stats.lock().unwrap();
                let s1 = outcome_score_for_p1(o1, true);
                if let Some((o2, _)) = g2 {
                    let s2 = outcome_score_for_p1(o2, false);
                    st.record_pair(s1, s2);
                } else {
                    match s1 {
                        2 => st.wins += 1,
                        1 => st.draws += 1,
                        _ => st.losses += 1,
                    }
                }
                let mut sprt_msg = String::new();
                if let Some(cfg) = sprt {
                    let llr = st.sprt_llr(cfg.elo0, cfg.elo1);
                    let lo = cfg.lower_bound();
                    let hi = cfg.upper_bound();
                    sprt_msg = format!(" | LLR {llr:+.2} [{lo:.2}, {hi:.2}]");
                    if llr >= hi || llr <= lo {
                        stop_flag.store(true, Ordering::Relaxed);
                    }
                }
                if verbose && (st.total() % 20 == 0 || st.total() == games as u64 || stop_flag.load(Ordering::Relaxed))
                {
                    let (elo, ci, los) = st.elo_ci();
                    eprintln!(
                        "  [{}/{}] +{} ={} -{} ({:.1}%) Elo {:+.1} ± {:.1} (LOS {:.1}%){sprt_msg}",
                        st.total(),
                        games,
                        st.wins,
                        st.draws,
                        st.losses,
                        st.score() * 100.0,
                        elo,
                        ci,
                        los
                    );
                }
            }
            Ok(())
        }));
    }

    for h in handles {
        h.join().map_err(|_| "worker thread panicked".to_string())??;
    }
    let final_stats = *stats.lock().unwrap();
    Ok(final_stats)
}

/// Backward-compatible wrapper without PGN recording.
#[allow(clippy::too_many_arguments)]
pub fn run_match_with_book(
    p1: &PlayerSpec,
    p2: &PlayerSpec,
    games: usize,
    tc1: TimeControl,
    tc2: TimeControl,
    threads: usize,
    book: Option<&[String]>,
    book_offset: usize,
    sprt: Option<SprtConfig>,
    verbose: bool,
) -> Result<MatchStats, String> {
    run_match_with_book_pgn(p1, p2, "p1", "p2", games, tc1, tc2, threads, book, book_offset, sprt, None, verbose)
}

/// Backward-compatible wrapper using the built-in 60-opening ECO book.
pub fn run_match(
    p1: &PlayerSpec,
    p2: &PlayerSpec,
    games: usize,
    tc1: TimeControl,
    tc2: TimeControl,
    threads: usize,
    verbose: bool,
) -> Result<MatchStats, String> {
    run_match_with_book(p1, p2, games, tc1, tc2, threads, None, 0, None, verbose)
}
