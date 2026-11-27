# Chatarunga

**Chatarunga** is an open-source, zero-dependency UCI chess engine written completely from scratch in pure Rust, combining Plain Magic Bitboards, an 8th-generation alpha-beta search with depth-guarded Lazy SMP, an exact 3-piece KPK retrograde bitbase, and an embedded 6.0-billion-step SIMD SCReLU NNUE evaluator (`(768×16 → 1024)×2 → 8` trained on 1.2B unique Bullet S2 positions with Finny Tables).

Named after *Chaturanga*, the ancient Indian precursor to chess.

- **Technical Report (14-page PDF):** [`docs/paper.pdf`](docs/paper.pdf)
- **Zero external crates:** `[dependencies]` in `Cargo.toml` is completely empty.
- **Zero-config single binary:** The champion `h100_v7_screlu_1024_6b.nnue` network (25.2 MB) is embedded directly inside the executable.

---

## Strength & Multi-Rival CCRL 40/15 Calibration

Multi-anchor consensus on balanced ECO openings (including true `10+0.1` real-clock UCI increment matches):
- **Chatarunga 2.0 (v2.0.0 release):** **`~3,440–3,480` CCRL (1-CPU)** / **`~3,520 ± 50` CCRL (4-CPU)**
- **Search v9 (`main`):** **`~3,470–3,515` CCRL (1-CPU)** / **`~3,510–3,555` CCRL (4-CPU)** *(CCRL Top 50 threshold = `3,573`; **`+46 Elo`** cumulative over Search v8 base with zero lost pairs)*

All matches were played using color-swapped opening pairs (logs verbatim in [`logs/`](logs/)):

| Engine Version & Config | Opponent (CCRL 40/15 Anchor) | TC / Book | W / D / L | Score | Elo Diff (95% CI) | Implied CCRL 40/15 |
|---|---|---|---|---|---|---|
| **Chatarunga v9 (1-CPU, `main`)** | **v8 + OPT3b** (1-CPU, same net) | `6+0.06` ECO | **19 / 89 / 12** | **52.9%** | **`+20.3 ± 25.6`** *(LOS 94.0%, 0 LL)* | **`+20.3` over OPT3b** |
| **v8 + OPT3b (1-CPU, bit-exact)** | **Search v8 base** (1-CPU) | 50 ms ECO | **27 / 75 / 18** | **53.8%** | **`+26.1 ± 29.0`** *(LOS 96.2%, 0 LL)* | **`+26.1` from SIMD/Finny** |
| **Chatarunga v9 (1-CPU, `main`)** | **`Velvet 8.1.1`** (1T, CCRL = `3,530`) | `10+0.1` ECO | **7 / 43 / 10** | **47.5%** | **`−17.4 ± 46.8`** | **`~3,513 ± 47`** *(real clock)* |
| **Chatarunga v9 (1-CPU, `main`)** | `Motor 0.9.0` (1T, CCRL #51 = `3,570`) | `10+0.1` ECO | 0 / 36 / 24 | 30.0% | `−147.2 ± 52.6` | **`~3,423 ± 53`** *(real-clock floor)* |
| **Chatarunga v9 (4-CPU, `main`)** | **Stockfish 18** (1T, CCRL = `3,628`) | `10+0.1` ECO | **0 / 40 / 20** | **33.3%** | **`−120.4 ± 42.0`** | **`~3,508 ± 42`** *(real SF18 TM)* |
| **Search v8 (1-CPU)** | **Chatarunga 2.0** (1-CPU, same net) | 50 ms UHO | **49 / 82 / 29** | **56.3%** | **`+43.7 ± 31.0`** *(LOS 99.7%, 0 LL)* | **`+43.7` over v2.0** |
| **Search v8 (4-CPU)** | **Stockfish 18** (1T, CCRL = `3,628`) | 150 ms ECO | **4 / 36 / 20** | **36.7%** | **`−94.9 ± 45.1`** *(1 LL pair)* | **`~3,533 ± 45`** *(fixed movetime)* |
| **Chatarunga 2.0 (1-CPU)** | **Chatarunga 1.0** (1-CPU) | 50 ms UHO | **46 / 54 / 20** | **60.8%** | **`+76.5 ± 37.2`** *(LOS 100%, 0 LL)* | **`+76.5` over v1** |
| **Chatarunga 2.0 (1-CPU)** | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms ECO | **12 / 37 / 11** | **50.8%** | **`+5.8 ± 52.4`** *(0 LL)* | **`~3,480`** |
| **Chatarunga 2.0 (4-CPU)** | **Stockfish 18** (1T, CCRL = `3,628`) | 150 ms ECO | **7 / 28 / 25** | **35.0%** | **`−107.5 ± 63.5`** *(7 wins, 3 won pairs)* | **`~3,521 ± 64`** |
| Chatarunga 1.0 (1-CPU) | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms ECO | 3 / 39 / 18 | 37.5% | `−88.7 ± 51.1` | `~3,385–3,400` |
| Chatarunga 1.0 (4-CPU) | Stockfish 18 (1T, CCRL = `3,628`) | 150 ms ECO | 1 / 32 / 27 | 28.3% | `−161.2 ± 55.7` | `~3,467 ± 56` |

*\*Earlier `go movetime 100` matches against Velvet 8.1.1 showed Velvet exiting early at ~77 ms; upgrading `src/arena.rs` to full `10+0.1` UCI increment clocks eliminated all movetime quirks and anchored Chatarunga v9 (1T) at `~3,513 ± 47` against Velvet 8.1.1 (1T).*

### What's New on `main` (Search v9 & Hot-Loop SIMD — `+46 Elo` over Search v8, `+90 Elo` over v2.0)
- **Bit-Exact Hot-Loop SIMD (`OPT3b`, `+26.1 Elo`, 0 LL):** Fused register-tiled `finny_apply_fused` (ARM64 NEON + x86-64 AVX2), zero-memset `MaybeUninit` `MoveList`/`MovePicker`/`qsearch` buffers, and 4-accumulator NEON `screlu_dot_simd` (+15.3% median CPU NPS).
- **Real-Clock Pacing, Pondering & Runtime Fixes (`v9`, `+20.3 Elo` at `6+0.06`, 0 LL):** Full UCI pondering (`go ponder` / `ponderhit` / legal `bestmove <m> ponder <p>`), `2×` power-of-two `Hash` sizing fix (now allocates full requested MB instead of half), UCI `hashfull` reporting, 50-move TT cutoff guard, KPK bitbase startup pre-warm (eliminating a 117 ms move-1 stall), increment clock pacing (`mtg=26`), 64-piece `finny_refresh` hardening, and real UCI increment clock support (`--tc BASE+INC`) in `arena`.
- **Search v8 (`+43.7 Elo` over v2.0):** Zero-redundancy `MovePicker` quiet history caching, LMR-depth futility/history pruning, cutnode & low-score negative extensions, MVV-LVA + capture-history ProbCut sorting, `opponent_worsening` RFP/LMR, killer/countermove LMR protection, and QSearch quiet-evasion pruning.
- **7 Wins vs. Full-Strength Stockfish 18:** Full 60-game PGN recorded in [`logs/chatarunga_v2_sf18_60g.pgn`](logs/chatarunga_v2_sf18_60g.pgn) (plus `10+0.1` real-clock PGN in [`logs/v9_4t_vs_sf18_1t_10p01.pgn`](logs/v9_4t_vs_sf18_1t_10p01.pgn)).

---

## Quick Start

### Download Prebuilt Releases
Download the standalone binary for your platform from [GitHub Releases](https://github.com/deedy/chess-chatarunga/releases) (Windows x86-64 AVX2/BMI2, Windows x86-64 POPCNT, Linux x86-64 AVX2, and macOS Apple Silicon) and load it directly in CuteChess, Arena, BanksiaGUI, or En Croissant.

### Build from Source (Rust 1.80+)

```bash
# Native optimized build (enables ARM64 NEON / x86-64 AVX2 automatically)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run UCI engine (embedded NNUE, zero external files required)
./target/release/chatarunga

# Standard 6-position node / NPS benchmark
./target/release/chatarunga bench

# Run all 23 unit, perft, rules, pondering, KPK bitbase & bit-exact SIMD tests
cargo test --release
```

---

## UCI Options

| Option | Type | Default | Range | Description |
|---|---|---|---|---|
| `Hash` | spin | `64` | `1..4096` | Lockless 4-way clustered transposition table size (MB) |
| `Threads` | spin | `1` | `1..64` | Depth-guarded Lazy SMP search threads |
| `Ponder` | check | `false` | `true/false` | Full UCI `go ponder` / `ponderhit` / `bestmove <m> ponder <p>` support |
| `Move Overhead` | spin | `10` | `0..2000` | Safety buffer subtracted from clock allocations (ms) |
| `EvalFile` | string | `<embedded>` | path / `classical` | Override embedded NNUE with external CHNN file or classical PeSTO |

---

## License

MIT License — see [`LICENSE`](LICENSE).
