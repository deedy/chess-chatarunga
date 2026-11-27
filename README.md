# Chatarunga

**Chatarunga** is an open-source, zero-dependency UCI chess engine written completely from scratch in pure Rust, combining Plain Magic Bitboards, an 8th-generation alpha-beta search with depth-guarded Lazy SMP, an exact 3-piece KPK retrograde bitbase, and an embedded 6.0-billion-step SIMD SCReLU NNUE evaluator (`(768×16 → 1024)×2 → 8` trained on 1.2B unique Bullet S2 positions with Finny Tables).

Named after *Chaturanga*, the ancient Indian precursor to chess.

- **Technical Report (14-page PDF):** [`docs/paper.pdf`](docs/paper.pdf)
- **Zero external crates:** `[dependencies]` in `Cargo.toml` is completely empty.
- **Zero-config single binary:** The champion `h100_v7_screlu_1024_6b.nnue` network (25.2 MB) is embedded directly inside the executable.

---

## Strength & Multi-Rival CCRL 40/15 Calibration

Multi-anchor consensus on balanced ECO openings:
- **Chatarunga 2.0 (v2.0.0 release):** **`~3,440–3,480` CCRL (1-CPU)** / **`~3,520 ± 50` CCRL (4-CPU)**
- **Search v8 (`main`):** **`~3,460–3,520` CCRL (1-CPU)** / **`~3,535–3,570` CCRL (4-CPU)** *(CCRL Top 50 threshold = `3,573`)*

All matches were played using color-swapped opening pairs (logs verbatim in [`logs/`](logs/)):

| Engine Version & Config | Opponent (CCRL 40/15 Anchor) | TC / Book | W / D / L | Score | Elo Diff (95% CI) | Implied CCRL 40/15 |
|---|---|---|---|---|---|---|
| **Search v8 (1-CPU, `main`)** | **Chatarunga 2.0** (1-CPU, same net) | 50 ms UHO | **49 / 82 / 29** | **56.3%** | **`+43.7 ± 31.0`** *(LOS 99.7%, 0 LL)* | **`+43.7` over v2.0** |
| **Search v8 (4-CPU, `main`)** | **Stockfish 18** (1T, CCRL = `3,628`) | 150 ms ECO | **4 / 36 / 20** | **36.7%** | **`−94.9 ± 45.1`** *(1 LL pair)* | **`~3,533 ± 45`** *(cleanest anchor)* |
| **Search v8 (4-CPU, `main`)** | `Velvet 8.1.1` (4T, CCRL #47 = `3,577`) | 100 ms ECO | **14 / 37 / 9** | **54.2%** | **`+29.0 ± 49.0`** *(LOS 88.0%)* | `~3,575`* |
| **Search v8 (1-CPU, `main`)** | `Velvet 8.1.1` (1T, CCRL = `3,530`) | 100 ms ECO | **20 / 33 / 7** | **60.8%** | **`+76.5 ± 52.9`** *(LOS 99.8%)* | `~3,570`* |
| **Search v8 (1-CPU, `main`)** | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms ECO | 13 / 34 / 13 | 50.0% | `0.0 ± 58.4` | `~3,474–3,510` |
| **Search v8 (1-CPU, `main`)** | `Motor 0.9.0` (1T, CCRL #51 = `3,570`) | 100 ms ECO | 4 / 30 / 26 | 31.7% | `−133.6 ± 62.2` | **`~3,436 ± 62`** *(floor)* |
| **Chatarunga 2.0 (1-CPU)** | **Chatarunga 1.0** (1-CPU) | 50 ms UHO | **46 / 54 / 20** | **60.8%** | **`+76.5 ± 37.2`** *(LOS 100%, 0 LL)* | **`+76.5` over v1** |
| **Chatarunga 2.0 (1-CPU)** | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms ECO | **12 / 37 / 11** | **50.8%** | **`+5.8 ± 52.4`** *(0 LL)* | **`~3,480`** |
| **Chatarunga 2.0 (4-CPU)** | **Stockfish 18** (1T, CCRL = `3,628`) | 150 ms ECO | **7 / 28 / 25** | **35.0%** | **`−107.5 ± 63.5`** *(7 wins, 3 won pairs)* | **`~3,521 ± 64`** |
| Chatarunga 1.0 (1-CPU) | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms ECO | 3 / 39 / 18 | 37.5% | `−88.7 ± 51.1` | `~3,385–3,400` |
| Chatarunga 1.0 (4-CPU) | Stockfish 18 (1T, CCRL = `3,628`) | 150 ms ECO | 1 / 32 / 27 | 28.3% | `−161.2 ± 55.7` | `~3,467 ± 56` |

*\*Timing diagnostics revealed that Velvet 8.1.1 exited early at ~77 ms on `go movetime 100` (~25–35 Elo handicap, adjusted above), while Stockfish 18 used hand-tuned ARM64 NEON and the full 150 ms. A fifth match against Carp 3.0.1 (`+49 =11 -0`) was excluded after diagnostics showed Carp used only ~55 ms/move.*

### What's New on `main` (Search v8 — `+43.7 Elo` over Chatarunga 2.0)
- **Zero-Redundancy `MovePicker` & LMR-Depth Pruning:** Returns the cached 6-table `quiet_history` directly from `MovePicker::next` (eliminating 2 redundant random-memory lookups per quiet move; depth-10 bench jumped `759k → 1.17M nps` with 18% fewer nodes), paired with LMR-depth futility/history pruning (`lmr_depth = new_depth - r`), cutnode & low-score negative extensions, material-phase eval scaling, cross-history malus, LMR re-search continuation feedback, and root aspiration fail-high depth reduction.
- **ProbCut Ordering & Wave 2/3 Refinements:** MVV-LVA + capture-history sorting inside ProbCut, `opponent_worsening` RFP/LMR adjustments, `-0.5 ply` killer/countermove LMR protection, QSearch quiet-evasion pruning, PV first-child `cutnode` fix, and NMP TT-upper-bound skip.
- **CHNN v4 (`VERSION_SCRELU_ASYM = 4`) Support & Multi-Dataset Lessons:** Added asymmetric 16-king-bucket NNUE format support (`src/nnue.rs`, `tools/modal_train_v3_screlu.py`) and verified across 2.2B-position H200 experiments that raw un-rescored Lc0 T80 loses `-115.2 Elo` while mixing unbalanced `S1/UHO` gains `+43.7 Elo` on UHO yet regresses on balanced ECO (`31.7%` vs SF18) — keeping our clean pure-S2 `h100_v7_screlu_1024_6b.nnue` as champion.
- **7 Wins vs. Full-Strength Stockfish 18:** Full 60-game PGN recorded in [`logs/chatarunga_v2_sf18_60g.pgn`](logs/chatarunga_v2_sf18_60g.pgn).

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

# Run all 22 unit, perft, rules, KPK bitbase & bit-exact SIMD tests
cargo test --release
```

---

## UCI Options

| Option | Type | Default | Range | Description |
|---|---|---|---|---|
| `Hash` | spin | `64` | `1..4096` | Lockless 4-way clustered transposition table size (MB) |
| `Threads` | spin | `1` | `1..64` | Depth-guarded Lazy SMP search threads |
| `Move Overhead` | spin | `10` | `0..2000` | Safety buffer subtracted from clock allocations (ms) |
| `EvalFile` | string | `<embedded>` | path / `classical` | Override embedded NNUE with external CHNN file or classical PeSTO |

---

## License

MIT License — see [`LICENSE`](LICENSE).
