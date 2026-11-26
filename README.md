# Chatarunga

**Chatarunga** is an open-source, zero-dependency UCI chess engine written completely from scratch in pure Rust, combining Plain Magic Bitboards, a 7th-generation alpha-beta search with depth-guarded Lazy SMP, an exact 3-piece KPK retrograde bitbase, and an embedded 6.0-billion-step SIMD SCReLU NNUE evaluator (`(768×16 → 1024)×2 → 8` trained on 1.2B unique Bullet S2 positions with Finny Tables).

Named after *Chaturanga*, the ancient Indian precursor to chess.

- **Technical Report (14-page PDF):** [`docs/paper.pdf`](docs/paper.pdf)
- **Zero external crates:** `[dependencies]` in `Cargo.toml` is completely empty.
- **Zero-config single binary:** The champion `h100_v7_screlu_1024_6b.nnue` network (25.2 MB) is embedded directly inside the executable.

---

## Strength & Direct CCRL 40/15 Calibration

All matches were played using color-swapped opening pairs (logs verbatim in [`logs/`](logs/)):

| Engine Version & Config | Opponent (CCRL 40/15 Anchor) | TC / Book | W / D / L | Score | Elo Diff (95% CI) | Implied CCRL 40/15 |
|---|---|---|---|---|---|---|
| **Chatarunga 2.0 (1-CPU)** | **Chatarunga 1.0** (1-CPU) | 50 ms UHO | **46 / 54 / 20** | **60.8%** | **`+76.5 ± 37.2`** *(LOS 100%, 0 LL)* | **`+76.5` over v1** |
| **Chatarunga 2.0 (1-CPU)** | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms ECO | **12 / 37 / 11** | **50.8%** | **`+5.8 ± 52.4`** *(0 LL)* | **`~3,480`** *(~#95!)* |
| **Chatarunga 2.0 (4-CPU)** | **Stockfish 18** (1T, CCRL = `3,628`) | 150 ms ECO | **7 / 28 / 25** | **35.0%** | **`−107.5 ± 63.5`** *(3 won pairs!)* | **`~3,521 ± 64`** *(~#75–#79!)* |
| Chatarunga 1.0 (1-CPU) | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms ECO | 3 / 39 / 18 | 37.5% | `−88.7 ± 51.1` | `~3,385–3,400` |
| Chatarunga 1.0 (4-CPU) | Stockfish 18 (1T, CCRL = `3,628`) | 150 ms ECO | 1 / 32 / 27 | 28.3% | `−161.2 ± 55.7` | `~3,467 ± 56` |

### What's New in Chatarunga 2.0 (`+76.5 Elo` vs. v1 / `+94.5 Elo` vs. `akimbo` 1-CPU)
- **Search v7:** TT-PV (`tt_pv`) tracking & 16-bit TT static-eval caching (skipping redundant NNUE evals on TT hits), 1024-scale fractional LMR with `tt_pv`/`cutnode`/`improving`/`tt_capture`/`corrplexity` conditioning + `do_shallower` re-searches, fixed `improving` fallback across check plies, Pawn History (`pawn_hist`), hindsight eval-gain history, NMP verification search (`depth ≥ 12`), fail-soft RFP, best-move stability & eval-drop panic time management, and **38-parameter SPSA tuning** across 8,400 UHO games.
- **6.0B-Step SCReLU NNUE (`h100_v7_screlu_1024_6b.nnue`):** Switched from Pairwise-CReLU (512 output pairs) to **SCReLU (1,024 active neurons/side — 2× output capacity)** trained globally shuffled across 1.2B unique positions in H200 VRAM for 6.0B steps with Power-2.5 loss (`|pred − target|^2.5`) and 4×-unrolled ARM64 NEON + x86-64 AVX2 accumulator SIMD.
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
