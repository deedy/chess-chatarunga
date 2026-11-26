# Chatarunga

**Chatarunga** is an open-source, zero-dependency UCI chess engine written completely from scratch in pure Rust, combining Plain Magic Bitboards, a 6th-generation alpha-beta search with depth-guarded Lazy SMP, an exact 3-piece KPK retrograde bitbase, and an embedded 1.0-billion-position SIMD NNUE evaluator (`(768×16 → 1024)×2 → 8` Pairwise-CReLU with Finny Tables).

Named after *Chaturanga*, the ancient Indian precursor to chess.

- **Technical Report (14-page PDF):** [`docs/paper.pdf`](docs/paper.pdf) — the full narrative from a 2,660-Elo baseline to beating full-strength Stockfish 18, including colored chessboard diagrams, implied piece-value analysis across 10,017 positions, and direct CCRL calibration.
- **Zero external crates:** `[dependencies]` in `Cargo.toml` is completely empty.
- **Zero-config single binary:** The champion `h100_v5_v2_1b.nnue` network is embedded directly inside the executable.

---

## Strength & Direct CCRL 40/15 Calibration

All matches were played using color-swapped ECO opening pairs (logs verbatim in [`logs/`](logs/)):

| Chatarunga Config | CCRL 40/15 Anchor Opponent | TC | W / D / L | Score | Elo Diff (95% CI) | Implied CCRL 40/15 |
|---|---|---|---|---|---|---|
| **Chatarunga 1.0 (1-CPU)** | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms | 3 / 39 / 18 | 37.5% | `−88.7 ± 51.1` | **`~3,385–3,400`** |
| **Chatarunga 1.0 (4-CPU)** | `akimbo 1.1.0-dev` (1T, CCRL #98 = `3,474`) | 100 ms | **19 / 40 / 1** | **65.0%** | **`+107.5 ± 49.0`** | `~3,580`* |
| **Chatarunga 1.0 (4-CPU)** | **Stockfish 18** (1T, CCRL = `3,628`) | 150 ms | **1 / 32 / 27** | **28.3%** | **`−161.2 ± 55.7`** | **`~3,467 ± 56`** |

*\*Comparing 4-CPU against a 1-CPU opponent at 100 ms/move slightly overstates 40/15 multi-core CCRL; the Stockfish 18 anchor (`~3,467 CCRL`) provides the conservative 4-CPU benchmark.*

### Highlight: Checkmate Victory Over Unthrottled Stockfish 18 (Round 25, `1-0`)

In Round 25 of [`logs/sf18_full_60g.pgn`](logs/sf18_full_60g.pgn), Chatarunga (White) defeated unthrottled Stockfish 18 (Black) with **`19. c4!`**, **`21. f4!`**, the deflection sacrifice **`32. Bg6!!`**, and the clinical endgame rook sacrifice **`47. Rxa4!!`** leading to **`60. Qh7#`**.

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

## Key Technical Highlights

- **Bitboard & Move Generation (`src/board.rs`, `src/bitboard.rs`):** 184-byte `Copy` board state with Plain Magic Bitboards and const-generic partitioned legal/tactical/quiet move generation (**539M+ perft nps** on Kiwipete, **20.3M legal movegen pos/s** on Apple M5 Max).
- **Search (`src/search.rs`):** Lockless 4-way 64-byte atomic clustered TT (Hyatt XOR-key check + software prefetch), depth-guarded Lazy SMP, full staged `MovePicker`, Threat-Aware Quiet History, 1/2/4/6-ply Continuation History, 6-source Multi-Correction History (Pawn, W/B Non-Pawn, Minor, Major, 1- & 2-ply Continuation), ProbCut, Singular Extensions, Cut-node LMR, and 29 multi-stage SPSA-tuned parameters (`+184 Elo` verified).
- **Endgame Knowledge (`src/endgame.rs`):** Exact 196,608-state 3-piece KPK retrograde bitbase generated at startup in ~8 ms (24 KB), plus wrong-colored rook pawn draws, `KNNK`, `KBNK` mop-up, and opposite-color bishop drawishness scaling.
- **SIMD NNUE (`src/nnue.rs`):** Embedded `(768×16 → 1024)×2 → 8` Pairwise-CReLU network trained on 1.0 billion Bullet S2 positions on Modal NVIDIA H100 GPUs, with Finny Table differential king-bucket refreshes and bit-exact **ARM64 NEON** + **x86-64 AVX2** SIMD inference kernels (**2.50M search nps** single-core on M5 Max).

---

## License

MIT License — see [`LICENSE`](LICENSE).
