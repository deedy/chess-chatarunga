# Chess AI Experiments Log

Every experiment is backed by raw logs in `chess/logs/`. Matches use color-swapped pairs across 60 balanced ECO openings (`src/book.rs`). All NNUE nets were trained purely from our own engine's self-play data.

## 1. Classical Alpha-Beta Baseline (`Search v1`)

- **Engine**: 152-byte `Copy` bitboard + mailbox board, plain Magic Bitboards, strict legal movegen (**501M perft nps** kiwipete, **19.3M pos/s** legal movegen).
- **Eval**: Tapered PeSTO MG/EG piece-square tables, passed/isolated/doubled pawns, piece mobility, rook open/semi-open files, king pawn shield, bishop pair, tempo (`src/eval.rs`).
- **Search v1**: Iterative deepening, aspiration windows, PVS, main TT, qsearch with SEE & delta pruning, check extensions, RFP, NMP, razoring, futility, LMP, LMR, killers, countermoves, butterfly history.

| Match (50 ms/move) | Games | W / D / L | Score | Elo (95% CI) | Log |
|---|---|---|---|---|---|
| `classical` vs `sf:2200` | 40 | 33 / 0 / 7 | 82.5% | +269 ± 167 | `logs/classical_vs_sf2200.txt` |
| `classical` vs `sf:2500` | 40 | 33 / 3 / 4 | 86.3% | +319 ± 175 | `logs/classical_vs_sf2500.txt` |
| `classical` vs `sf:2700` | 40 | 20 / 2 / 18 | 52.5% | +17 ± 109 | `logs/classical_vs_sf2700.txt` |
| `classical` vs `sf:2900` | 40 | 3 / 5 / 32 | 13.8% | −319 ± 158 | `logs/classical_vs_sf2900.txt` |
| `classical` vs `sf:3100` | 40 | 2 / 5 / 33 | 11.3% | −359 ± 174 | `logs/classical_vs_sf3100.txt` |

**Result**: Classical baseline anchors around **~2,650–2,700 Stockfish UCI Elo**.

## 2. Generation 1 NNUE & The WDL-Scale Discovery

Generated `data/gen0.bin`: **3,001,436** quiet positions from **32,812** classical self-play games at 6,000 nodes/move (7,075 pos/s across 14 threads, `logs/datagen_gen0.log`).

| Candidate | Architecture | Training | Result vs `classical` (50 ms) | Verdict |
|---|---|---|---|---|
| `gen1_256` (first attempt) | `(768->256)x2->8` | `wdl=0.45, scale=400`, 25 ep | 12.5% (−338 ± 136 Elo, 60g) | **Rejected** — `K=400` mismatched classical's true WDL slope (`K≈200`), inflating evals ~1.7× and overfitting 32k game outcomes (`train=0.0089` vs `val=0.0108`). |
| `gen1` (`test_wdl0`) | `(768->256)x2->8` | `wdl=0.0, scale=400`, 15 ep (22.4 s) | **75.0% (+191 ± 78 Elo, 60g, LOS 100%)** | **Kept** — +108 Elo even at depth 1; runs at 3.61M nps (96% of classical). |
| `gen1_hl512_w0` | `(768->512)x2->8` | `wdl=0.0, scale=250`, 20 ep | −70 ± 74 Elo vs `gen1` (60g) | **Rejected** — `scale=250` saturates sigmoid on material imbalances and HL=512 is slower at 50 ms. |
| `gen1_hl512_w15` | `(768->512)x2->8` | `wdl=0.15, scale=220`, 20 ep | −134 ± 81 Elo vs `gen1` (60g) | **Rejected** — WDL blending still overfits 32k games. |
| `gen1_kb4_256` | `(768*4->256)x2->8` | `wdl=0.0, scale=400`, 20 ep | −47 ± 73 Elo vs `gen1` (60g) | **Rejected** — 4 king buckets overfit 32k games without Finny table cache. |

**Key lesson**: On datasets under ~100k games, each game contributes ~90 positions sharing a single noisy WDL bit, so pure search-score targets (`wdl=0.0`) with `scale=400` provide ~100× lower target variance and clean non-saturated gradients across material imbalances.

## 3. Search v2 + 5-Instruction NEON SCReLU Kernel

Upgrades:
- Replaced widening `vmull_s16` + `vmovl_s16` + `vmlaq_s32` in `screlu_dot_simd` with 5-instruction `vmulq_s16(c, w)` + `vmlal_s16` + `vmlal_high_s16` (bit-identical since `|w| <= 127` and `0 <= c <= 255`).
- Added Quiescence TT probe/store, TT-refined static eval, incremental `pawn_hash` + Pawn Correction History, Capture History, and 1-ply Continuation History.
- Switched `train_nnue` validation split to a game-disjoint tail split + best-val checkpoint saving.

**Impact**:
- Depth-10 nodes on 6 benchmark positions dropped **35%** (`164k -> 107k`), and **NNUE search became faster in wall-clock time than classical search** (`0.035 s` vs `0.041 s`).
- `gen1` vs `sf:2900` (50 ms, 40 games) jumped from **20.0% (−241 Elo)** → **33.8% (−117 Elo)** (+124 Elo from the search upgrade, +202 Elo over classical v1).

## 4. Generation 2 NNUE (Pure NNUE Self-Play)

Generated `data/gen1.bin`: **4,001,528** quiet positions from **43,621** Gen 1 NNUE self-play games at 6,000 nodes/move (5,156 pos/s, `logs/datagen_gen1.log`).

| Candidate | Data | Architecture | Result vs `gen1` / `gen2` (50 ms, 60g) | Verdict |
|---|---|---|---|---|
| `gen2` (`gen2_256`) | `gen1.bin` (4M) | `(768->256)x2->8`, `wdl=0.0` | **+95 ± 70 Elo vs `gen1` (63.3%, LOS 99.8%)**, **+250 ± 99 Elo vs `classical` (80.8%)** | **Kept** — dropping classical `gen0.bin` gave a massive jump. |
| `gen2_512_comb` | `gen0+gen1` (7M) | `(768->512)x2->8`, `wdl=0.0` | −29 ± 57 Elo vs `gen1` | **Rejected** — mixing classical `gen0.bin` dragged eval back toward classical. |
| `gen2_kb4_512_comb` | `gen0+gen1` (7M) | `(768*4->512)x2->8`, `wdl=0.0` | +41 ± 70 Elo vs `gen1` | **Rejected** — peaked at epoch 2 (`val=0.00274`) then overfit. |
| `gen2_512` | `gen1.bin` (4M) | `(768->512)x2->8`, best-val | −64 ± 62 Elo vs `gen2` | **Rejected** — NPS cost outweighed accuracy on 4M pos. |
| `gen2_kb4_256` | `gen1.bin` (4M) | `(768*4->256)x2->8`, best-val | −23 ± 67 Elo vs `gen2` | **Rejected** — slightly slower without Finny table. |
| `gen2_256_w15` | `gen1.bin` (4M) | `(768->256)x2->8`, `wdl=0.15, scale=400` | −127 ± 72 Elo vs `gen2` | **Rejected** — confirmed WDL noise hurts at 43k games. |
| `gen2_256_k220_w25` | `gen1.bin` (4M) | `(768->256)x2->8`, `wdl=0.25, scale=220` | −108 ± 75 Elo vs `gen2` | **Rejected** — even with matched `K=220`, WDL overfit by epoch 4. |

## 5. Search v3 & Generation 3 NNUE

**Search v3** added Singular Extensions + Multi-Cut, 2-ply Follow-up Continuation History, Check-safe LMP/Futility/History Leaf Pruning, and 50-move draw dampening.
With Search v3, `gen2` scored **65.0% (+108 ± 85 Elo, LOS 99.6%)** vs `sf:2700` (60 games) and **25.0% (−191 Elo)** vs `sf:3000` (60 games) — both independent anchors pinning it at **~2,808–2,809 Stockfish UCI Elo**.

Generated `data/gen2.bin`: **3,501,931** deeper quiet positions from **38,572** Gen 2 + Search v3 games at 7,000 nodes/move (`logs/datagen_gen2.log`), bringing our pure NNUE self-play corpus (`gen1 + gen2`) to **7,503,459 positions** across **82,193 games**.

| Candidate | Data | Architecture | Result vs `gen2` (50 ms, 60g) | Verdict |
|---|---|---|---|---|
| `gen3_256` | `gen2.bin` (3.5M) | `(768->256)x2->8` | +17 ± 68 Elo (52.5%) | Superseded by 7.5M combined NNUE corpus |
| `gen3_256_comb` | `gen1+gen2` (7.5M) | `(768->256)x2->8` | **+59 ± 61 Elo (58.3%, LOS 97.4%)** | Strong (3.71M nps, 392 KB) |
| `gen3_kb4_256` | `gen1+gen2` (7.5M) | `(768*4->256)x2->8` | −12 ± 67 Elo (48.3%) | **Rejected** — still peaked at epoch 4 |
| `gen3_384` (**`gen3`**) | `gen1+gen2` (7.5M) | `(768->384)x2->8` | **+59 ± 67 Elo (58.3%, LOS 96.1%)** | **Champion** — tied `gen3_256_comb` at 50 ms (49.5%, 100g) and edged ahead at 150 ms (51.7%, 60g) at 3.39M nps (588 KB) |

### Final Evaluation of `nets/gen3.nnue` (Search v3)

| Opponent | Time / Move | Games | W / D / L | Score | Elo (95% CI) | Log |
|---|---|---|---|---|---|---|
| `classical` (Search v3) | 50 ms | 100 | 69 / 16 / 15 | **77.0%** | **+210 ± 73** | `logs/final_gen3_vs_classical.txt` |
| `nnue:gen2` | 50 ms | 60 | 22 / 26 / 12 | **58.3%** | **+59 ± 67** | `logs/gen3_384_vs_gen2.txt` |
| Stockfish 18 `UCI_Elo 2700` | 50 ms | 60 | 41 / 5 / 14 | **72.5%** | **+168 ± 97** | `logs/final_gen3_vs_sf2700_50ms.txt` |
| Stockfish 18 `UCI_Elo 2900` | 50 ms | 60 | 10 / 18 / 32 | 31.7% | −134 ± 78 | `logs/final_gen3_vs_sf2900_50ms.txt` |
| Stockfish 18 `UCI_Elo 2900` | 150 ms | 40 | 8 / 14 / 18 | **37.5%** | **−89 ± 90** | `logs/final_gen3_vs_sf2900_150ms.txt` |
| Stockfish 18 `UCI_Elo 3000` | 150 ms | 40 | 2 / 11 / 27 | 18.8% | −255 ± 110 | `logs/final_gen3_vs_sf3000_150ms.txt` |

## 6. Breaking 3,000 Elo: Search v4 + Pairwise-CReLU + Finny Tables + Modal H100 (200M Positions)

Upgraded the engine to **Search v4** (4-way 64-byte Cache-Line Clustered TT + ARM64 prefetching, staged TT-first move execution, lazy `gives_check_fast` before `make_move`, Multi-Correction History across pawn + White/Black non-pawn + continuation hashes, ProbCut, Cut-node LMR, and Node-TM) and **NNUE v2** (CHNN v2 Pairwise-CReLU `clamp(a) * clamp(b)` with 6-instruction NEON SIMD + Finny Tables for 8 horizontally mirrored king buckets).

Built a fused native CUDA trainer (`tools/modal_train.py`) running on **Modal NVIDIA H100 80GB HBM3** GPUs, streaming pregenerated Bullet S2 shards (`linrock/bullet-training-data`) onto a persistent Modal Volume and training incrementally across **200,000,000 positions**:
- **Scale calibration discovery**: Initial `h100_v1` (trained with `scale=400`, unfiltered `±30,000` mate scores, and rising `wdl 0.35→0.50`) crushed `gen3` by **+191 Elo at depth 1** (`+29 =2 -9`), yet lost `−319 Elo` at 50 ms because its 2.6× inflated centipawns (`−2,579 cp` for a Queen) broke RFP/Futility/ProbCut margins. Simply patching the 4-byte header `scale` from `400 → 140` produced a **+446 Elo swing** (`−319 → +127 Elo` vs `gen3`).
- **Stage 1 (`h100_v2`, 100M Shard 1 positions)**: Filtered `|score| > 3500`, fixed `wdl=0.30`, 20 cosine-decayed epochs (`lr 0.001 → 0.00002`, **8.01M pos/s**, 253 s on H100, best ep 16 `val=0.011956`, `scale=160`).
- **Stage 2 Incremental (`h100_v3`, +100M Shard 2 positions = 200M total)**: Warm-started from `h100_v2`, 12 epochs (`lr 0.0003 → 0.00001`, `wdl=0.35`, **7.99M pos/s**, 150 s on H100, best ep 11 `val=0.013166`, `scale=150`). Runs at **2.32M nps** single-core with Finny Tables.
- **Speed ablation (`h100_512_v2`, 6.3 MB, `HL=512, KB=8`, 200M pos)**: Trained at **15.0M pos/s** on H100, searches at **2.65M nps**; scored **49.2% (−6 ± 74 Elo)** vs Stockfish 3000 (100 ms), trailing `h100_v3` by `−29 Elo` head-to-head (`+11 =33 -16`) because `h100_v3` shrinks the depth-11 tree by 16%.

### Final 3,000+ Elo Benchmarks (Single-Threaded, Search v4)

| Engine | Opponent | Time / Move | Games | W / D / L | Score | Elo (95% CI) | Log |
|---|---|---|---|---|---|---|---|
| `h100_v2` (100M) | `nnue:gen3` | 50 ms | 40 | 20 / 12 / 8 | **65.0%** | **+108 ± 95** | `logs/h100_v2_vs_gen3.txt` |
| `h100_v2` (100M) | Stockfish `sf:2900` | 50 ms | 40 | 18 / 8 / 14 | **55.0%** | **+35 ± 99** | `logs/h100_v2_vs_sf2900.txt` |
| `h100_v2` (100M) | Stockfish `sf:3100` | 150 ms | 40 | 7 / 21 / 12 | **43.8%** | **−44 ± 75** | `logs/h100_v2_vs_sf3100_150ms.txt` |
| `h100_512_v2` (200M) | Stockfish `sf:3000` | 100 ms | 60 | 20 / 19 / 21 | **49.2%** | **−6 ± 74** | `logs/h100_512_vs_sf3000_100ms.txt` |
| **`h100_v3` (200M)** | `classical` (Search v4) | 50 ms | 40 | 32 / 5 / 3 | **86.3%** | **+319 ± 158** | `logs/h100_v3_vs_classical.txt` |
| **`h100_v3` (200M)** | `h100_v2` (100M) | 50 ms | 60 | 15 / 33 / 12 | **52.5%** | +17 ± 60 | `logs/h100_v3_vs_v2.txt` |
| **`h100_v3` (200M)** | `h100_512_v2` (200M) | 50 ms | 60 | 16 / 33 / 11 | **54.2%** | +29 ± 60 | `logs/h100_512_vs_1024_50ms.txt` |
| **`h100_v3` (200M)** | Stockfish `sf:3100` | 150 ms | 60 | 7 / 30 / 23 | **36.7%** | **−95 ± 63** | `logs/h100_v3_vs_sf3100_150ms.txt` |
| **`h100_v3` (200M)** | Stockfish `sf:3190` | 150 ms | 40 | 6 / 14 / 20 | **32.5%** | **−127 ± 92** | `logs/h100_v3_vs_sf3190_150ms.txt` |

## 7. Breaking 3,200 (1T) & 3,370+ Elo (4T): Search v5 + SPSA + Lockless Lazy SMP + 1B-Position NNUE

Upgraded the engine across three coordinated fronts (full academic write-up in [`docs/paper.pdf`](paper.pdf) / [`docs/paper.tex`](paper.tex)):

1. **Search v5 + Parallel SPSA Tuning (`src/search.rs`, `src/bin/spsa.rs`)**:
   - Implemented full staged `MovePicker` (`TT -> Good Captures -> Killers/CM -> Quiets -> Bad Captures`), Threat-Aware Quiet History (`[2][64][64][4]` indexed by `(from_attacked << 1) | to_attacked`), 4-ply Continuation History, Quiet SEE Pruning (`-18 * depth^2`), pentanomial SPRT (`src/elo.rs`), and 241,670-opening `UHO_4060_v4.epd` support.
   - Ran 300 iterations of parallel SPSA (8,400 UHO games at 18k nodes/move on 14 threads in 841 s, `logs/spsa_v5.log`), jointly tuning 25 parameters including `eval_scale_pct` (`100 -> 74`).
   - **Held-out verification (`logs/spsa_verify_v5.log`)**: **+184.0 ± 36.6 Elo** (`+114 =69 -17`, 74.25%, LOS 100.0%, zero `LL` pairs) over untuned Search v5 at 30 ms/move.
   - **Single-threaded Stockfish 3190 breakthrough (`logs/v5_spsa_h100v3_vs_sf3190.log`)**: `h100_v3` (1T) jumped from 32.5% (−127 Elo) to **56.7% (+46.6 ± 66.1 Elo, `+23 =22 -15`, LOS 92.2%) → ~3,237 Stockfish 18 UCI Elo (1 thread)**.

2. **Lockless Lazy SMP Multi-Threading (`src/search.rs`, `src/uci.rs`, `src/arena.rs`)**:
   - Converted the 4-way 64-byte cache-line clustered TT to Hyatt/Stockfish XOR-key lockless `AtomicU64` slots (`Ordering::Relaxed`) with persistent helper `Searcher`s, depth/aspiration diversification, thread voting, UCI `Threads`, and arena `:t=N` specs.
   - **4T vs 1T scaling (`logs/smp_4t_vs_1t.log`)**: **+168.4 ± 36.6 Elo** (`+39 =38 -3`, 72.5%, penta `[0, 0, 8, 28, 4]`, LOS 100.0%).

3. **1.0-Billion-Position 4-Architecture NNUE Study (`CHNN v2` vs `CHNN v3` Deep `sdot` + Threat Inputs)**:
   - Implemented **CHNN v3** (`(768*16 [+ 3*768 threats] -> HL)x2 -> 16 (CReLU+SCReLU+skip) -> 32 -> 1`, `ob=8`) with stable inline-asm ARM64 `sdot` NNZ-transposed matmul and fused threat-pairwise quantization, plus fused multi-shard CUDA trainer `tools/modal_train_v3.py`.
   - Trained all four architectures identically across **10 Bullet S2 shards (1.0 billion positions, 2.0B updates, WSD LR, WDL=0.35)** on Modal A100/H100/H200 GPUs (4.5–13.1M pos/s):
     - `h100_v4.nnue` (v3 Deep+Threats `16×1024`, 30.0 MB): **`val = 0.010010`**, **+183.2 ± 57.2 Elo** over `h100_v3` at equal 20k nodes (`logs/v4_vs_v3_nodes.log`), `0.70M nps` (−20.3 Elo at 50 ms).
     - `h100_v5_deep.nnue` (v3 Deep `16×1024`, no threats, 25.3 MB): `val = 0.010526`, `1.14M nps`, +23.2 Elo over `h100_v3` at 50 ms; 56.7% (1T) and **73.8% (+179.5 Elo)** (4T) vs `sf:3190`.
     - `h100_v5_512t.nnue` (v3 Deep+Threats `16×512`, 15.0 MB): `val = 0.010674`, `1.10M nps`, +20.3 Elo over `h100_v3` at 50 ms.
     - **`h100_v5_v2_1b.nnue`** (v2 Pairwise `16×1024`, 25.2 MB): `val = 0.011467`, **`1.60M nps`**, **wall-clock champion** — beats `h100_v5_deep` **+70.4 ± 36.8 Elo** (`+42 =60 -18`, LOS 100.0%)!

### Final Champion Benchmarks (`h100_v5_v2_1b.nnue` + SPSA Search v5 + Lazy SMP)

| Engine | Opponent | Time / Move | Games | W / D / L | Score | Elo (95% CI) | Log |
|---|---|---|---|---|---|---|---|
| `h100_v5_v2_1b` (1T) | `classical` (Search v5) | 50 ms | 60 | 57 / 3 / 0 | **97.5%** | **+636.4 ± 547.5** | `logs/final_v2_1b_vs_classical.log` |
| `h100_v5_v2_1b` (1T) | `h100_v5_deep` (1T) | 50 ms UHO | 120 | 42 / 60 / 18 | **60.0%** | **+70.4 ± 36.8** | `logs/v5_v2_1b_vs_v5_deep_50ms.log` |
| `h100_v5_v2_1b` (1T) | Stockfish `sf:3190` | 150 ms | 60 | 19 / 28 / 13 | **55.0%** | **+34.9 ± 62.2** | `logs/final_v2_1b_1t_vs_sf3190.log` |
| **`h100_v5_v2_1b` (4T)** | Stockfish `sf:3190` | 150 ms | 40 | 21 / 17 / 2 | **73.8%** | **+179.5 ± 81.6** | `logs/final_v2_1b_4t_vs_sf3190.log` |
| **`h100_v5_v2_1b` (4T)** | **Stockfish 18 Full (1T)** | 150 ms | 40 | 0 / 20 / 20 | **25.0%** | **−190.8 ± 65.5** | `logs/final_v2_1b_4t_vs_sf_full_1t.log` |
