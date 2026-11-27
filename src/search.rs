//! Modern Alpha-Beta Chess Search Engine (v5):
//! - 4-way 64-byte Cache-Line Clustered Transposition Table + ARM64/x86 software prefetching
//! - Full Staged MovePicker (TT -> Good Captures -> Killers/Countermove -> Quiets -> Bad Captures)
//! - Threat-Aware Quiet History (indexed by enemy attack map on from/to squares)
//! - 1-ply, 2-ply, and 4-ply Continuation History + Capture History
//! - Multi-Correction History (Pawn + White Non-Pawn + Black Non-Pawn + 1-ply Continuation)
//! - ProbCut, Singular Extensions & Multi-Cut, Cut-Node LMR with deeper re-search
//! - QSearch TT, TT-refined eval, NMP, RFP, Razoring, check-safe LMP/Futility/History/SEE pruning
//! - Runtime-tunable `SearchParams` (for SPSA tuning) + Node-based Time Management

use crate::bitboard::bb;
use crate::board::{mb_piece, Board};
use crate::endgame::{probe_endgame, EndgameEval};
use crate::eval::{evaluate_classical, see_ge, SEE_VAL};
use crate::nnue::{AccStack, NnueNet};
use crate::types::{Move, MoveList, Piece, MAX_MOVES};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const INF: i32 = 32_000;
pub const MATE_SCORE: i32 = 31_000;
pub const MATE_BOUND: i32 = 30_000;
pub const MAX_PLY: usize = 96;
const CORR_SIZE: usize = 16_384;
const CORR_MASK: usize = CORR_SIZE - 1;
const CORR_GRAIN: i32 = 256;
const CORR_MAX: i32 = 1024;

/// Tunable search parameters (exposed for SPSA tuning and A/B SPRT testing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchParams {
    pub rfp_mult: i32,
    pub rfp_improving: i32,
    pub razor_mult: i32,
    pub nmp_base: i32,      // /100
    pub nmp_depth_div: i32, // /100
    pub nmp_eval_div: i32,
    pub probcut_margin: i32,
    pub futility_mult: i32,
    pub futility_base: i32,
    pub lmp_base: i32, // /100
    pub see_quiet_prune: i32,
    pub see_cap_prune: i32,
    pub hist_prune_mult: i32,
    pub lmr_base: i32, // /100
    pub lmr_div: i32,  // /100
    pub lmr_hist_div: i32,
    pub lmr_deeper_margin: i32,
    pub se_depth_mult: i32, // /10
    pub se_double_margin: i32,
    pub asp_delta: i32,
    pub qs_delta_margin: i32,
    pub corr_pawn_w: i32,
    pub corr_np_w: i32,
    pub corr_cont_w: i32,
    pub eval_scale_pct: i32,
    pub corr_minor_w: i32,
    pub corr_major_w: i32,
    pub corr_cont2_w: i32,
    pub dual_threshold: i32,
    pub lmr_ttpv_w: i32,
    pub lmr_cut_w: i32,
    pub lmr_imp_w: i32,
    pub lmr_ttcap_w: i32,
    pub lmr_corr_w: i32,
    pub rfp_ttpv: i32,
    pub hist_bonus_mult: i32,
    pub hist_malus_mult: i32,
    pub pawn_hist_w: i32,
}

impl Default for SearchParams {
    fn default() -> Self {
        // Multi-stage SPSA-tuned: v5 (8.4k UHO) + v6 (5.6k UHO) + v7 38-param (8.4k UHO, 22k nodes)
        Self {
            rfp_mult: 71,
            rfp_improving: 56,
            razor_mult: 240,
            nmp_base: 303,
            nmp_depth_div: 367,
            nmp_eval_div: 204,
            probcut_margin: 211,
            futility_mult: 107,
            futility_base: 78,
            lmp_base: 328,
            see_quiet_prune: 15,
            see_cap_prune: 87,
            hist_prune_mult: 2702,
            lmr_base: 74,
            lmr_div: 236,
            lmr_hist_div: 5724,
            lmr_deeper_margin: 67,
            se_depth_mult: 22,
            se_double_margin: 30,
            asp_delta: 17,
            qs_delta_margin: 234,
            corr_pawn_w: 32,
            corr_np_w: 28,
            corr_cont_w: 20,
            eval_scale_pct: 70,
            corr_minor_w: 21,
            corr_major_w: 21,
            corr_cont2_w: 19,
            dual_threshold: 791,
            lmr_ttpv_w: 472,
            lmr_cut_w: 874,
            lmr_imp_w: 826,
            lmr_ttcap_w: 685,
            lmr_corr_w: 488,
            rfp_ttpv: 54,
            hist_bonus_mult: 294,
            hist_malus_mult: 229,
            pawn_hist_w: 101,
        }
    }
}

pub const NUM_PARAMS: usize = 38;

pub struct ParamMeta {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
    pub c_end: f64,
}

pub const PARAM_META: [ParamMeta; NUM_PARAMS] = [
    ParamMeta { name: "rfp_mult", min: 35.0, max: 130.0, c_end: 6.0 },
    ParamMeta { name: "rfp_improving", min: 0.0, max: 140.0, c_end: 8.0 },
    ParamMeta { name: "razor_mult", min: 100.0, max: 420.0, c_end: 18.0 },
    ParamMeta { name: "nmp_base", min: 200.0, max: 450.0, c_end: 15.0 },
    ParamMeta { name: "nmp_depth_div", min: 220.0, max: 650.0, c_end: 25.0 },
    ParamMeta { name: "nmp_eval_div", min: 100.0, max: 380.0, c_end: 16.0 },
    ParamMeta { name: "probcut_margin", min: 100.0, max: 360.0, c_end: 16.0 },
    ParamMeta { name: "futility_mult", min: 45.0, max: 180.0, c_end: 9.0 },
    ParamMeta { name: "futility_base", min: 0.0, max: 180.0, c_end: 10.0 },
    ParamMeta { name: "lmp_base", min: 150.0, max: 600.0, c_end: 25.0 },
    ParamMeta { name: "see_quiet_prune", min: 8.0, max: 50.0, c_end: 3.0 },
    ParamMeta { name: "see_cap_prune", min: 30.0, max: 160.0, c_end: 8.0 },
    ParamMeta { name: "hist_prune_mult", min: 800.0, max: 5500.0, c_end: 250.0 },
    ParamMeta { name: "lmr_base", min: 25.0, max: 150.0, c_end: 8.0 },
    ParamMeta { name: "lmr_div", min: 130.0, max: 350.0, c_end: 12.0 },
    ParamMeta { name: "lmr_hist_div", min: 2500.0, max: 12000.0, c_end: 500.0 },
    ParamMeta { name: "lmr_deeper_margin", min: 20.0, max: 140.0, c_end: 8.0 },
    ParamMeta { name: "se_depth_mult", min: 10.0, max: 40.0, c_end: 2.0 },
    ParamMeta { name: "se_double_margin", min: 8.0, max: 90.0, c_end: 5.0 },
    ParamMeta { name: "asp_delta", min: 8.0, max: 45.0, c_end: 2.5 },
    ParamMeta { name: "qs_delta_margin", min: 80.0, max: 340.0, c_end: 16.0 },
    ParamMeta { name: "corr_pawn_w", min: 5.0, max: 120.0, c_end: 7.0 },
    ParamMeta { name: "corr_np_w", min: 5.0, max: 100.0, c_end: 6.0 },
    ParamMeta { name: "corr_cont_w", min: 0.0, max: 80.0, c_end: 5.0 },
    ParamMeta { name: "eval_scale_pct", min: 65.0, max: 140.0, c_end: 4.0 },
    ParamMeta { name: "corr_minor_w", min: 0.0, max: 90.0, c_end: 6.0 },
    ParamMeta { name: "corr_major_w", min: 0.0, max: 90.0, c_end: 6.0 },
    ParamMeta { name: "corr_cont2_w", min: 0.0, max: 80.0, c_end: 5.0 },
    ParamMeta { name: "dual_threshold", min: 200.0, max: 1200.0, c_end: 45.0 },
    ParamMeta { name: "lmr_ttpv_w", min: 100.0, max: 1200.0, c_end: 65.0 },
    ParamMeta { name: "lmr_cut_w", min: 300.0, max: 1500.0, c_end: 75.0 },
    ParamMeta { name: "lmr_imp_w", min: 200.0, max: 1400.0, c_end: 75.0 },
    ParamMeta { name: "lmr_ttcap_w", min: 100.0, max: 1300.0, c_end: 75.0 },
    ParamMeta { name: "lmr_corr_w", min: 0.0, max: 1024.0, c_end: 65.0 },
    ParamMeta { name: "rfp_ttpv", min: 0.0, max: 140.0, c_end: 10.0 },
    ParamMeta { name: "hist_bonus_mult", min: 120.0, max: 450.0, c_end: 20.0 },
    ParamMeta { name: "hist_malus_mult", min: 100.0, max: 420.0, c_end: 20.0 },
    ParamMeta { name: "pawn_hist_w", min: 25.0, max: 200.0, c_end: 12.0 },
];

impl SearchParams {
    pub fn to_vec(&self) -> [f64; NUM_PARAMS] {
        [
            self.rfp_mult as f64,
            self.rfp_improving as f64,
            self.razor_mult as f64,
            self.nmp_base as f64,
            self.nmp_depth_div as f64,
            self.nmp_eval_div as f64,
            self.probcut_margin as f64,
            self.futility_mult as f64,
            self.futility_base as f64,
            self.lmp_base as f64,
            self.see_quiet_prune as f64,
            self.see_cap_prune as f64,
            self.hist_prune_mult as f64,
            self.lmr_base as f64,
            self.lmr_div as f64,
            self.lmr_hist_div as f64,
            self.lmr_deeper_margin as f64,
            self.se_depth_mult as f64,
            self.se_double_margin as f64,
            self.asp_delta as f64,
            self.qs_delta_margin as f64,
            self.corr_pawn_w as f64,
            self.corr_np_w as f64,
            self.corr_cont_w as f64,
            self.eval_scale_pct as f64,
            self.corr_minor_w as f64,
            self.corr_major_w as f64,
            self.corr_cont2_w as f64,
            self.dual_threshold as f64,
            self.lmr_ttpv_w as f64,
            self.lmr_cut_w as f64,
            self.lmr_imp_w as f64,
            self.lmr_ttcap_w as f64,
            self.lmr_corr_w as f64,
            self.rfp_ttpv as f64,
            self.hist_bonus_mult as f64,
            self.hist_malus_mult as f64,
            self.pawn_hist_w as f64,
        ]
    }

    pub fn from_vec(v: &[f64; NUM_PARAMS]) -> Self {
        let c = |i: usize| v[i].clamp(PARAM_META[i].min, PARAM_META[i].max).round() as i32;
        Self {
            rfp_mult: c(0),
            rfp_improving: c(1),
            razor_mult: c(2),
            nmp_base: c(3),
            nmp_depth_div: c(4),
            nmp_eval_div: c(5),
            probcut_margin: c(6),
            futility_mult: c(7),
            futility_base: c(8),
            lmp_base: c(9),
            see_quiet_prune: c(10),
            see_cap_prune: c(11),
            hist_prune_mult: c(12),
            lmr_base: c(13),
            lmr_div: c(14),
            lmr_hist_div: c(15),
            lmr_deeper_margin: c(16),
            se_depth_mult: c(17),
            se_double_margin: c(18),
            asp_delta: c(19),
            qs_delta_margin: c(20),
            corr_pawn_w: c(21),
            corr_np_w: c(22),
            corr_cont_w: c(23),
            eval_scale_pct: c(24),
            corr_minor_w: c(25),
            corr_major_w: c(26),
            corr_cont2_w: c(27),
            dual_threshold: c(28),
            lmr_ttpv_w: c(29),
            lmr_cut_w: c(30),
            lmr_imp_w: c(31),
            lmr_ttcap_w: c(32),
            lmr_corr_w: c(33),
            rfp_ttpv: c(34),
            hist_bonus_mult: c(35),
            hist_malus_mult: c(36),
            pawn_hist_w: c(37),
        }
    }

    pub fn to_kv_string(&self) -> String {
        let v = self.to_vec();
        let parts: Vec<String> = (0..NUM_PARAMS).map(|i| format!("{}={}", PARAM_META[i].name, v[i] as i32)).collect();
        parts.join(",")
    }

    pub fn apply_kv_string(&mut self, s: &str) -> Result<(), String> {
        let mut v = self.to_vec();
        for item in s.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let (k, val_s) = item.split_once('=').ok_or_else(|| format!("bad param '{item}' (need k=v)"))?;
            let k = k.trim();
            let val: f64 = val_s.trim().parse().map_err(|_| format!("bad param value '{val_s}' for {k}"))?;
            let idx =
                PARAM_META.iter().position(|m| m.name == k).ok_or_else(|| format!("unknown search param '{k}'"))?;
            v[idx] = val;
        }
        *self = Self::from_vec(&v);
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

const NO_EVAL: i16 = i16::MIN;

/// Unpacked Transposition Table entry (fits in a single 64-bit atomic word + 64-bit XOR key).
#[derive(Clone, Copy, Default)]
pub struct TtEntry {
    pub key: u64,
    pub best_move: Move,
    pub score: i16,
    pub depth: i8,
    pub flags: u8, // bits 0..2 = Bound, bit 2 = tt_pv, bits 3..8 = age (5 bits)
    pub static_eval: i16,
}

impl TtEntry {
    #[inline(always)]
    fn bound(self) -> Bound {
        match self.flags & 3 {
            1 => Bound::Upper,
            2 => Bound::Lower,
            3 => Bound::Exact,
            _ => Bound::None,
        }
    }
    #[inline(always)]
    fn tt_pv(self) -> bool {
        (self.flags & 4) != 0
    }
    #[inline(always)]
    fn age(self) -> u8 {
        self.flags >> 3
    }
    #[inline(always)]
    fn pack_flags(b: Bound, tt_pv: bool, age: u8) -> u8 {
        (b as u8) | (u8::from(tt_pv) << 2) | ((age & 31) << 3)
    }
    #[inline(always)]
    fn pack_data(best_move: Move, score: i16, depth: i8, flags: u8, static_eval: i16) -> u64 {
        (best_move.0 as u64)
            | (((score as u16) as u64) << 16)
            | (((depth as u8) as u64) << 32)
            | ((flags as u64) << 40)
            | (((static_eval as u16) as u64) << 48)
    }
    #[inline(always)]
    fn unpack(key: u64, data: u64) -> Self {
        Self {
            key,
            best_move: Move((data & 0xFFFF) as u16),
            score: ((data >> 16) & 0xFFFF) as u16 as i16,
            depth: ((data >> 32) & 0xFF) as u8 as i8,
            flags: ((data >> 40) & 0xFF) as u8,
            static_eval: ((data >> 48) & 0xFFFF) as u16 as i16,
        }
    }
}

/// Lockless 16-byte atomic TT slot (Hyatt/Stockfish XOR-key integrity check).
#[repr(C)]
struct AtomicTtSlot {
    key_xor_data: AtomicU64,
    data: AtomicU64,
}

impl AtomicTtSlot {
    const fn new() -> Self {
        Self { key_xor_data: AtomicU64::new(0), data: AtomicU64::new(0) }
    }
}

/// 4-way associative 64-byte cache-line aligned lockless atomic TT cluster.
#[repr(C, align(64))]
struct TtCluster {
    entries: [AtomicTtSlot; 4],
}

pub struct TranspositionTable {
    clusters: Vec<TtCluster>,
    mask: usize,
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new_mb(mb: usize) -> Self {
        let bytes = mb.max(1) * 1024 * 1024;
        let count = (bytes / std::mem::size_of::<TtCluster>()).next_power_of_two() >> 1;
        let count = count.max(256);
        let mut clusters = Vec::with_capacity(count);
        for _ in 0..count {
            clusters.push(TtCluster {
                entries: [AtomicTtSlot::new(), AtomicTtSlot::new(), AtomicTtSlot::new(), AtomicTtSlot::new()],
            });
        }
        Self { clusters, mask: count - 1, age: AtomicU8::new(0) }
    }

    pub fn clear(&self) {
        for c in &self.clusters {
            for s in &c.entries {
                s.key_xor_data.store(0, Ordering::Relaxed);
                s.data.store(0, Ordering::Relaxed);
            }
        }
        self.age.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        let prev = self.age.load(Ordering::Relaxed);
        self.age.store(prev.wrapping_add(1) & 31, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let idx = (hash as usize) & self.mask;
        let ptr = unsafe { self.clusters.as_ptr().add(idx) };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptr, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        let _ = ptr;
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64) -> Option<TtEntry> {
        let cluster = unsafe { self.clusters.get_unchecked((hash as usize) & self.mask) };
        for s in &cluster.entries {
            let d = s.data.load(Ordering::Relaxed);
            let k = s.key_xor_data.load(Ordering::Relaxed);
            if (k ^ d) == hash {
                let e = TtEntry::unpack(hash, d);
                if e.bound() != Bound::None {
                    return Some(e);
                }
            }
        }
        None
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        hash: u64,
        depth: i32,
        score: i32,
        bound: Bound,
        best_move: Move,
        ply: usize,
        tt_pv: bool,
        static_eval: i16,
    ) {
        let idx = (hash as usize) & self.mask;
        let cluster = unsafe { self.clusters.get_unchecked(idx) };
        let cur_age = self.age.load(Ordering::Relaxed);

        let mut replace_i = 0usize;
        let mut worst_val = i32::MAX;
        let mut replace_entry = TtEntry::default();
        let mut same_key = false;

        for i in 0..4 {
            let d = cluster.entries[i].data.load(Ordering::Relaxed);
            let k = cluster.entries[i].key_xor_data.load(Ordering::Relaxed);
            let e = TtEntry::unpack(k ^ d, d);
            if e.key == hash {
                replace_i = i;
                replace_entry = e;
                same_key = true;
                break;
            }
            if e.bound() == Bound::None {
                replace_i = i;
                replace_entry = e;
                break;
            }
            let age_diff = cur_age.wrapping_sub(e.age()) & 31;
            let quality = (e.depth as i32) - 8 * (age_diff as i32);
            if quality < worst_val {
                worst_val = quality;
                replace_i = i;
                replace_entry = e;
            }
        }

        if !same_key
            || bound == Bound::Exact
            || depth as i8 + 3 >= replace_entry.depth
            || replace_entry.age() != cur_age
        {
            let stored_score = score_to_tt(score, ply) as i16;
            let mv = if best_move.is_none() && same_key { replace_entry.best_move } else { best_move };
            let ev = if static_eval == NO_EVAL && same_key { replace_entry.static_eval } else { static_eval };
            let pv_flag = tt_pv || (same_key && replace_entry.tt_pv());
            let new_data = TtEntry::pack_data(
                mv,
                stored_score,
                depth.clamp(-1, 127) as i8,
                TtEntry::pack_flags(bound, pv_flag, cur_age),
                ev,
            );
            let slot = &cluster.entries[replace_i];
            slot.data.store(new_data, Ordering::Relaxed);
            slot.key_xor_data.store(hash ^ new_data, Ordering::Relaxed);
        }
    }
}

#[inline(always)]
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

#[inline(always)]
fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

#[derive(Clone)]
pub enum EvalMode {
    Classical,
    Nnue(Arc<NnueNet>),
    /// Stockfish-style Dual-Net evaluation: tiny fast gate net for clear-cut nodes (`|eval| >= dual_threshold`),
    /// deep/big net lazily materialized only on close nodes (`|eval| < dual_threshold`).
    DualNnue {
        small: Arc<NnueNet>,
        big: Arc<NnueNet>,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    pub max_depth: i32,
    pub max_nodes: u64,
    pub soft_ms: Option<u64>,
    pub hard_ms: Option<u64>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self { max_depth: 64, max_nodes: u64::MAX, soft_ms: None, hard_ms: None }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PickStage {
    TtMove,
    GenTacticals,
    GoodTacticals,
    Killer1,
    Killer2,
    CounterMove,
    GenQuiets,
    Quiets,
    BadTacticals,
    EvasionsInit,
    Evasions,
    Done,
}

/// Full staged move picker: avoids generating quiet moves until TT, good captures, killers, and countermove are tried.
struct MovePicker {
    stage: PickStage,
    tt_move: Move,
    skipped_tt: Move,
    excl: Move,
    k0: Move,
    k1: Move,
    cm: Move,
    ml: MoveList,
    scores: [i32; MAX_MOVES],
    idx: usize,
    bad_caps: [Move; 64],
    bad_len: usize,
    bad_idx: usize,
}

impl MovePicker {
    #[inline]
    fn new(in_check: bool, tt_move: Move, skipped_tt: Move, excl: Move, k0: Move, k1: Move, cm: Move) -> Self {
        let stage = if in_check {
            PickStage::EvasionsInit
        } else if !tt_move.is_none() && tt_move != skipped_tt && tt_move != excl {
            PickStage::TtMove
        } else {
            PickStage::GenTacticals
        };
        Self {
            stage,
            tt_move,
            skipped_tt,
            excl,
            k0,
            k1,
            cm,
            ml: MoveList::new(),
            scores: [0; MAX_MOVES],
            idx: 0,
            bad_caps: [Move::NONE; 64],
            bad_len: 0,
            bad_idx: 0,
        }
    }

    #[inline(always)]
    fn pick_best(&mut self) -> Option<(Move, i32)> {
        if self.idx >= self.ml.len() {
            return None;
        }
        let mut best_i = self.idx;
        let mut best_s = self.scores[self.idx];
        for j in (self.idx + 1)..self.ml.len() {
            if self.scores[j] > best_s {
                best_s = self.scores[j];
                best_i = j;
            }
        }
        if best_i != self.idx {
            self.ml.swap(self.idx, best_i);
            self.scores.swap(self.idx, best_i);
        }
        let m = self.ml[self.idx];
        let sc = self.scores[self.idx];
        self.idx += 1;
        Some((m, sc))
    }

    /// Returns `Some((move, is_good_or_non_bad_cap, cached_quiet_hist_or_min))`.
    #[inline]
    fn next(&mut self, s: &Searcher, b: &Board, ply: usize, threat_map: &mut Option<u64>) -> Option<(Move, bool, i32)> {
        loop {
            match self.stage {
                PickStage::TtMove => {
                    self.stage = PickStage::GenTacticals;
                    if b.is_legal_tt_move(self.tt_move) {
                        self.skipped_tt = self.tt_move;
                        return Some((self.tt_move, true, i32::MIN));
                    }
                }
                PickStage::GenTacticals => {
                    b.tactical_moves(&mut self.ml);
                    let c = b.side.idx();
                    for (i, &m) in self.ml.as_slice().iter().enumerate() {
                        let mvv = score_capture(b, m);
                        let ch = if m.is_capture() {
                            let att_p = mb_piece(b.mailbox[m.from() as usize]).idx();
                            s.cap_hist[c * 6 + att_p][m.to() as usize][captured_piece_idx(b, m)] / 4
                        } else {
                            0
                        };
                        self.scores[i] = mvv + ch;
                    }
                    self.idx = 0;
                    self.stage = PickStage::GoodTacticals;
                }
                PickStage::GoodTacticals => {
                    while let Some((m, _)) = self.pick_best() {
                        if m == self.skipped_tt || m == self.excl {
                            continue;
                        }
                        if see_ge(b, m, 0) {
                            return Some((m, true, i32::MIN));
                        } else if self.bad_len < self.bad_caps.len() {
                            self.bad_caps[self.bad_len] = m;
                            self.bad_len += 1;
                        }
                    }
                    self.stage = PickStage::Killer1;
                }
                PickStage::Killer1 => {
                    self.stage = PickStage::Killer2;
                    let m = self.k0;
                    if !m.is_none()
                        && m != self.skipped_tt
                        && m != self.excl
                        && !m.is_tactical()
                        && b.is_legal_tt_move(m)
                    {
                        return Some((m, false, i32::MIN));
                    }
                }
                PickStage::Killer2 => {
                    self.stage = PickStage::CounterMove;
                    let m = self.k1;
                    if !m.is_none()
                        && m != self.k0
                        && m != self.skipped_tt
                        && m != self.excl
                        && !m.is_tactical()
                        && b.is_legal_tt_move(m)
                    {
                        return Some((m, false, i32::MIN));
                    }
                }
                PickStage::CounterMove => {
                    self.stage = PickStage::GenQuiets;
                    let m = self.cm;
                    if !m.is_none()
                        && m != self.k0
                        && m != self.k1
                        && m != self.skipped_tt
                        && m != self.excl
                        && !m.is_tactical()
                        && b.is_legal_tt_move(m)
                    {
                        return Some((m, false, i32::MIN));
                    }
                }
                PickStage::GenQuiets => {
                    b.quiet_moves(&mut self.ml);
                    let tm = *threat_map.get_or_insert_with(|| b.enemy_attacks());
                    for (i, &m) in self.ml.as_slice().iter().enumerate() {
                        let mp = mb_piece(b.mailbox[m.from() as usize]).idx();
                        self.scores[i] = s.quiet_history(b, ply, m, mp, tm);
                    }
                    self.idx = 0;
                    self.stage = PickStage::Quiets;
                }
                PickStage::Quiets => {
                    while let Some((m, sc)) = self.pick_best() {
                        if m == self.skipped_tt || m == self.excl || m == self.k0 || m == self.k1 || m == self.cm {
                            continue;
                        }
                        return Some((m, false, sc));
                    }
                    self.stage = PickStage::BadTacticals;
                }
                PickStage::BadTacticals => {
                    if self.bad_idx < self.bad_len {
                        let m = self.bad_caps[self.bad_idx];
                        self.bad_idx += 1;
                        return Some((m, false, i32::MIN));
                    }
                    self.stage = PickStage::Done;
                    return None;
                }
                PickStage::EvasionsInit => {
                    // When in check, tactical_moves generates ALL strictly legal evasions.
                    b.tactical_moves(&mut self.ml);
                    let c = b.side.idx();
                    let tm = *threat_map.get_or_insert_with(|| b.enemy_attacks());
                    for (i, &m) in self.ml.as_slice().iter().enumerate() {
                        if m == self.tt_move {
                            self.scores[i] = 2_000_000;
                        } else if m.is_tactical() {
                            let mvv = score_capture(b, m);
                            let ch = if m.is_capture() {
                                let att_p = mb_piece(b.mailbox[m.from() as usize]).idx();
                                s.cap_hist[c * 6 + att_p][m.to() as usize][captured_piece_idx(b, m)] / 4
                            } else {
                                0
                            };
                            self.scores[i] = if see_ge(b, m, 0) { 1_000_000 + mvv + ch } else { -100_000 + mvv + ch };
                        } else {
                            let mp = mb_piece(b.mailbox[m.from() as usize]).idx();
                            self.scores[i] = s.quiet_history(b, ply, m, mp, tm);
                        }
                    }
                    self.idx = 0;
                    self.stage = PickStage::Evasions;
                }
                PickStage::Evasions => {
                    while let Some((m, sc)) = self.pick_best() {
                        if m == self.skipped_tt || m == self.excl {
                            continue;
                        }
                        let qh = if !m.is_tactical() && m != self.tt_move { sc } else { i32::MIN };
                        return Some((m, true, qh));
                    }
                    self.stage = PickStage::Done;
                    return None;
                }
                PickStage::Done => return None,
            }
        }
    }
}

/// Searcher instance (reusable across moves; owns lockless 4-way clustered TT, Lazy SMP helpers,
/// multi-corrhist, threat-aware histories, and Finny-cached NNUE accumulator stack).
pub struct Searcher {
    pub tt: Arc<TranspositionTable>,
    pub eval: EvalMode,
    pub params: SearchParams,
    pub threads: usize,
    thread_id: usize,
    helpers: Vec<Box<Searcher>>,
    shared_stop: Option<Arc<AtomicBool>>,
    shared_nodes: Option<Arc<AtomicU64>>,
    flushed_nodes: u64,
    acc: Option<AccStack>,
    small_acc: Option<AccStack>,
    killers: [[Move; 2]; MAX_PLY],
    // Threat-aware butterfly history: [side][from][to][(from_attacked << 1) | to_attacked]
    history: Box<[[[[i16; 4]; 64]; 64]; 2]>,
    // Pawn-structure quiet history: [pawn_hash & 511][piece][to]
    pawn_hist: Box<[[[i16; 64]; 6]; 512]>,
    cap_hist: Box<[[[i32; 6]; 64]; 12]>,
    cont_hist: Box<[[[[i16; 64]; 6]; 64]; 6]>,
    follow_hist: Box<[[[[i16; 64]; 6]; 64]; 6]>,
    follow4_hist: Box<[[[[i16; 64]; 6]; 64]; 6]>,
    follow6_hist: Box<[[[[i16; 64]; 6]; 64]; 6]>,
    // Multi-Correction History tables (Pawn + W/B Non-Pawn + Minor + Major + 1-ply & 2-ply Cont)
    pawn_corrhist: Box<[[i32; CORR_SIZE]; 2]>,
    wnp_corrhist: Box<[[i32; CORR_SIZE]; 2]>,
    bnp_corrhist: Box<[[i32; CORR_SIZE]; 2]>,
    minor_corrhist: Box<[[i32; CORR_SIZE]; 2]>,
    major_corrhist: Box<[[i32; CORR_SIZE]; 2]>,
    cont_corrhist: Box<[[[i32; 64]; 6]; 2]>,
    cont2_corrhist: Box<[[[i32; 64]; 6]; 2]>,
    countermoves: [[Move; 64]; 64],
    static_evals: [i32; MAX_PLY],
    move_stack: [Move; MAX_PLY],
    piece_stack: [usize; MAX_PLY],
    threat_stack: [u8; MAX_PLY],
    excluded: [Move; MAX_PLY],
    root_nodes: Box<[[u64; 64]; 64]>,
    pv_table: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    rep_history: Vec<u64>,
    lmr: [[i32; 64]; 64],
    nodes: u64,
    stopped: bool,
    start: Instant,
    limits: SearchLimits,
    uci_info: bool,
}

/// Allocate a zero-initialized value directly on the heap with zero stack footprint.
/// Critical on Windows where the default main-thread stack is 1 MB.
#[inline]
fn box_zeroed<T>() -> Box<T> {
    let layout = std::alloc::Layout::new::<T>();
    assert!(layout.size() > 0);
    // SAFETY: Used exclusively for plain integer / nested array tables where all-zero bits are valid.
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr)
    }
}

fn build_lmr_table(params: &SearchParams) -> [[i32; 64]; 64] {
    // 1024-scale fixed-point LMR table for smooth fractional adjustments
    let base = params.lmr_base as f64 / 100.0;
    let div = (params.lmr_div as f64 / 100.0).max(0.5);
    let mut lmr = [[0i32; 64]; 64];
    for d in 1..64 {
        for m in 1..64 {
            lmr[d][m] = ((base + (d as f64).ln() * (m as f64).ln() / div) * 1024.0) as i32;
        }
    }
    lmr
}

impl Searcher {
    pub fn new(eval: EvalMode, tt_mb: usize) -> Self {
        Self::with_params(eval, tt_mb, SearchParams::default())
    }

    pub fn with_params(eval: EvalMode, tt_mb: usize, params: SearchParams) -> Self {
        let tt = Arc::new(TranspositionTable::new_mb(tt_mb));
        Self::with_shared_tt(eval, tt, params, 0)
    }

    fn with_shared_tt(eval: EvalMode, tt: Arc<TranspositionTable>, params: SearchParams, thread_id: usize) -> Self {
        let (acc, small_acc) = match &eval {
            EvalMode::Classical => (None, None),
            EvalMode::Nnue(net) => (Some(AccStack::new(net.hl, MAX_PLY + 8)), None),
            EvalMode::DualNnue { small, big } => {
                (Some(AccStack::new(big.hl, MAX_PLY + 8)), Some(AccStack::new(small.hl, MAX_PLY + 8)))
            }
        };
        let lmr = build_lmr_table(&params);
        Self {
            tt,
            eval,
            params,
            threads: 1,
            thread_id,
            helpers: Vec::new(),
            shared_stop: None,
            shared_nodes: None,
            flushed_nodes: 0,
            acc,
            small_acc,
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: box_zeroed(),
            pawn_hist: box_zeroed(),
            cap_hist: box_zeroed(),
            cont_hist: box_zeroed(),
            follow_hist: box_zeroed(),
            follow4_hist: box_zeroed(),
            follow6_hist: box_zeroed(),
            pawn_corrhist: box_zeroed(),
            wnp_corrhist: box_zeroed(),
            bnp_corrhist: box_zeroed(),
            minor_corrhist: box_zeroed(),
            major_corrhist: box_zeroed(),
            cont_corrhist: box_zeroed(),
            cont2_corrhist: box_zeroed(),
            countermoves: [[Move::NONE; 64]; 64],
            static_evals: [0; MAX_PLY],
            move_stack: [Move::NONE; MAX_PLY],
            piece_stack: [6; MAX_PLY],
            threat_stack: [0; MAX_PLY],
            excluded: [Move::NONE; MAX_PLY],
            root_nodes: box_zeroed(),
            pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            rep_history: Vec::with_capacity(512),
            lmr,
            nodes: 0,
            stopped: false,
            start: Instant::now(),
            limits: SearchLimits::default(),
            uci_info: false,
        }
    }

    /// Configure Lazy SMP thread count (1 = single-threaded, >1 = persistent helper workers sharing the lockless TT).
    pub fn set_threads(&mut self, n: usize) {
        let n = n.clamp(1, 64);
        self.threads = n;
        let want_helpers = n - 1;
        self.helpers.truncate(want_helpers);
        while self.helpers.len() < want_helpers {
            let tid = self.helpers.len() + 1;
            self.helpers.push(Box::new(Self::with_shared_tt(
                self.eval.clone(),
                Arc::clone(&self.tt),
                self.params,
                tid,
            )));
        }
    }

    pub fn set_params(&mut self, params: SearchParams) {
        self.params = params;
        self.lmr = build_lmr_table(&params);
        for h in &mut self.helpers {
            h.set_params(params);
        }
    }

    pub fn set_uci_info(&mut self, on: bool) {
        self.uci_info = on;
    }

    fn clear_local_tables(&mut self) {
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        for c in self.history.iter_mut() {
            for f in c.iter_mut() {
                for t in f.iter_mut() {
                    *t = [0; 4];
                }
            }
        }
        for p in self.pawn_hist.iter_mut() {
            for pc in p.iter_mut() {
                pc.fill(0);
            }
        }
        for a in self.cap_hist.iter_mut() {
            for b in a.iter_mut() {
                b.fill(0);
            }
        }
        for tbl in [&mut self.cont_hist, &mut self.follow_hist, &mut self.follow4_hist, &mut self.follow6_hist] {
            for a in tbl.iter_mut() {
                for b in a.iter_mut() {
                    for c in b.iter_mut() {
                        c.fill(0);
                    }
                }
            }
        }
        for c in 0..2 {
            self.pawn_corrhist[c].fill(0);
            self.wnp_corrhist[c].fill(0);
            self.bnp_corrhist[c].fill(0);
            self.minor_corrhist[c].fill(0);
            self.major_corrhist[c].fill(0);
            for p in 0..6 {
                self.cont_corrhist[c][p].fill(0);
                self.cont2_corrhist[c][p].fill(0);
            }
        }
        self.countermoves = [[Move::NONE; 64]; 64];
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.clear_local_tables();
        for h in &mut self.helpers {
            h.clear_local_tables();
        }
    }

    pub fn search(&mut self, board: &Board, game_history: &[u64], limits: SearchLimits) -> SearchResult {
        self.search_with_stop(board, game_history, limits, None)
    }

    pub fn search_with_stop(
        &mut self,
        board: &Board,
        game_history: &[u64],
        limits: SearchLimits,
        external_stop: Option<Arc<AtomicBool>>,
    ) -> SearchResult {
        self.tt.new_search();
        if self.threads <= 1 || self.helpers.is_empty() {
            return self.search_worker(board, game_history, limits, external_stop, None);
        }

        let stop = external_stop.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let tot_nodes = Arc::new(AtomicU64::new(0));
        let mut helpers = std::mem::take(&mut self.helpers);

        let helper_limits = SearchLimits {
            max_depth: limits.max_depth,
            max_nodes: limits.max_nodes,
            soft_ms: None,
            hard_ms: limits.hard_ms,
        };

        let (best_res, helper_results) = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(helpers.len());
            for h in helpers.iter_mut() {
                let stop_c = Arc::clone(&stop);
                let nodes_c = Arc::clone(&tot_nodes);
                let jh = std::thread::Builder::new()
                    .stack_size(8 * 1024 * 1024)
                    .spawn_scoped(s, move || {
                        h.search_worker(board, game_history, helper_limits, Some(stop_c), Some(nodes_c))
                    })
                    .expect("failed to spawn SMP helper thread");
                handles.push(jh);
            }
            let main_res =
                self.search_worker(board, game_history, limits, Some(Arc::clone(&stop)), Some(Arc::clone(&tot_nodes)));
            stop.store(true, Ordering::Relaxed);
            let mut hres = Vec::with_capacity(handles.len());
            for jh in handles {
                if let Ok(r) = jh.join() {
                    hres.push(r);
                }
            }
            (main_res, hres)
        });
        self.helpers = helpers;

        // Depth-guarded Lazy SMP thread voting: never allow a shallower helper with a horizon-inflated
        // score to override a deeper completed search! Only threads within 1 ply of `max_depth` may
        // vote, with cubic depth weighting and strict preference for deepest completed PV.
        let mut all_res = Vec::with_capacity(1 + helper_results.len());
        let mut sum_nodes = best_res.nodes;
        all_res.push(best_res);
        for hr in helper_results {
            sum_nodes += hr.nodes;
            if !hr.best_move.is_none() && hr.depth > 0 {
                all_res.push(hr);
            }
        }
        let max_depth = all_res.iter().map(|r| r.depth).max().unwrap_or(0);
        // Prefer strictly deepest completed threads; among ties at max_depth, use move consensus + score
        let mut best_idx = 0usize;
        let mut best_key = (i32::MIN, 0i32, i32::MIN);
        for (i, r) in all_res.iter().enumerate() {
            if r.score >= MATE_BOUND {
                if all_res[best_idx].score < MATE_BOUND || r.score > all_res[best_idx].score {
                    best_idx = i;
                    best_key = (i32::MAX, i32::MAX, r.score);
                }
                continue;
            }
            if best_key.0 == i32::MAX {
                continue;
            }
            if r.depth < max_depth {
                continue;
            }
            // Count how many threads (at depth >= max_depth - 1) agree on this exact move
            let consensus =
                all_res.iter().filter(|o| o.depth >= max_depth - 1 && o.best_move == r.best_move).count() as i32;
            let key = (r.depth, consensus, r.score);
            if key > best_key {
                best_key = key;
                best_idx = i;
            }
        }
        let mut chosen = all_res.swap_remove(best_idx);
        chosen.nodes = sum_nodes;
        chosen
    }

    fn search_worker(
        &mut self,
        board: &Board,
        game_history: &[u64],
        limits: SearchLimits,
        shared_stop: Option<Arc<AtomicBool>>,
        shared_nodes: Option<Arc<AtomicU64>>,
    ) -> SearchResult {
        self.nodes = 0;
        self.flushed_nodes = 0;
        self.stopped = false;
        self.shared_stop = shared_stop;
        self.shared_nodes = shared_nodes;
        self.start = Instant::now();
        self.limits = limits;
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        self.excluded = [Move::NONE; MAX_PLY];
        for r in self.root_nodes.iter_mut() {
            r.fill(0);
        }
        for c in 0..2 {
            for f in 0..64 {
                for t in 0..64 {
                    for th in 0..4 {
                        self.history[c][f][t][th] /= 2;
                    }
                }
            }
        }

        self.rep_history.clear();
        if game_history.is_empty() {
            self.rep_history.push(board.hash);
        } else {
            self.rep_history.extend_from_slice(game_history);
        }

        match &self.eval {
            EvalMode::Classical => {}
            EvalMode::Nnue(net) => {
                if let Some(acc) = &mut self.acc {
                    acc.init_root(net, board);
                }
            }
            EvalMode::DualNnue { small, big } => {
                if let Some(acc) = &mut self.acc {
                    acc.init_root(big, board);
                }
                if let Some(sacc) = &mut self.small_acc {
                    sacc.init_root(small, board);
                }
            }
        }

        let mut root_moves = MoveList::new();
        board.legal_moves(&mut root_moves);
        if root_moves.is_empty() {
            self.shared_stop = None;
            self.shared_nodes = None;
            return SearchResult {
                best_move: Move::NONE,
                score: if board.in_check() { -MATE_SCORE } else { 0 },
                depth: 0,
                nodes: 0,
                time_ms: 0,
                pv: vec![],
            };
        }

        let mut best_move = root_moves[0];
        let mut best_score = 0;
        let mut best_pv = vec![best_move];
        let mut completed_depth = 0;
        let mut bm_stability = 0u32;
        let mut prev_best_move = Move::NONE;
        let mut prev_best_score = 0i32;
        let max_d = limits.max_depth.min((MAX_PLY - 4) as i32);
        let is_clock_mode = matches!((limits.soft_ms, limits.hard_ms), (Some(s), Some(h)) if h > s);
        let is_main = self.thread_id == 0;

        for depth in 1..=max_d {
            // Lazy SMP helper depth diversification (skip periodic depths so threads explore distinct subtrees)
            if !is_main && depth >= 4 && ((depth as usize + self.thread_id) % 4 == 0) {
                continue;
            }
            let mut alpha = -INF;
            let mut beta = INF;
            let mut delta = self.params.asp_delta + if is_main { 0 } else { ((self.thread_id as i32) & 3) * 3 };

            if depth >= 5 {
                alpha = (best_score - delta).max(-INF);
                beta = (best_score + delta).min(INF);
            }

            let mut asp_depth = depth;
            let score = loop {
                let s = self.negamax(board, asp_depth, 0, alpha, beta, true, false);
                if self.stopped {
                    break s;
                }
                if s <= alpha {
                    alpha = (s - delta).max(-INF);
                    asp_depth = depth;
                    delta = delta.saturating_mul(2);
                } else if s >= beta {
                    beta = (s + delta).min(INF);
                    if asp_depth > 1 && s.abs() < MATE_BOUND {
                        asp_depth -= 1;
                    }
                    delta = delta.saturating_mul(2);
                } else {
                    break s;
                }
                if delta > 2000 {
                    alpha = -INF;
                    beta = INF;
                    asp_depth = depth;
                }
            };

            if self.stopped && depth > 1 {
                break;
            }

            completed_depth = depth;
            best_score = score;
            if self.pv_len[0] > 0 && !self.pv_table[0][0].is_none() {
                best_move = self.pv_table[0][0];
                best_pv = self.pv_table[0][..self.pv_len[0]].to_vec();
            }

            if is_main && self.uci_info {
                let elapsed = self.start.elapsed().as_millis().max(1) as u64;
                let reported_nodes = self
                    .shared_nodes
                    .as_ref()
                    .map(|sn| sn.load(Ordering::Relaxed) + (self.nodes - self.flushed_nodes))
                    .unwrap_or(self.nodes);
                let nps = reported_nodes * 1000 / elapsed;
                let pv_str: Vec<String> = best_pv.iter().map(|m| m.to_uci()).collect();
                let score_str = if best_score.abs() >= MATE_BOUND {
                    let plies = MATE_SCORE - best_score.abs();
                    let mate_in = (plies + 1) / 2;
                    if best_score > 0 {
                        format!("mate {mate_in}")
                    } else {
                        format!("mate -{mate_in}")
                    }
                } else {
                    format!("cp {best_score}")
                };
                println!(
                    "info depth {depth} score {score_str} nodes {reported_nodes} nps {nps} time {elapsed} pv {}",
                    pv_str.join(" ")
                );
            }

            if is_main {
                if best_move == prev_best_move {
                    bm_stability = bm_stability.saturating_add(1);
                } else {
                    bm_stability = 0;
                    prev_best_move = best_move;
                }
                let score_drop = if depth >= 5 { prev_best_score - best_score } else { 0 };
                prev_best_score = best_score;

                if best_score.abs() >= MATE_BOUND && depth >= 4 {
                    break;
                }
                if root_moves.len() == 1 && limits.soft_ms.is_some() && depth >= 4 {
                    break;
                }
                if let Some(soft) = limits.soft_ms {
                    let eff_soft = if is_clock_mode && depth >= 6 && self.nodes > 0 {
                        let bm_nodes = self.root_nodes[best_move.from() as usize][best_move.to() as usize];
                        let frac = bm_nodes as f64 / self.nodes as f64;
                        let node_scale = (1.45 - frac).clamp(0.45, 1.50);
                        // Best-move stability & eval-drop panic scaling
                        let stab_scale = match bm_stability {
                            0 => 1.25,
                            1 => 1.10,
                            2..=4 => 1.00,
                            5..=7 => 0.86,
                            _ => 0.74,
                        };
                        let drop_scale = if score_drop > 50 {
                            1.50
                        } else if score_drop > 25 {
                            1.25
                        } else {
                            1.00
                        };
                        ((soft as f64) * (node_scale * stab_scale * drop_scale).clamp(0.40, 2.00)) as u64
                    } else {
                        soft
                    };
                    if self.start.elapsed().as_millis() as u64 >= eff_soft {
                        break;
                    }
                }
            }
            if let Some(sn) = &self.shared_nodes {
                let cur = sn.load(Ordering::Relaxed) + (self.nodes - self.flushed_nodes);
                if cur >= limits.max_nodes {
                    break;
                }
            } else if self.nodes >= limits.max_nodes {
                break;
            }
        }

        if let Some(sn) = &self.shared_nodes {
            let rem = self.nodes - self.flushed_nodes;
            if rem > 0 {
                sn.fetch_add(rem, Ordering::Relaxed);
                self.flushed_nodes = self.nodes;
            }
        }
        self.shared_stop = None;
        self.shared_nodes = None;

        SearchResult {
            best_move,
            score: best_score,
            depth: completed_depth,
            nodes: self.nodes,
            time_ms: self.start.elapsed().as_millis() as u64,
            pv: best_pv,
        }
    }

    #[inline(always)]
    fn check_limits(&mut self) -> bool {
        if self.shared_stop.is_none() {
            if self.nodes >= self.limits.max_nodes {
                self.stopped = true;
                return true;
            }
            if (self.nodes & 2047) == 0 {
                if let Some(hard) = self.limits.hard_ms {
                    if self.start.elapsed().as_millis() as u64 >= hard {
                        self.stopped = true;
                        return true;
                    }
                }
            }
            return false;
        }
        if (self.nodes & 2047) == 0 {
            if let Some(stop) = &self.shared_stop {
                if stop.load(Ordering::Relaxed) {
                    self.stopped = true;
                    return true;
                }
            }
            if let Some(sn) = &self.shared_nodes {
                let delta = self.nodes - self.flushed_nodes;
                let tot = sn.fetch_add(delta, Ordering::Relaxed) + delta;
                self.flushed_nodes = self.nodes;
                if tot >= self.limits.max_nodes {
                    self.stopped = true;
                    if let Some(stop) = &self.shared_stop {
                        stop.store(true, Ordering::Relaxed);
                    }
                    return true;
                }
            }
            if let Some(hard) = self.limits.hard_ms {
                if self.start.elapsed().as_millis() as u64 >= hard {
                    self.stopped = true;
                    if let Some(stop) = &self.shared_stop {
                        stop.store(true, Ordering::Relaxed);
                    }
                    return true;
                }
            }
        }
        false
    }

    #[inline(always)]
    fn push_acc_move(&mut self, ply: usize, prev: &Board, next: &Board, m: Move) {
        if let Some(acc) = &mut self.acc {
            acc.push_move(ply, prev, next, m);
        }
        if let Some(sacc) = &mut self.small_acc {
            sacc.push_move(ply, prev, next, m);
        }
    }

    #[inline(always)]
    fn push_acc_null(&mut self, ply: usize, next: &Board) {
        if let Some(acc) = &mut self.acc {
            acc.push_null(ply, next);
        }
        if let Some(sacc) = &mut self.small_acc {
            sacc.push_null(ply, next);
        }
    }

    /// Zero-cost Stockfish-style simple material balance (in centipawns, STM perspective)
    /// used to gate Dual-Net evaluation with zero accumulator overhead on balanced nodes.
    #[inline(always)]
    fn simple_material_eval(b: &Board) -> i32 {
        let wp = b.piece_bb(crate::types::Color::White, Piece::Pawn).count_ones() as i32;
        let wn = b.piece_bb(crate::types::Color::White, Piece::Knight).count_ones() as i32;
        let wb = b.piece_bb(crate::types::Color::White, Piece::Bishop).count_ones() as i32;
        let wr = b.piece_bb(crate::types::Color::White, Piece::Rook).count_ones() as i32;
        let wq = b.piece_bb(crate::types::Color::White, Piece::Queen).count_ones() as i32;
        let bp = b.piece_bb(crate::types::Color::Black, Piece::Pawn).count_ones() as i32;
        let bn = b.piece_bb(crate::types::Color::Black, Piece::Knight).count_ones() as i32;
        let bb_c = b.piece_bb(crate::types::Color::Black, Piece::Bishop).count_ones() as i32;
        let br = b.piece_bb(crate::types::Color::Black, Piece::Rook).count_ones() as i32;
        let bq = b.piece_bb(crate::types::Color::Black, Piece::Queen).count_ones() as i32;
        let diff = (wp - bp) * 100 + (wn - bn) * 320 + (wb - bb_c) * 340 + (wr - br) * 510 + (wq - bq) * 980;
        if b.side == crate::types::Color::White {
            diff
        } else {
            -diff
        }
    }

    #[inline(always)]
    fn material_phase_scale(&self, b: &Board, ev: i32) -> i32 {
        let knights = (b.pieces[Piece::Knight.idx()]).count_ones() as i32;
        let bishops = (b.pieces[Piece::Bishop.idx()]).count_ones() as i32;
        let rooks = (b.pieces[Piece::Rook.idx()]).count_ones() as i32;
        let queens = (b.pieces[Piece::Queen.idx()]).count_ones() as i32;
        let mat = (knights + bishops) * 3 + rooks * 5 + queens * 10; // 0..62
        let scale = (self.params.eval_scale_pct * 1024 / 100) + (mat - 32) * 4;
        (ev * scale / 1024).clamp(-MATE_BOUND + 100, MATE_BOUND - 100)
    }

    #[inline(always)]
    fn raw_eval(&mut self, b: &Board, ply: usize) -> (i32, bool) {
        match &self.eval {
            EvalMode::Classical => (evaluate_classical(b), false),
            EvalMode::Nnue(net) => {
                let acc = self.acc.as_mut().unwrap();
                let ev = acc.evaluate(net, ply, b);
                (self.material_phase_scale(b, ev), false)
            }
            EvalMode::DualNnue { small, big } => {
                // True Stockfish Dual-Net gating: check zero-cost material balance FIRST!
                // Balanced nodes pay ZERO small-net overhead; lopsided nodes skip big-net completely!
                if Self::simple_material_eval(b).abs() >= self.params.dual_threshold {
                    let sacc = self.small_acc.as_mut().unwrap();
                    let s_ev = sacc.evaluate(small, ply, b);
                    return (self.material_phase_scale(b, s_ev), true);
                }
                let acc = self.acc.as_mut().unwrap();
                let b_ev = acc.evaluate(big, ply, b);
                (self.material_phase_scale(b, b_ev), false)
            }
        }
    }

    /// Returns `(corrected_eval, uncorr_static_eval_for_tt, used_small, corrplexity)`.
    #[inline(always)]
    fn corrected_eval(&mut self, b: &Board, ply: usize, tt_static: i16) -> (i32, i16, bool, i32) {
        let eg = probe_endgame(b);
        if let EndgameEval::Exact(sc) = eg {
            return (sc, sc.clamp(-30000, 30000) as i16, false, 0);
        }
        let (mut raw, used_small) =
            if tt_static != NO_EVAL { (tt_static as i32, false) } else { self.raw_eval(b, ply) };
        let cached_raw = if used_small { NO_EVAL } else { raw.clamp(-30000, 30000) as i16 };
        if let EndgameEval::Scale(num, den) = eg {
            raw = raw * num / den;
        }
        if b.halfmove > 16 {
            raw = raw * (200 - b.halfmove as i32) / 200;
        }
        let c = b.side.idx();
        let p_corr = self.pawn_corrhist[c][(b.pawn_hash as usize) & CORR_MASK];
        let w_corr = self.wnp_corrhist[c][(b.non_pawn_hash[0] as usize) & CORR_MASK];
        let b_corr = self.bnp_corrhist[c][(b.non_pawn_hash[1] as usize) & CORR_MASK];
        let min_corr = self.minor_corrhist[c][(b.minor_hash as usize) & CORR_MASK];
        let maj_corr = self.major_corrhist[c][(b.major_hash as usize) & CORR_MASK];
        let c_corr = if ply >= 1 && self.piece_stack[ply - 1] < 6 && !self.move_stack[ply - 1].is_none() {
            self.cont_corrhist[c][self.piece_stack[ply - 1]][self.move_stack[ply - 1].to() as usize]
        } else {
            0
        };
        let c2_corr = if ply >= 2 && self.piece_stack[ply - 2] < 6 && !self.move_stack[ply - 2].is_none() {
            self.cont2_corrhist[c][self.piece_stack[ply - 2]][self.move_stack[ply - 2].to() as usize]
        } else {
            0
        };
        let corr = (p_corr * self.params.corr_pawn_w
            + (w_corr + b_corr) * self.params.corr_np_w
            + min_corr * self.params.corr_minor_w
            + maj_corr * self.params.corr_major_w
            + c_corr * self.params.corr_cont_w
            + c2_corr * self.params.corr_cont2_w)
            / (120 * CORR_GRAIN);
        ((raw + corr).clamp(-MATE_BOUND + 100, MATE_BOUND - 100), cached_raw, used_small, corr.abs())
    }

    #[inline(always)]
    fn update_corrhist(&mut self, b: &Board, ply: usize, diff: i32, depth: i32) {
        let c = b.side.idx();
        let weight = (depth * depth + 2 * depth + 1).min(256);
        let scaled = (diff * CORR_GRAIN).clamp(-CORR_MAX * CORR_GRAIN, CORR_MAX * CORR_GRAIN);
        update_corr_slot(&mut self.pawn_corrhist[c][(b.pawn_hash as usize) & CORR_MASK], scaled, weight);
        update_corr_slot(&mut self.wnp_corrhist[c][(b.non_pawn_hash[0] as usize) & CORR_MASK], scaled, weight);
        update_corr_slot(&mut self.bnp_corrhist[c][(b.non_pawn_hash[1] as usize) & CORR_MASK], scaled, weight);
        update_corr_slot(&mut self.minor_corrhist[c][(b.minor_hash as usize) & CORR_MASK], scaled, weight);
        update_corr_slot(&mut self.major_corrhist[c][(b.major_hash as usize) & CORR_MASK], scaled, weight);
        if ply >= 1 && self.piece_stack[ply - 1] < 6 && !self.move_stack[ply - 1].is_none() {
            let pp = self.piece_stack[ply - 1];
            let pt = self.move_stack[ply - 1].to() as usize;
            update_corr_slot(&mut self.cont_corrhist[c][pp][pt], scaled, weight);
        }
        if ply >= 2 && self.piece_stack[ply - 2] < 6 && !self.move_stack[ply - 2].is_none() {
            let pp2 = self.piece_stack[ply - 2];
            let pt2 = self.move_stack[ply - 2].to() as usize;
            update_corr_slot(&mut self.cont2_corrhist[c][pp2][pt2], scaled, weight);
        }
    }

    #[inline]
    fn is_repetition(&self, hash: u64, halfmove: u8) -> bool {
        let len = self.rep_history.len();
        if len < 3 || halfmove < 4 {
            return false;
        }
        let limit = (halfmove as usize).min(len - 1);
        let start = len - 1 - limit;
        let mut i = len - 3;
        while i >= start {
            if self.rep_history[i] == hash {
                return true;
            }
            if i < 2 {
                break;
            }
            i -= 2;
        }
        false
    }

    #[inline(always)]
    fn quiet_history(&self, b: &Board, ply: usize, m: Move, moved_p: usize, threat_map: u64) -> i32 {
        let from = m.from();
        let to = m.to();
        let th = threat_idx(from, to, threat_map);
        let mut h = self.history[b.side.idx()][from as usize][to as usize][th] as i32;
        h +=
            (self.pawn_hist[(b.pawn_hash as usize) & 511][moved_p][to as usize] as i32) * self.params.pawn_hist_w / 100;
        if ply >= 1 {
            let pm = self.move_stack[ply - 1];
            let pp = self.piece_stack[ply - 1];
            if !pm.is_none() && pp < 6 {
                h += self.cont_hist[pp][pm.to() as usize][moved_p][to as usize] as i32;
            }
        }
        if ply >= 2 {
            let pm2 = self.move_stack[ply - 2];
            let pp2 = self.piece_stack[ply - 2];
            if !pm2.is_none() && pp2 < 6 {
                h += self.follow_hist[pp2][pm2.to() as usize][moved_p][to as usize] as i32;
            }
        }
        if ply >= 4 {
            let pm4 = self.move_stack[ply - 4];
            let pp4 = self.piece_stack[ply - 4];
            if !pm4.is_none() && pp4 < 6 {
                h += (self.follow4_hist[pp4][pm4.to() as usize][moved_p][to as usize] as i32) / 2;
            }
        }
        if ply >= 6 {
            let pm6 = self.move_stack[ply - 6];
            let pp6 = self.piece_stack[ply - 6];
            if !pm6.is_none() && pp6 < 6 {
                h += (self.follow6_hist[pp6][pm6.to() as usize][moved_p][to as usize] as i32) / 2;
            }
        }
        h
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn update_quiet_histories(
        &mut self,
        c: usize,
        pawn_hash: u64,
        ply: usize,
        m: Move,
        moved_p: usize,
        threat_map: u64,
        bonus: i32,
    ) {
        let from = m.from();
        let to = m.to();
        let th = threat_idx(from, to, threat_map);
        update_history_i16(&mut self.history[c][from as usize][to as usize][th], bonus);
        update_history_i16(&mut self.pawn_hist[(pawn_hash as usize) & 511][moved_p][to as usize], bonus);
        if ply >= 1 {
            let pm = self.move_stack[ply - 1];
            let pp = self.piece_stack[ply - 1];
            if !pm.is_none() && pp < 6 {
                update_history_i16(&mut self.cont_hist[pp][pm.to() as usize][moved_p][to as usize], bonus);
            }
        }
        if ply >= 2 {
            let pm2 = self.move_stack[ply - 2];
            let pp2 = self.piece_stack[ply - 2];
            if !pm2.is_none() && pp2 < 6 {
                update_history_i16(&mut self.follow_hist[pp2][pm2.to() as usize][moved_p][to as usize], bonus);
            }
        }
        if ply >= 4 {
            let pm4 = self.move_stack[ply - 4];
            let pp4 = self.piece_stack[ply - 4];
            if !pm4.is_none() && pp4 < 6 {
                update_history_i16(&mut self.follow4_hist[pp4][pm4.to() as usize][moved_p][to as usize], bonus);
            }
        }
        if ply >= 6 {
            let pm6 = self.move_stack[ply - 6];
            let pp6 = self.piece_stack[ply - 6];
            if !pm6.is_none() && pp6 < 6 {
                update_history_i16(&mut self.follow6_hist[pp6][pm6.to() as usize][moved_p][to as usize], bonus);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        b: &Board,
        mut depth: i32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        allow_null: bool,
        cutnode: bool,
    ) -> i32 {
        self.pv_len[ply] = 0;

        if ply > 0 {
            if b.halfmove >= 100 || b.is_insufficient_material() || self.is_repetition(b.hash, b.halfmove) {
                return 0;
            }
            let r_alpha = alpha.max(-MATE_SCORE + ply as i32);
            let r_beta = beta.min(MATE_SCORE - ply as i32 - 1);
            if r_alpha >= r_beta {
                return r_alpha;
            }
        }

        let in_check = b.in_check();
        if in_check && depth < (MAX_PLY as i32 - ply as i32 - 2) {
            depth += 1;
        }

        if depth <= 0 || ply >= MAX_PLY - 2 {
            return self.qsearch(b, ply, alpha, beta);
        }

        self.nodes += 1;
        if self.check_limits() {
            return 0;
        }

        let is_pv = beta - alpha > 1;
        let orig_alpha = alpha;
        let excl = self.excluded[ply];

        let tt_entry = if excl.is_none() { self.tt.probe(b.hash) } else { None };
        let mut tt_move = Move::NONE;
        let mut tt_pv = is_pv;
        let mut tt_static = NO_EVAL;
        if let Some(e) = tt_entry {
            tt_move = e.best_move;
            tt_pv |= e.tt_pv();
            tt_static = e.static_eval;
            if !is_pv && e.depth as i32 >= depth {
                let s = score_from_tt(e.score as i32, ply);
                match e.bound() {
                    Bound::Exact => return s,
                    Bound::Lower if s >= beta => return s,
                    Bound::Upper if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let (raw_eval, cached_raw, used_small, corrplexity) = if in_check {
            (-INF, NO_EVAL, false, 0)
        } else if !excl.is_none() && self.static_evals[ply] != -INF {
            (self.static_evals[ply], NO_EVAL, false, 0)
        } else {
            self.corrected_eval(b, ply, tt_static)
        };
        let mut eval = raw_eval;
        if !in_check {
            if let Some(e) = tt_entry {
                let ts = score_from_tt(e.score as i32, ply);
                if ts.abs() < MATE_BOUND
                    && (e.bound() == Bound::Exact
                        || (e.bound() == Bound::Lower && ts > eval)
                        || (e.bound() == Bound::Upper && ts < eval))
                {
                    eval = ts;
                }
            }
        }
        self.static_evals[ply] = raw_eval;

        // Hindsight eval-gain history update on opponent's prior quiet move (exact threat bucket)
        if !in_check
            && ply >= 1
            && excl.is_none()
            && !self.move_stack[ply - 1].is_none()
            && !self.move_stack[ply - 1].is_tactical()
            && self.static_evals[ply - 1] != -INF
        {
            let gain = -(raw_eval + self.static_evals[ply - 1]);
            if gain.abs() >= 15 {
                let pm = self.move_stack[ply - 1];
                let opp_c = b.side.idx() ^ 1;
                let th = (self.threat_stack[ply - 1] as usize).min(3);
                let nudge = (gain * 2).clamp(-150, 150);
                let slot = &mut self.history[opp_c][pm.from() as usize][pm.to() as usize][th];
                update_history_i16(slot, nudge);
            }
        }

        // Fixed `improving` flag: fall back to ply-4 when ply-2 was in check (static_eval == -INF)!
        let improving = if in_check {
            false
        } else if ply >= 2 && self.static_evals[ply - 2] != -INF {
            raw_eval > self.static_evals[ply - 2]
        } else if ply >= 4 && self.static_evals[ply - 4] != -INF {
            raw_eval > self.static_evals[ply - 4]
        } else {
            true
        };
        let opponent_worsening =
            !in_check && ply >= 1 && self.static_evals[ply - 1] != -INF && raw_eval + self.static_evals[ply - 1] > 1;
        let mut mate_threat = false;

        // Non-PV whole-node pruning
        if !is_pv && !in_check && excl.is_none() {
            // 1. Reverse Futility Pruning (fail-soft, tt_pv & opponent_worsening aware)
            let rfp_margin = self.params.rfp_mult * depth
                - self.params.rfp_improving * i32::from(improving)
                - 18 * i32::from(!improving && opponent_worsening)
                + self.params.rfp_ttpv * i32::from(tt_pv);
            if depth <= 8 && eval - rfp_margin >= beta && beta.abs() < MATE_BOUND {
                return (eval + beta) / 2;
            }

            // 2. Razoring
            if depth <= 3 && eval + self.params.razor_mult * depth < alpha {
                let q = self.qsearch(b, ply, alpha, beta);
                if q <= alpha {
                    return q;
                }
            }

            // 3. Null Move Pruning (+ verification search at high depth to prevent zugzwang blunders)
            if allow_null && depth >= 3 && eval >= beta && b.has_non_pawn_material(b.side) {
                let r = (self.params.nmp_base + depth * 10000 / self.params.nmp_depth_div) / 100
                    + ((eval - beta) / self.params.nmp_eval_div.max(50)).clamp(0, 3);
                let tt_forbids_nmp = matches!(
                    tt_entry,
                    Some(e) if e.bound() == Bound::Upper
                        && e.depth as i32 >= depth - r
                        && score_from_tt(e.score as i32, ply) < beta
                );
                if !tt_forbids_nmp {
                    let next = b.make_null_move();
                    self.tt.prefetch(next.hash);
                    self.push_acc_null(ply, &next);
                    self.move_stack[ply] = Move::NONE;
                    self.piece_stack[ply] = 6;
                    self.rep_history.push(next.hash);
                    let mut score = -self.negamax(&next, depth - r, ply + 1, -beta, -beta + 1, false, !cutnode);
                    self.rep_history.pop();
                    if self.stopped {
                        return 0;
                    }
                    if score >= beta {
                        if score >= MATE_BOUND {
                            score = beta;
                        }
                        if depth < 12 {
                            return score;
                        }
                        // Verification search at depth >= 12 with null move disabled
                        let v = self.negamax(b, depth - r, ply, beta - 1, beta, false, cutnode);
                        if self.stopped {
                            return 0;
                        }
                        if v >= beta {
                            return score;
                        }
                    }
                    if score <= -MATE_BOUND {
                        mate_threat = true;
                    }
                }
            }

            // 4. ProbCut
            if depth >= 5 && beta.abs() < MATE_BOUND {
                let prob_beta = (beta + self.params.probcut_margin).min(MATE_BOUND - 1);
                let prob_depth = depth - 4;
                if let Some(e) = tt_entry {
                    if e.depth as i32 >= prob_depth
                        && matches!(e.bound(), Bound::Lower | Bound::Exact)
                        && score_from_tt(e.score as i32, ply) >= prob_beta
                    {
                        return score_from_tt(e.score as i32, ply);
                    }
                }
                let mut tml = MoveList::new();
                b.tactical_moves(&mut tml);
                let see_thresh = (prob_beta - raw_eval).max(0);
                // Order ProbCut tacticals by MVV-LVA + capture history so cutoffs trigger on the first capture
                let mut pc_scores = [0i32; MAX_MOVES];
                let pc_c = b.side.idx();
                for (i, &m) in tml.as_slice().iter().enumerate() {
                    let ch = if m.is_capture() {
                        let att_p = mb_piece(b.mailbox[m.from() as usize]).idx();
                        self.cap_hist[pc_c * 6 + att_p][m.to() as usize][captured_piece_idx(b, m)] / 4
                    } else {
                        0
                    };
                    pc_scores[i] = score_capture(b, m) + ch;
                }
                for idx in 0..tml.len() {
                    let mut best_i = idx;
                    let mut best_s = pc_scores[idx];
                    for j in (idx + 1)..tml.len() {
                        if pc_scores[j] > best_s {
                            best_s = pc_scores[j];
                            best_i = j;
                        }
                    }
                    if best_i != idx {
                        tml.swap(idx, best_i);
                        pc_scores.swap(idx, best_i);
                    }
                    let m = tml[idx];
                    if !see_ge(b, m, see_thresh) {
                        continue;
                    }
                    let next = b.make_move(m);
                    self.tt.prefetch(next.hash);
                    self.push_acc_move(ply, b, &next, m);
                    let moved_p = mb_piece(b.mailbox[m.from() as usize]).idx();
                    self.move_stack[ply] = m;
                    self.piece_stack[ply] = moved_p;
                    self.rep_history.push(next.hash);

                    let mut sc = -self.qsearch(&next, ply + 1, -prob_beta, -prob_beta + 1);
                    if sc >= prob_beta && !self.stopped {
                        sc = -self.negamax(&next, prob_depth, ply + 1, -prob_beta, -prob_beta + 1, true, !cutnode);
                    }
                    self.rep_history.pop();
                    if self.stopped {
                        return 0;
                    }
                    if sc >= prob_beta {
                        self.tt.store(b.hash, prob_depth + 1, sc, Bound::Lower, m, ply, tt_pv, cached_raw);
                        return sc;
                    }
                }
            }
        }

        // Internal Iterative Reduction (IIR)
        if depth >= 3 && tt_move.is_none() && excl.is_none() && (is_pv || cutnode) {
            depth -= 1;
        }

        let mut best_move = Move::NONE;
        let mut best_score = -INF;
        let mut quiet_tried = [(Move::NONE, 0usize); 64];
        let mut quiet_cnt = 0usize;
        let mut cap_tried = [(Move::NONE, 0usize, 0usize); 32];
        let mut cap_cnt = 0usize;
        let mut searched = 0usize;
        let mut skipped_tt = Move::NONE;
        let mut threat_map: Option<u64> = None;

        // Stage 0: Staged TT-first move execution (avoids any movegen on ~40% of cut-nodes!)
        if ply > 0 && excl.is_none() && !tt_move.is_none() && b.is_legal_tt_move(tt_move) {
            let m = tt_move;
            skipped_tt = m;
            let is_quiet = !m.is_tactical();
            let moved_p = mb_piece(b.mailbox[m.from() as usize]).idx();

            let mut ext = 0i32;
            if depth >= 7 {
                if let Some(e) = tt_entry {
                    let tt_s = score_from_tt(e.score as i32, ply);
                    if e.depth as i32 >= depth - 3
                        && matches!(e.bound(), Bound::Lower | Bound::Exact)
                        && tt_s.abs() < MATE_BOUND
                    {
                        let s_beta = (tt_s - depth * self.params.se_depth_mult / 10).max(-MATE_BOUND + 1);
                        let s_depth = (depth - 1) / 2;
                        self.excluded[ply] = m;
                        let s_score = self.negamax(b, s_depth, ply, s_beta - 1, s_beta, false, cutnode);
                        self.excluded[ply] = Move::NONE;
                        if self.stopped {
                            return 0;
                        }
                        if s_score < s_beta {
                            ext = 1;
                            if !is_pv && s_score + self.params.se_double_margin < s_beta && ply < MAX_PLY / 2 {
                                ext = 2;
                            }
                        } else if s_beta >= beta {
                            return s_beta;
                        } else if tt_s >= beta {
                            ext = -2 + i32::from(is_pv);
                        } else if cutnode && !is_pv {
                            ext = -2;
                        } else if tt_s <= alpha {
                            ext = -1;
                        }
                    }
                }
            }

            let next = b.make_move(m);
            self.tt.prefetch(next.hash);
            self.push_acc_move(ply, b, &next, m);
            self.move_stack[ply] = m;
            self.piece_stack[ply] = moved_p;
            self.threat_stack[ply] = 0;
            self.rep_history.push(next.hash);

            let new_depth = (depth - 1 + ext).clamp(0, (MAX_PLY - ply - 2) as i32);
            let first_cut = if is_pv { false } else { !cutnode };
            let score = -self.negamax(&next, new_depth, ply + 1, -beta, -alpha, true, first_cut);
            self.rep_history.pop();
            searched = 1;
            if self.stopped {
                return 0;
            }

            best_score = score;
            best_move = m;
            if score > alpha {
                alpha = score;
                self.pv_table[ply][0] = m;
                let child_len = self.pv_len[ply + 1].min(MAX_PLY - ply - 2);
                for k in 0..child_len {
                    self.pv_table[ply][k + 1] = self.pv_table[ply + 1][k];
                }
                self.pv_len[ply] = child_len + 1;

                if alpha >= beta {
                    let bonus_d = depth + i32::from(eval <= orig_alpha);
                    let bonus = (self.params.hist_bonus_mult * bonus_d - 220).clamp(0, 2100);
                    let c = b.side.idx();
                    if is_quiet {
                        if self.killers[ply][0] != m {
                            self.killers[ply][1] = self.killers[ply][0];
                            self.killers[ply][0] = m;
                        }
                        if ply >= 1 && !self.move_stack[ply - 1].is_none() {
                            let pm = self.move_stack[ply - 1];
                            self.countermoves[pm.from() as usize][pm.to() as usize] = m;
                        }
                        let tm = *threat_map.get_or_insert_with(|| b.enemy_attacks());
                        self.update_quiet_histories(c, b.pawn_hash, ply, m, moved_p, tm, bonus);
                    } else if m.is_capture() {
                        let vic = captured_piece_idx(b, m);
                        update_history(&mut self.cap_hist[c * 6 + moved_p][m.to() as usize][vic], bonus);
                    }
                    if !in_check && !used_small && best_score.abs() < MATE_BOUND && is_quiet && best_score > raw_eval {
                        self.update_corrhist(b, ply, best_score - raw_eval, depth);
                    }
                    self.tt.store(b.hash, depth, best_score, Bound::Lower, best_move, ply, tt_pv, cached_raw);
                    return best_score;
                }
            }
            if is_quiet {
                quiet_tried[0] = (m, moved_p);
                quiet_cnt = 1;
            } else if m.is_capture() {
                cap_tried[0] = (m, moved_p, captured_piece_idx(b, m));
                cap_cnt = 1;
            }
        }

        // Full Staged MovePicker (only reached if TT move missed or didn't cut off)
        let k0 = self.killers[ply][0];
        let k1 = self.killers[ply][1];
        let cm = if ply >= 1 && !self.move_stack[ply - 1].is_none() {
            let pm = self.move_stack[ply - 1];
            self.countermoves[pm.from() as usize][pm.to() as usize]
        } else {
            Move::NONE
        };
        let tt_capture = !tt_move.is_none() && tt_move.is_capture();
        let mut picker = MovePicker::new(in_check, tt_move, skipped_tt, excl, k0, k1, cm);
        let mut chk_info: Option<([u64; 5], u64)> = None;

        while let Some((m, is_good_cap, cached_qh)) = picker.next(self, b, ply, &mut threat_map) {
            let is_quiet = !m.is_tactical();
            let moved_p = mb_piece(b.mailbox[m.from() as usize]).idx();

            // Lazy check test: only compute check_squares once we actually need pruning/LMR!
            let need_prune = !is_pv && !in_check && !mate_threat && best_score > -MATE_BOUND && searched > 0;
            let need_lmr = depth >= 3 && searched >= if ply == 0 { 2 } else { 1 } && !in_check;
            let gives_check = if need_prune || need_lmr {
                let (csq, dc) = *chk_info.get_or_insert_with(|| (b.check_squares(), b.discovered_check_candidates()));
                b.gives_check_fast(m, &csq, dc)
            } else {
                false
            };

            let new_depth = (depth - 1).clamp(0, (MAX_PLY - ply - 2) as i32);

            // Compute quiet history ONCE (reusing MovePicker's precomputed 6-table lookup!)
            let qh = if is_quiet && (need_prune || need_lmr) {
                if cached_qh != i32::MIN {
                    cached_qh
                } else {
                    let tm = *threat_map.get_or_insert_with(|| b.enemy_attacks());
                    self.quiet_history(b, ply, m, moved_p, tm)
                }
            } else {
                0
            };

            // Precompute LMR reduction `r` BEFORE pruning so Futility & History pruning use `lmr_depth`!
            let mut r = 0;
            if searched > 0 && need_lmr {
                if is_quiet {
                    let mut r_fp = self.lmr[(depth as usize).min(63)][searched.min(63)];
                    if !is_pv {
                        r_fp += 512;
                    }
                    if !tt_pv {
                        r_fp += self.params.lmr_ttpv_w;
                    }
                    if cutnode {
                        r_fp += self.params.lmr_cut_w;
                    }
                    if !improving {
                        r_fp += if opponent_worsening { self.params.lmr_imp_w / 2 } else { self.params.lmr_imp_w };
                    }
                    if tt_capture {
                        r_fp += self.params.lmr_ttcap_w;
                    }
                    if m == k0 || m == k1 || m == cm {
                        r_fp -= 512;
                    }
                    if gives_check || mate_threat {
                        r_fp -= 900;
                    }
                    if corrplexity > 150 {
                        r_fp -= self.params.lmr_corr_w;
                    }
                    r_fp -= qh * 1024 / self.params.lmr_hist_div.max(1000);
                    r = ((r_fp + 512) >> 10).clamp(0, new_depth - 1).max(0);
                } else if !gives_check && ((!is_good_cap && searched >= 2) || (!is_pv && searched >= 4)) {
                    let ch = if m.is_capture() {
                        self.cap_hist[b.side.idx() * 6 + moved_p][m.to() as usize][captured_piece_idx(b, m)]
                    } else {
                        0
                    };
                    let mut r_fp = self.lmr[(depth as usize).min(63)][searched.min(63)] / 2 - ch * 1024 / 8000;
                    if is_good_cap {
                        r_fp /= 2;
                    }
                    if cutnode {
                        r_fp += 384;
                    }
                    if !improving {
                        r_fp += 384;
                    }
                    r = ((r_fp + 512) >> 10).clamp(0, new_depth - 1).max(0);
                }
            }

            if need_prune && !gives_check {
                if is_quiet {
                    let lmr_depth = (new_depth - r).max(0);
                    let lmp_limit = (self.params.lmp_base / 100 + depth * depth) / (2 - i32::from(improving));
                    if depth <= 5 && searched as i32 >= lmp_limit {
                        continue;
                    }
                    if lmr_depth <= 7
                        && eval + self.params.futility_mult * lmr_depth + self.params.futility_base + qh / 32 <= alpha
                    {
                        continue;
                    }
                    if lmr_depth <= 4 && qh < -self.params.hist_prune_mult * lmr_depth.max(1) {
                        continue;
                    }
                    if depth <= 6 && !see_ge(b, m, -self.params.see_quiet_prune * depth * depth) {
                        continue;
                    }
                } else if depth <= 5 && !is_good_cap && !see_ge(b, m, -self.params.see_cap_prune * depth) {
                    continue;
                }
            }

            let next = b.make_move(m);
            self.tt.prefetch(next.hash);
            self.push_acc_move(ply, b, &next, m);
            self.move_stack[ply] = m;
            self.piece_stack[ply] = moved_p;
            self.threat_stack[ply] = if let Some(tm) = threat_map { threat_idx(m.from(), m.to(), tm) as u8 } else { 0 };
            self.rep_history.push(next.hash);

            let nodes_before = self.nodes;
            let mut score;
            if searched == 0 {
                let first_cut = if is_pv { false } else { !cutnode };
                score = -self.negamax(&next, new_depth, ply + 1, -beta, -alpha, true, first_cut);
            } else {
                score = -self.negamax(&next, new_depth - r, ply + 1, -alpha - 1, -alpha, true, true);
                if r > 0 && score > alpha {
                    let do_deeper = score > best_score + self.params.lmr_deeper_margin + 2 * new_depth;
                    let do_shallower = score < best_score + new_depth;
                    let re_depth = (new_depth + i32::from(do_deeper) - i32::from(do_shallower))
                        .clamp(0, (MAX_PLY - ply - 2) as i32);
                    score = -self.negamax(&next, re_depth, ply + 1, -alpha - 1, -alpha, true, !cutnode);
                    // LMR re-search continuation history feedback (Stockfish-style)
                    if is_quiet && !self.stopped && (score <= alpha || score >= beta) {
                        let bonus = if score >= beta {
                            (self.params.hist_bonus_mult * depth - 220).clamp(0, 2100) / 2
                        } else {
                            -((self.params.hist_malus_mult * depth - 180).clamp(0, 1800) / 2)
                        };
                        let tm = *threat_map.get_or_insert_with(|| b.enemy_attacks());
                        self.update_quiet_histories(b.side.idx(), b.pawn_hash, ply, m, moved_p, tm, bonus);
                    }
                }
                if is_pv && score > alpha && score < beta {
                    score = -self.negamax(&next, new_depth, ply + 1, -beta, -alpha, true, false);
                }
            }

            if ply == 0 {
                self.root_nodes[m.from() as usize][m.to() as usize] += self.nodes - nodes_before;
            }
            self.rep_history.pop();
            searched += 1;
            if self.stopped {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    self.pv_table[ply][0] = m;
                    let child_len = self.pv_len[ply + 1].min(MAX_PLY - ply - 2);
                    for k in 0..child_len {
                        self.pv_table[ply][k + 1] = self.pv_table[ply + 1][k];
                    }
                    self.pv_len[ply] = child_len + 1;

                    if alpha >= beta {
                        let bonus_d = depth + i32::from(eval <= orig_alpha);
                        let bonus = (self.params.hist_bonus_mult * bonus_d - 220).clamp(0, 2100);
                        let malus = -((self.params.hist_malus_mult * depth - 180).clamp(0, 1800));
                        let c = b.side.idx();
                        if is_quiet {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            if ply >= 1 && !self.move_stack[ply - 1].is_none() {
                                let pm = self.move_stack[ply - 1];
                                self.countermoves[pm.from() as usize][pm.to() as usize] = m;
                            }
                            let tm = *threat_map.get_or_insert_with(|| b.enemy_attacks());
                            self.update_quiet_histories(c, b.pawn_hash, ply, m, moved_p, tm, bonus);
                            for &(qm, qp) in &quiet_tried[..quiet_cnt] {
                                self.update_quiet_histories(c, b.pawn_hash, ply, qm, qp, tm, malus);
                            }
                            for &(cm, cp, cv) in &cap_tried[..cap_cnt] {
                                update_history(&mut self.cap_hist[c * 6 + cp][cm.to() as usize][cv], malus);
                            }
                        } else if m.is_capture() {
                            let vic = captured_piece_idx(b, m);
                            update_history(&mut self.cap_hist[c * 6 + moved_p][m.to() as usize][vic], bonus);
                            for &(cm, cp, cv) in &cap_tried[..cap_cnt] {
                                update_history(&mut self.cap_hist[c * 6 + cp][cm.to() as usize][cv], malus);
                            }
                            if quiet_cnt > 0 {
                                let tm = *threat_map.get_or_insert_with(|| b.enemy_attacks());
                                for &(qm, qp) in &quiet_tried[..quiet_cnt] {
                                    self.update_quiet_histories(c, b.pawn_hash, ply, qm, qp, tm, malus);
                                }
                            }
                        }
                        break;
                    }
                }
            }

            if is_quiet {
                if quiet_cnt < quiet_tried.len() {
                    quiet_tried[quiet_cnt] = (m, moved_p);
                    quiet_cnt += 1;
                }
            } else if m.is_capture() && cap_cnt < cap_tried.len() {
                cap_tried[cap_cnt] = (m, moved_p, captured_piece_idx(b, m));
                cap_cnt += 1;
            }
        }

        if searched == 0 {
            return if !excl.is_none() {
                -INF
            } else if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            };
        }

        if excl.is_none() {
            let bound = if best_score >= beta {
                Bound::Lower
            } else if best_score > orig_alpha {
                Bound::Exact
            } else {
                Bound::Upper
            };

            if !in_check
                && !used_small
                && best_score.abs() < MATE_BOUND
                && (best_move.is_none() || !best_move.is_tactical())
                && (bound == Bound::Exact
                    || (bound == Bound::Lower && best_score > raw_eval)
                    || (bound == Bound::Upper && best_score < raw_eval))
            {
                self.update_corrhist(b, ply, best_score - raw_eval, depth);
            }

            self.tt.store(b.hash, depth, best_score, bound, best_move, ply, tt_pv, cached_raw);
        }
        best_score
    }

    fn qsearch(&mut self, b: &Board, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if self.check_limits() {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.corrected_eval(b, ply, NO_EVAL).0;
        }

        let in_check = b.in_check();
        let orig_alpha = alpha;

        let tt_entry = self.tt.probe(b.hash);
        let mut tt_move = Move::NONE;
        let mut tt_pv = false;
        let mut tt_static = NO_EVAL;
        if let Some(e) = tt_entry {
            tt_move = e.best_move;
            tt_pv = e.tt_pv();
            tt_static = e.static_eval;
            let s = score_from_tt(e.score as i32, ply);
            match e.bound() {
                Bound::Exact => return s,
                Bound::Lower if s >= beta => return s,
                Bound::Upper if s <= alpha => return s,
                _ => {}
            }
        }

        let mut best_score = -INF;
        let mut cached_raw = NO_EVAL;
        if !in_check {
            let (raw_sp, craw, _, _) = self.corrected_eval(b, ply, tt_static);
            cached_raw = craw;
            let mut stand_pat = raw_sp;
            if let Some(e) = tt_entry {
                let ts = score_from_tt(e.score as i32, ply);
                if ts.abs() < MATE_BOUND
                    && (e.bound() == Bound::Exact
                        || (e.bound() == Bound::Lower && ts > stand_pat)
                        || (e.bound() == Bound::Upper && ts < stand_pat))
                {
                    stand_pat = ts;
                }
            }
            if stand_pat >= beta {
                return stand_pat;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
            best_score = stand_pat;
        }

        let mut ml = MoveList::new();
        b.tactical_moves(&mut ml);
        if ml.is_empty() {
            return if in_check { -MATE_SCORE + ply as i32 } else { best_score };
        }

        let mut scores = [0i32; MAX_MOVES];
        let c = b.side.idx();
        for i in 0..ml.len() {
            let m = ml[i];
            if m == tt_move {
                scores[i] = 2_000_000;
            } else {
                let mvv = score_capture(b, m);
                let ch = if m.is_capture() {
                    let att_p = mb_piece(b.mailbox[m.from() as usize]).idx();
                    self.cap_hist[c * 6 + att_p][m.to() as usize][captured_piece_idx(b, m)] / 4
                } else {
                    0
                };
                scores[i] = mvv + ch;
            }
        }

        let mut best_move = Move::NONE;
        let mut qs_searched = 0usize;
        for idx in 0..ml.len() {
            let mut best_i = idx;
            let mut best_s = scores[idx];
            for j in (idx + 1)..ml.len() {
                if scores[j] > best_s {
                    best_s = scores[j];
                    best_i = j;
                }
            }
            if best_i != idx {
                ml.swap(idx, best_i);
                scores.swap(idx, best_i);
            }

            let m = ml[idx];
            if !in_check {
                let gain = capture_gain(b, m);
                if !m.is_promo() && best_score + gain + self.params.qs_delta_margin < alpha {
                    continue;
                }
                if !see_ge(b, m, 0) {
                    continue;
                }
            } else if best_score > -MATE_BOUND && qs_searched >= 2 && !m.is_tactical() && !see_ge(b, m, 0) {
                continue;
            }
            qs_searched += 1;

            let next = b.make_move(m);
            self.tt.prefetch(next.hash);
            self.push_acc_move(ply, b, &next, m);
            let score = -self.qsearch(&next, ply + 1, -beta, -alpha);
            if self.stopped {
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > orig_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(b.hash, 0, best_score, bound, best_move, ply, tt_pv, cached_raw);
        best_score
    }
}

#[inline(always)]
fn threat_idx(from: u8, to: u8, threat_map: u64) -> usize {
    let f = usize::from((threat_map & bb(from)) != 0);
    let t = usize::from((threat_map & bb(to)) != 0);
    (f << 1) | t
}

#[inline(always)]
fn update_corr_slot(slot: &mut i32, scaled_diff: i32, weight: i32) {
    let new_val = (*slot * (256 - weight) + scaled_diff * weight) / 256;
    *slot = new_val.clamp(-CORR_MAX * CORR_GRAIN, CORR_MAX * CORR_GRAIN);
}

#[inline(always)]
fn captured_piece_idx(b: &Board, m: Move) -> usize {
    if m.is_ep() {
        Piece::Pawn.idx()
    } else {
        mb_piece(b.mailbox[m.to() as usize]).idx()
    }
}

#[inline(always)]
fn update_history(slot: &mut i32, bonus: i32) {
    const MAX_HIST: i32 = 16_384;
    let clamped = bonus.clamp(-MAX_HIST, MAX_HIST);
    *slot += clamped - (*slot * clamped.abs()) / MAX_HIST;
}

#[inline(always)]
fn update_history_i16(slot: &mut i16, bonus: i32) {
    const MAX_HIST: i32 = 16_384;
    let cur = *slot as i32;
    let clamped = bonus.clamp(-MAX_HIST, MAX_HIST);
    *slot = (cur + clamped - (cur * clamped.abs()) / MAX_HIST).clamp(-16_384, 16_384) as i16;
}

#[inline(always)]
fn score_capture(b: &Board, m: Move) -> i32 {
    let victim = if m.is_ep() {
        SEE_VAL[Piece::Pawn.idx()]
    } else if m.is_capture() {
        SEE_VAL[mb_piece(b.mailbox[m.to() as usize]).idx()]
    } else {
        0
    };
    let attacker = SEE_VAL[mb_piece(b.mailbox[m.from() as usize]).idx()];
    let promo = m.promo_piece().map(|p| SEE_VAL[p.idx()]).unwrap_or(0);
    victim * 10 - attacker + promo * 8
}

#[inline(always)]
fn capture_gain(b: &Board, m: Move) -> i32 {
    let mut g = if m.is_ep() {
        SEE_VAL[Piece::Pawn.idx()]
    } else if m.is_capture() {
        SEE_VAL[mb_piece(b.mailbox[m.to() as usize]).idx()]
    } else {
        0
    };
    if let Some(p) = m.promo_piece() {
        g += SEE_VAL[p.idx()] - SEE_VAL[Piece::Pawn.idx()];
    }
    g
}
