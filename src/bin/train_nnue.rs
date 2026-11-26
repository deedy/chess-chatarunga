//! Ultra-fast multi-threaded sparse AdamW NNUE trainer in pure Rust (with factorized king buckets).
//!
//! Architecture: `(768 * KING_BUCKETS -> HL) x 2 -> OUTPUT_BUCKETS` with SCReLU activation (`clamp(x, 0, 1)^2`).
//! When `king_buckets > 1`, uses virtual factorized base features (`768 -> HL`) during training and folds them
//! into every king bucket on export — faster convergence & stronger generalization with zero inference cost!
//!
//! Example:
//!   train_nnue --data data/gen0.bin --out nets/gen1.nnue --hl 256 --kb 1 --ob 8 --epochs 25 --threads 16

use chatarunga::data::PackedPos;
use chatarunga::nnue::{
    feature_index, NnueNet, DEFAULT_QA, DEFAULT_QB, DEFAULT_SCALE, VERSION_SCRELU,
};
use chatarunga::{Color, Piece, Rng};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const MAX_WEIGHT: f32 = 1.98;

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn output_bucket_from_occ(occ: u64, ob: usize) -> usize {
    if ob <= 1 {
        return 0;
    }
    let cnt = occ.count_ones() as usize;
    let div = 32usize.div_ceil(ob);
    (cnt.saturating_sub(2) / div).min(ob - 1)
}

fn load_dataset(paths: &[String]) -> Vec<PackedPos> {
    let mut all = Vec::new();
    for p in paths {
        let mut f = File::open(p).unwrap_or_else(|e| panic!("open {p}: {e}"));
        let len = f.metadata().unwrap().len() as usize;
        let n = len / 32;
        let mut raw = vec![0u8; n * 32];
        f.read_exact(&mut raw).expect("read dataset");
        all.reserve(n);
        for chunk in raw.chunks_exact(32) {
            let arr: &[u8; 32] = chunk.try_into().unwrap();
            all.push(PackedPos::from_bytes(arr));
        }
        println!("Loaded {n} positions from {p}");
    }
    all
}

#[derive(Clone)]
struct Params {
    kb: usize,
    hl: usize,
    ob: usize,
    factorized: bool,
    /// Total feature rows = `kb * 768 + (if factorized { 768 } else { 0 })`.
    n_feats: usize,
    ft_w: Vec<f32>,
    ft_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl Params {
    fn new(kb: usize, hl: usize, ob: usize, seed: u64) -> Self {
        let factorized = kb > 1;
        let n_feats = kb * 768 + if factorized { 768 } else { 0 };
        let mut rng = Rng::new(seed);
        let ft_scale = (1.0 / (32.0f64).sqrt()) as f32 * 0.25;
        let out_scale = (1.0 / (2.0 * hl as f64).sqrt()) as f32;
        let mut ft_w = vec![0.0f32; n_feats * hl];
        if factorized {
            // Initialize base features with small random weights, bucket offsets near zero.
            let base_off = kb * 768 * hl;
            for v in &mut ft_w[base_off..] {
                *v = (rng.next_f64() as f32 * 2.0 - 1.0) * ft_scale;
            }
        } else {
            for v in &mut ft_w {
                *v = (rng.next_f64() as f32 * 2.0 - 1.0) * ft_scale;
            }
        }
        let ft_b = vec![0.0f32; hl];
        let out_w: Vec<f32> = (0..ob * 2 * hl)
            .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * out_scale)
            .collect();
        let out_b = vec![0.0f32; ob];
        Self {
            kb,
            hl,
            ob,
            factorized,
            n_feats,
            ft_w,
            ft_b,
            out_w,
            out_b,
        }
    }

    fn from_nnue(net: &NnueNet) -> Self {
        let kb = net.king_buckets;
        let hl = net.hl;
        let ob = net.output_buckets;
        let factorized = kb > 1;
        let n_feats = kb * 768 + if factorized { 768 } else { 0 };
        let qa = net.qa as f32;
        let qb = net.qb as f32;
        let mut ft_w = vec![0.0f32; n_feats * hl];
        for (i, &v) in net.ft_w.iter().enumerate() {
            ft_w[i] = v as f32 / qa;
        }
        let ft_b: Vec<f32> = net.ft_b.iter().map(|&v| v as f32 / qa).collect();
        let out_w: Vec<f32> = net.out_w.iter().map(|&v| v as f32 / qb).collect();
        let out_b: Vec<f32> = net.out_b.iter().map(|&v| v as f32 / (qa * qb)).collect();
        Self {
            kb,
            hl,
            ob,
            factorized,
            n_feats,
            ft_w,
            ft_b,
            out_w,
            out_b,
        }
    }

    fn quantize(&self, scale: i32) -> NnueNet {
        let qa = DEFAULT_QA as f32;
        let qb = DEFAULT_QB as f32;
        let mut ft_w = vec![0i16; self.kb * 768 * self.hl];
        let base_start = self.kb * 768 * self.hl;
        for b in 0..self.kb {
            for f in 0..768 {
                let row_off = (b * 768 + f) * self.hl;
                let base_off = base_start + f * self.hl;
                for h in 0..self.hl {
                    let mut val = self.ft_w[row_off + h];
                    if self.factorized {
                        val += self.ft_w[base_off + h];
                    }
                    ft_w[row_off + h] =
                        (val * qa).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                }
            }
        }
        let ft_b: Vec<i16> = self
            .ft_b
            .iter()
            .map(|&v| (v * qa).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect();
        let out_w: Vec<i16> = self
            .out_w
            .iter()
            .map(|&v| {
                (v.clamp(-MAX_WEIGHT, MAX_WEIGHT) * qb)
                    .round()
                    .clamp(-127.0, 127.0) as i16
            })
            .collect();
        let out_b: Vec<i32> = self
            .out_b
            .iter()
            .map(|&v| (v * qa * qb).round() as i32)
            .collect();
        NnueNet {
            version: VERSION_SCRELU,
            king_buckets: self.kb,
            hl: self.hl,
            output_buckets: self.ob,
            qa: DEFAULT_QA,
            qb: DEFAULT_QB,
            scale,
            has_threats: false,
            l1_size: 0,
            l2_size: 0,
            ft_w,
            ft_b,
            out_w,
            out_b,
            l1_w: Vec::new(),
            l1_w_t: Vec::new(),
            l1_b: Vec::new(),
            l2_w: Vec::new(),
            l2_b: Vec::new(),
            out_w_i8: Vec::new(),
        }
    }
}

struct GradBuf {
    ft_w: Vec<f32>,
    ft_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    loss_sum: f64,
}

impl GradBuf {
    fn new(n_feats: usize, hl: usize, ob: usize) -> Self {
        Self {
            ft_w: vec![0.0; n_feats * hl],
            ft_b: vec![0.0; hl],
            out_w: vec![0.0; ob * 2 * hl],
            out_b: vec![0.0; ob],
            loss_sum: 0.0,
        }
    }
    fn clear(&mut self) {
        self.ft_w.fill(0.0);
        self.ft_b.fill(0.0);
        self.out_w.fill(0.0);
        self.out_b.fill(0.0);
        self.loss_sum = 0.0;
    }
}

/// Extract sparse active features for both perspectives (White, Black).
/// Returns counts `(n_w, n_b)` written into `w_feats` and `b_feats` (up to 64 when factorized).
#[inline]
fn extract_features(
    pos: &PackedPos,
    kb: usize,
    factorized: bool,
    w_feats: &mut [usize; 64],
    b_feats: &mut [usize; 64],
) -> (usize, usize) {
    let mut pcs = [(0u8, Color::White, Piece::Pawn); 32];
    let (cnt, kings) = pos.decode_pieces(&mut pcs);
    let base_offset = kb * 768;
    let mut nw = 0usize;
    let mut nb = 0usize;
    for &(sq, c, p) in &pcs[..cnt] {
        let fw = feature_index(kb, Color::White, kings[0], c, p, sq);
        w_feats[nw] = fw;
        nw += 1;
        let fb = feature_index(kb, Color::Black, kings[1], c, p, sq);
        b_feats[nb] = fb;
        nb += 1;
        if factorized {
            w_feats[nw] = base_offset + (fw % 768);
            nw += 1;
            b_feats[nb] = base_offset + (fb % 768);
            nb += 1;
        }
    }
    (nw, nb)
}

fn process_chunk(
    positions: &[PackedPos],
    indices: &[usize],
    p: &Params,
    wdl_weight: f32,
    scale_f: f32,
    grad: &mut GradBuf,
    compute_grad: bool,
) {
    let hl = p.hl;
    let mut w_acc = vec![0.0f32; hl];
    let mut b_acc = vec![0.0f32; hl];
    let mut d_stm = vec![0.0f32; hl];
    let mut d_opp = vec![0.0f32; hl];
    let mut w_feats = [0usize; 64];
    let mut b_feats = [0usize; 64];

    for &idx in indices {
        let pos = &positions[idx];
        let (nw, nb) = extract_features(pos, p.kb, p.factorized, &mut w_feats, &mut b_feats);

        // Forward: feature transformer
        w_acc.copy_from_slice(&p.ft_b);
        for &f in &w_feats[..nw] {
            let row = &p.ft_w[f * hl..(f + 1) * hl];
            for h in 0..hl {
                w_acc[h] += row[h];
            }
        }
        b_acc.copy_from_slice(&p.ft_b);
        for &f in &b_feats[..nb] {
            let row = &p.ft_w[f * hl..(f + 1) * hl];
            for h in 0..hl {
                b_acc[h] += row[h];
            }
        }

        let white_stm = pos.stm == 0;
        let (stm_acc, opp_acc) = if white_stm {
            (&w_acc, &b_acc)
        } else {
            (&b_acc, &w_acc)
        };
        let ob = output_bucket_from_occ(pos.occ, p.ob);
        let ow_base = ob * 2 * hl;
        let w_stm = &p.out_w[ow_base..ow_base + hl];
        let w_opp = &p.out_w[ow_base + hl..ow_base + 2 * hl];

        let mut out = p.out_b[ob];
        for h in 0..hl {
            let a = stm_acc[h].clamp(0.0, 1.0);
            let b = opp_acc[h].clamp(0.0, 1.0);
            out += (a * a) * w_stm[h] + (b * b) * w_opp[h];
        }

        // Target from STM perspective
        let stm_score = if white_stm {
            pos.score as f32
        } else {
            -(pos.score as f32)
        };
        let stm_wdl = if white_stm {
            pos.result as f32 * 0.5
        } else {
            1.0 - pos.result as f32 * 0.5
        };
        let target = wdl_weight * stm_wdl + (1.0 - wdl_weight) * sigmoid(stm_score / scale_f);
        let pred = sigmoid(out);
        let diff = pred - target;
        grad.loss_sum += (diff * diff) as f64;

        if !compute_grad {
            continue;
        }

        // Backward: dL/dout = 2 * (pred - target) * pred * (1 - pred)
        let d_out = 2.0 * diff * pred * (1.0 - pred);
        grad.out_b[ob] += d_out;
        let gw_stm = &mut grad.out_w[ow_base..ow_base + hl];
        for h in 0..hl {
            let a = stm_acc[h];
            let c = a.clamp(0.0, 1.0);
            gw_stm[h] += d_out * (c * c);
            d_stm[h] = if a > 0.0 && a < 1.0 {
                d_out * w_stm[h] * (2.0 * a)
            } else {
                0.0
            };
        }
        let gw_opp = &mut grad.out_w[ow_base + hl..ow_base + 2 * hl];
        for h in 0..hl {
            let a = opp_acc[h];
            let c = a.clamp(0.0, 1.0);
            gw_opp[h] += d_out * (c * c);
            d_opp[h] = if a > 0.0 && a < 1.0 {
                d_out * w_opp[h] * (2.0 * a)
            } else {
                0.0
            };
        }

        let (d_w, d_b) = if white_stm {
            (&d_stm, &d_opp)
        } else {
            (&d_opp, &d_stm)
        };
        for h in 0..hl {
            grad.ft_b[h] += d_w[h] + d_b[h];
        }
        for &f in &w_feats[..nw] {
            let row = &mut grad.ft_w[f * hl..(f + 1) * hl];
            for h in 0..hl {
                row[h] += d_w[h];
            }
        }
        for &f in &b_feats[..nb] {
            let row = &mut grad.ft_w[f * hl..(f + 1) * hl];
            for h in 0..hl {
                row[h] += d_b[h];
            }
        }
    }
}

fn adamw_slice(
    w: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    g: &[f32],
    inv_bs: f32,
    lr: f32,
    bc1: f32,
    bc2: f32,
    clamp_max: Option<f32>,
) {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-8;
    for i in 0..w.len() {
        let gi = g[i] * inv_bs;
        let mi = B1 * m[i] + (1.0 - B1) * gi;
        let vi = B2 * v[i] + (1.0 - B2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        let m_hat = mi * bc1;
        let v_hat = vi * bc2;
        let mut nw = w[i] - lr * m_hat / (v_hat.sqrt() + EPS);
        if let Some(cm) = clamp_max {
            nw = nw.clamp(-cm, cm);
        }
        w[i] = nw;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data_arg = String::new();
    let mut out_path = "nets/gen1.nnue".to_string();
    let mut init_path: Option<String> = None;
    let mut hl = 256usize;
    let mut kb = 1usize;
    let mut ob = 8usize;
    let mut epochs = 20usize;
    let mut batch_size = 16_384usize;
    let mut lr_start = 0.001f32;
    let mut lr_end = 0.00005f32;
    let mut wdl_weight = 0.20f32;
    let mut scale_f = DEFAULT_SCALE as f32;
    let mut threads = 14usize;
    let mut seed = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" if i + 1 < args.len() => {
                data_arg = args[i + 1].clone();
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = args[i + 1].clone();
                i += 2;
            }
            "--init" if i + 1 < args.len() => {
                init_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--hl" if i + 1 < args.len() => {
                hl = args[i + 1].parse().unwrap_or(256);
                i += 2;
            }
            "--kb" if i + 1 < args.len() => {
                kb = args[i + 1].parse().unwrap_or(1);
                i += 2;
            }
            "--ob" if i + 1 < args.len() => {
                ob = args[i + 1].parse().unwrap_or(8);
                i += 2;
            }
            "--epochs" if i + 1 < args.len() => {
                epochs = args[i + 1].parse().unwrap_or(20);
                i += 2;
            }
            "--batch" if i + 1 < args.len() => {
                batch_size = args[i + 1].parse().unwrap_or(16_384);
                i += 2;
            }
            "--lr" if i + 1 < args.len() => {
                lr_start = args[i + 1].parse().unwrap_or(0.001);
                i += 2;
            }
            "--lr-end" if i + 1 < args.len() => {
                lr_end = args[i + 1].parse().unwrap_or(0.00005);
                i += 2;
            }
            "--wdl" if i + 1 < args.len() => {
                wdl_weight = args[i + 1].parse().unwrap_or(0.20);
                i += 2;
            }
            "--scale" if i + 1 < args.len() => {
                scale_f = args[i + 1].parse().unwrap_or(DEFAULT_SCALE as f32);
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                threads = args[i + 1].parse().unwrap_or(14);
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or(42);
                i += 2;
            }
            _ => i += 1,
        }
    }

    if data_arg.is_empty() {
        eprintln!("Usage: train_nnue --data f1.bin,f2.bin --out nets/net.nnue [--hl 256] [--kb 1] [--ob 8] [--epochs 20] [--wdl 0.3] [--threads 14]");
        std::process::exit(1);
    }

    let paths: Vec<String> = data_arg
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let dataset = Arc::new(load_dataset(&paths));
    let total_n = dataset.len();
    assert!(total_n > 1000, "dataset too small: {total_n}");

    // Game-disjoint train/val split: hold out the tail of the dataset (unseen games).
    let mut rng = Rng::new(seed);
    let val_n = (total_n / 50).clamp(1000, 50_000);
    let train_n = total_n - val_n;
    let mut train_idx: Vec<usize> = (0..train_n).collect();
    let val_idx: Vec<usize> = (train_n..total_n).collect();

    let mut params = match init_path {
        Some(p) => {
            let net = NnueNet::load(&p).expect("load init nnue");
            hl = net.hl;
            kb = net.king_buckets;
            ob = net.output_buckets;
            println!("Warm-starting from {p} (kb={kb}, hl={hl}, ob={ob})");
            Params::from_nnue(&net)
        }
        None => Params::new(kb, hl, ob, seed + 100),
    };

    let n_ft = params.ft_w.len();
    let n_out = params.out_w.len();
    let mut m_ft_w = vec![0.0f32; n_ft];
    let mut v_ft_w = vec![0.0f32; n_ft];
    let mut m_ft_b = vec![0.0f32; hl];
    let mut v_ft_b = vec![0.0f32; hl];
    let mut m_out_w = vec![0.0f32; n_out];
    let mut v_out_w = vec![0.0f32; n_out];
    let mut m_out_b = vec![0.0f32; ob];
    let mut v_out_b = vec![0.0f32; ob];

    let mut thread_grads: Vec<GradBuf> = (0..threads)
        .map(|_| GradBuf::new(params.n_feats, hl, ob))
        .collect();
    let mut total_grad = GradBuf::new(params.n_feats, hl, ob);

    let total_batches = epochs * (train_n / batch_size).max(1);
    let mut step = 0usize;

    println!(
        "Training NNUE: (768*{kb} -> {hl})x2 -> {ob} (factorized={}) | train={train_n} val={val_n} | epochs={epochs} batch={batch_size} wdl={wdl_weight} lr={lr_start}->{lr_end} | {threads} threads",
        params.factorized
    );
    let t_all = Instant::now();

    let mut best_val = f64::INFINITY;
    let mut best_params: Option<Params> = None;
    let mut best_ep = 0usize;

    for ep in 1..=epochs {
        let t_ep = Instant::now();
        // Shuffle training indices
        for j in (1..train_n).rev() {
            let k = rng.next_usize(j + 1);
            train_idx.swap(j, k);
        }

        let mut ep_loss = 0.0f64;
        let mut ep_cnt = 0usize;

        for batch in train_idx.chunks(batch_size) {
            if batch.len() < batch_size / 2 && train_n >= batch_size {
                continue;
            }
            step += 1;
            let progress = (step as f32 / total_batches as f32).min(1.0);
            let cos = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            let lr = lr_end + (lr_start - lr_end) * cos;

            // Parallel forward + backward over batch slices
            let chunk_sz = batch.len().div_ceil(threads);
            let p_ref = &params;
            let ds_ref = dataset.as_slice();
            thread::scope(|s| {
                for (t, gbuf) in thread_grads.iter_mut().enumerate() {
                    let lo = (t * chunk_sz).min(batch.len());
                    let hi = ((t + 1) * chunk_sz).min(batch.len());
                    if lo >= hi {
                        gbuf.loss_sum = 0.0;
                        continue;
                    }
                    let slice = &batch[lo..hi];
                    s.spawn(move || {
                        gbuf.clear();
                        process_chunk(ds_ref, slice, p_ref, wdl_weight, scale_f, gbuf, true);
                    });
                }
            });

            // Parallel gradient reduction + AdamW for ft_w
            let inv_bs = 1.0 / batch.len() as f32;
            let bc1 = 1.0 / (1.0 - 0.9f32.powi(step as i32));
            let bc2 = 1.0 / (1.0 - 0.999f32.powi(step as i32));

            let active_threads = threads.min(batch.len().div_ceil(chunk_sz));
            let tg_ref = &thread_grads[..active_threads];

            // Reduce small buffers (ft_b, out_w, out_b, loss_sum)
            total_grad.ft_b.fill(0.0);
            total_grad.out_w.fill(0.0);
            total_grad.out_b.fill(0.0);
            let mut batch_loss = 0.0f64;
            for g in tg_ref {
                batch_loss += g.loss_sum;
                for i in 0..hl {
                    total_grad.ft_b[i] += g.ft_b[i];
                }
                for i in 0..n_out {
                    total_grad.out_w[i] += g.out_w[i];
                }
                for i in 0..ob {
                    total_grad.out_b[i] += g.out_b[i];
                }
            }
            ep_loss += batch_loss;
            ep_cnt += batch.len();

            // Parallel reduce + AdamW on ft_w across threads
            let ft_chunk = n_ft.div_ceil(threads);
            thread::scope(|s| {
                let mut w_rest = params.ft_w.as_mut_slice();
                let mut m_rest = m_ft_w.as_mut_slice();
                let mut v_rest = v_ft_w.as_mut_slice();
                let mut offset = 0usize;
                for _ in 0..threads {
                    if w_rest.is_empty() {
                        break;
                    }
                    let len = ft_chunk.min(w_rest.len());
                    let (wc, wr) = w_rest.split_at_mut(len);
                    let (mc, mr) = m_rest.split_at_mut(len);
                    let (vc, vr) = v_rest.split_at_mut(len);
                    w_rest = wr;
                    m_rest = mr;
                    v_rest = vr;
                    let off = offset;
                    offset += len;
                    s.spawn(move || {
                        let mut gsum = vec![0.0f32; len];
                        for g in tg_ref {
                            let src = &g.ft_w[off..off + len];
                            for j in 0..len {
                                gsum[j] += src[j];
                            }
                        }
                        adamw_slice(wc, mc, vc, &gsum, inv_bs, lr, bc1, bc2, Some(MAX_WEIGHT));
                    });
                }
            });

            adamw_slice(
                &mut params.ft_b,
                &mut m_ft_b,
                &mut v_ft_b,
                &total_grad.ft_b,
                inv_bs,
                lr,
                bc1,
                bc2,
                Some(MAX_WEIGHT),
            );
            adamw_slice(
                &mut params.out_w,
                &mut m_out_w,
                &mut v_out_w,
                &total_grad.out_w,
                inv_bs,
                lr,
                bc1,
                bc2,
                Some(MAX_WEIGHT),
            );
            adamw_slice(
                &mut params.out_b,
                &mut m_out_b,
                &mut v_out_b,
                &total_grad.out_b,
                inv_bs,
                lr,
                bc1,
                bc2,
                None,
            );
        }

        // Validation loss
        let val_chunk = val_n.div_ceil(threads);
        let p_ref = &params;
        let ds_ref = dataset.as_slice();
        thread::scope(|s| {
            for (t, gbuf) in thread_grads.iter_mut().enumerate() {
                let lo = (t * val_chunk).min(val_n);
                let hi = ((t + 1) * val_chunk).min(val_n);
                if lo >= hi {
                    gbuf.loss_sum = 0.0;
                    continue;
                }
                let slice = &val_idx[lo..hi];
                s.spawn(move || {
                    gbuf.loss_sum = 0.0;
                    process_chunk(ds_ref, slice, p_ref, wdl_weight, scale_f, gbuf, false);
                });
            }
        });
        let val_loss: f64 = thread_grads.iter().map(|g| g.loss_sum).sum::<f64>() / val_n as f64;
        let train_loss = ep_loss / ep_cnt as f64;
        let dt = t_ep.elapsed().as_secs_f64();
        let lr_now = {
            let prog = (step as f32 / total_batches as f32).min(1.0);
            lr_end + (lr_start - lr_end) * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos())
        };
        let marker = if val_loss < best_val {
            best_val = val_loss;
            best_params = Some(params.clone());
            best_ep = ep;
            " *"
        } else {
            ""
        };
        println!(
            "Epoch {ep:2}/{epochs} | train_loss={train_loss:.6} val_loss={val_loss:.6}{marker} | lr={lr_now:.6} | {dt:.2}s ({:.0}k pos/s)",
            ep_cnt as f64 / dt / 1e3
        );
    }

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    let final_params = best_params.as_ref().unwrap_or(&params);
    let net = final_params.quantize(scale_f.round() as i32);
    net.save(&out_path).expect("save nnue");
    println!(
        "Saved best quantized NNUE (epoch {best_ep}, val_loss={best_val:.6}) to {out_path} ({:.1} KB) in {:.1}s total",
        std::fs::metadata(&out_path).map(|m| m.len() as f64 / 1024.0).unwrap_or(0.0),
        t_all.elapsed().as_secs_f64()
    );
}
