//! Elo difference, 95% confidence interval, LOS, and Pentanomial SPRT (Fishtest style).

#[derive(Clone, Copy, Debug, Default)]
pub struct MatchStats {
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
    /// Pentanomial pair outcomes: [LL, LD, DD+WL, WD, WW]
    pub penta: [u64; 5],
}

impl MatchStats {
    pub fn total(&self) -> u64 {
        self.wins + self.draws + self.losses
    }

    pub fn score(&self) -> f64 {
        let n = self.total();
        if n == 0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / n as f64
    }

    /// Record a completed color-swapped pair `(score_game1, score_game2)` where each score is `0` (loss), `1` (draw), or `2` (win).
    pub fn record_pair(&mut self, s1: u8, s2: u8) {
        for s in [s1, s2] {
            match s {
                2 => self.wins += 1,
                1 => self.draws += 1,
                _ => self.losses += 1,
            }
        }
        let idx = (s1 + s2).min(4) as usize;
        self.penta[idx] += 1;
    }

    /// Returns `(elo_diff, ci_95_halfwidth, los_percent)`.
    /// Uses pentanomial variance when pairs are recorded (tighter CI on UHO books).
    pub fn elo_ci(&self) -> (f64, f64, f64) {
        let n_pairs: u64 = self.penta.iter().sum();
        if n_pairs >= 5 && n_pairs * 2 == self.total() {
            let np = n_pairs as f64;
            let vals = [0.0, 0.25, 0.5, 0.75, 1.0];
            let mut mu = 0.0;
            for i in 0..5 {
                mu += vals[i] * (self.penta[i] as f64 / np);
            }
            let mut var = 0.0;
            for i in 0..5 {
                var += (self.penta[i] as f64 / np) * (vals[i] - mu).powi(2);
            }
            let se = (var / np).sqrt();
            let elo = score_to_elo(mu);
            let lo = score_to_elo((mu - 1.96 * se).clamp(1e-4, 1.0 - 1e-4));
            let hi = score_to_elo((mu + 1.96 * se).clamp(1e-4, 1.0 - 1e-4));
            let ci = 0.5 * (hi - lo);
            let z = if se > 1e-12 { (mu - 0.5) / se } else { 0.0 };
            let los = 50.0 * (1.0 + erf(z / std::f64::consts::SQRT_2));
            return (elo, ci, los);
        }

        let n = self.total() as f64;
        if n == 0.0 {
            return (0.0, 0.0, 50.0);
        }
        let mu = self.score();
        let w = self.wins as f64 / n;
        let d = self.draws as f64 / n;
        let l = self.losses as f64 / n;
        let var = w * (1.0 - mu).powi(2) + d * (0.5 - mu).powi(2) + l * (0.0 - mu).powi(2);
        let se = (var / n).sqrt();
        let elo = score_to_elo(mu);
        let lo = score_to_elo((mu - 1.96 * se).clamp(1e-4, 1.0 - 1e-4));
        let hi = score_to_elo((mu + 1.96 * se).clamp(1e-4, 1.0 - 1e-4));
        let ci = 0.5 * (hi - lo);
        let z = if se > 1e-12 { (mu - 0.5) / se } else { 0.0 };
        let los = 50.0 * (1.0 + erf(z / std::f64::consts::SQRT_2));
        (elo, ci, los)
    }

    /// Pentanomial normalized-Elo SPRT log-likelihood ratio (GSPRT / Fishtest formula).
    pub fn sprt_llr(&self, elo0: f64, elo1: f64) -> f64 {
        let n_pairs: u64 = self.penta.iter().sum();
        if n_pairs < 4 {
            return 0.0;
        }
        // Regularize empty buckets with 0.5 pseudocount to avoid zero variance early on
        let np = n_pairs as f64 + 2.5;
        let vals = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut p = [0.0f64; 5];
        let mut mu = 0.0;
        for i in 0..5 {
            p[i] = (self.penta[i] as f64 + 0.5) / np;
            mu += vals[i] * p[i];
        }
        let mut var = 0.0;
        for i in 0..5 {
            var += p[i] * (vals[i] - mu).powi(2);
        }
        if var < 1e-9 {
            return 0.0;
        }
        let s0 = elo_to_score(elo0);
        let s1 = elo_to_score(elo1);
        let n = n_pairs as f64;
        // Generalized SPRT LLR (exact asymptotic match to Fishtest pentanomial LLR)
        0.5 * n * (((mu - s0).powi(2) - (mu - s1).powi(2)) / var)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SprtConfig {
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl SprtConfig {
    pub fn new(elo0: f64, elo1: f64) -> Self {
        Self {
            elo0,
            elo1,
            alpha: 0.05,
            beta: 0.05,
        }
    }

    pub fn lower_bound(&self) -> f64 {
        (self.beta / (1.0 - self.alpha)).ln()
    }

    pub fn upper_bound(&self) -> f64 {
        ((1.0 - self.beta) / self.alpha).ln()
    }
}

pub fn elo_to_score(elo: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-elo / 400.0))
}

pub fn score_to_elo(score: f64) -> f64 {
    let s = score.clamp(1e-4, 1.0 - 1e-4);
    -400.0 * (1.0 / s - 1.0).log10()
}

/// Abramowitz & Stegun 7.1.26 erf approximation (max error 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}
