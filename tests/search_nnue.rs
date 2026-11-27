//! Tests for alpha-beta search, tactical mates, SEE, NNUE SIMD/incremental exactness, and PackedPos.

use chatarunga::data::PackedPos;
use chatarunga::nnue::{
    AccStack, NnueNet, DEFAULT_QA, DEFAULT_QB, DEFAULT_SCALE, VERSION_DEEP_THREATS, VERSION_PAIRWISE, VERSION_SCRELU,
    VERSION_SCRELU_ASYM,
};
use chatarunga::search::{EvalMode, SearchLimits, Searcher, MATE_BOUND};
use chatarunga::{see_ge, Board, MoveList, Rng};

fn random_net(version: u32, king_buckets: usize, hl: usize, output_buckets: usize, seed: u64) -> NnueNet {
    let mut rng = Rng::new(seed);
    if version == VERSION_DEEP_THREATS {
        let has_threats = true;
        let l1_size = 16;
        let l2_size = 32;
        let n_ft = (king_buckets + 3) * 768 * hl;
        let ft_w: Vec<i16> = (0..n_ft).map(|_| ((rng.next_u64() % 121) as i16) - 60).collect();
        let ft_b: Vec<i16> = (0..hl).map(|_| ((rng.next_u64() % 81) as i16) - 40).collect();
        let l1_w: Vec<i8> = (0..output_buckets * l1_size * hl).map(|_| ((rng.next_u64() % 121) as i8) - 60).collect();
        let l1_w_t = chatarunga::nnue::pack_l1_w_transposed(&l1_w, output_buckets, hl);
        let l1_b: Vec<i32> = (0..output_buckets * l1_size).map(|_| ((rng.next_u64() % 4001) as i32) - 2000).collect();
        let l2_w: Vec<i8> =
            (0..output_buckets * l2_size * (2 * l1_size)).map(|_| ((rng.next_u64() % 121) as i8) - 60).collect();
        let l2_b: Vec<i32> = (0..output_buckets * l2_size).map(|_| ((rng.next_u64() % 4001) as i32) - 2000).collect();
        let out_w_i8: Vec<i8> = (0..output_buckets * l2_size).map(|_| ((rng.next_u64() % 121) as i8) - 60).collect();
        let out_b: Vec<i32> = (0..output_buckets).map(|_| ((rng.next_u64() % 4001) as i32) - 2000).collect();
        return NnueNet {
            version,
            king_buckets,
            hl,
            output_buckets,
            qa: DEFAULT_QA,
            qb: DEFAULT_QB,
            scale: 160,
            has_threats,
            l1_size,
            l2_size,
            ft_w,
            ft_b,
            out_w: Vec::new(),
            out_b,
            l1_w,
            l1_w_t,
            l1_b,
            l2_w,
            l2_b,
            out_w_i8,
        };
    }
    let n_ft = king_buckets * 768 * hl;
    let n_out = if version == VERSION_PAIRWISE { output_buckets * hl } else { output_buckets * 2 * hl };
    let ft_w: Vec<i16> = (0..n_ft).map(|_| ((rng.next_u64() % 121) as i16) - 60).collect();
    let ft_b: Vec<i16> = (0..hl).map(|_| ((rng.next_u64() % 81) as i16) - 40).collect();
    let out_w: Vec<i16> = (0..n_out).map(|_| ((rng.next_u64() % 161) as i16) - 80).collect();
    let out_b: Vec<i32> = (0..output_buckets).map(|_| ((rng.next_u64() % 2001) as i32) - 1000).collect();
    NnueNet {
        version,
        king_buckets,
        hl,
        output_buckets,
        qa: DEFAULT_QA,
        qb: DEFAULT_QB,
        scale: DEFAULT_SCALE,
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

#[test]
fn fast_check_and_tt_legality_match_oracle() {
    let mut rng = Rng::new(777);
    let mut ml = MoveList::new();
    for _ in 0..100 {
        let mut b = Board::startpos();
        for _ in 0..80 {
            b.legal_moves(&mut ml);
            if ml.is_empty() {
                break;
            }
            let chk_sq = b.check_squares();
            let dc = b.discovered_check_candidates();
            for &m in ml.as_slice() {
                assert!(b.is_legal_tt_move(m), "legal move {m} rejected at {}", b.to_fen());
                assert_eq!(
                    b.gives_check_fast(m, &chk_sq, dc),
                    b.make_move(m).in_check(),
                    "gives_check_fast mismatch on {m} at {}",
                    b.to_fen()
                );
            }
            let m = ml[rng.next_usize(ml.len())];
            b.apply_move(m);
        }
    }
}

#[test]
fn see_sanity_checks() {
    // Winning capture: QxR undefended, PxN defended by pawn -> good.
    let b = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let m = b.parse_uci_move("e4d5").unwrap();
    assert!(see_ge(&b, m, 0));
    assert!(see_ge(&b, m, 100));
    assert!(!see_ge(&b, m, 101));

    // Losing capture: QxP defended by pawn
    let b2 = Board::from_fen("4k3/8/2p5/3p4/8/3Q4/8/4K3 w - - 0 1").unwrap();
    let m2 = b2.parse_uci_move("d3d5").unwrap();
    assert!(!see_ge(&b2, m2, 0));
    assert!(see_ge(&b2, m2, -850));
}

#[test]
fn search_finds_mate_in_1_and_2() {
    let mut s = Searcher::new(EvalMode::Classical, 16);

    // Mate in 1: White Qf7#
    let b1 = Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4").unwrap();
    let r1 = s.search(&b1, &[b1.hash], SearchLimits { max_depth: 4, ..Default::default() });
    assert_eq!(r1.best_move.to_uci(), "h5f7");
    assert!(r1.score >= MATE_BOUND);

    // Mate in 2: 1. Qg7+ Rxg7 2. Rf8# (classic back-rank deflection)
    s.clear();
    let b2 = Board::from_fen("6k1/5ppp/8/8/8/8/1Q3PPP/5RK1 w - - 0 1").unwrap();
    let r2 = s.search(&b2, &[b2.hash], SearchLimits { max_depth: 5, ..Default::default() });
    assert!(r2.score > 0);

    // Forced mate in 2: 1. Re8+ Rxe8 2. Rxe8#
    s.clear();
    let b3 = Board::from_fen("2r3k1/5ppp/8/8/8/8/4RPPP/4R1K1 w - - 0 1").unwrap();
    let r3 = s.search(&b3, &[b3.hash], SearchLimits { max_depth: 6, ..Default::default() });
    assert_eq!(r3.best_move.to_uci(), "e2e8");
    assert!(r3.score >= MATE_BOUND);

    // Lazy SMP 4-thread search sanity & lockless TT integrity check
    s.set_threads(4);
    s.clear();
    let r4 = s.search(&b3, &[b3.hash], SearchLimits { max_depth: 6, ..Default::default() });
    assert_eq!(r4.best_move.to_uci(), "e2e8");
    assert!(r4.score >= MATE_BOUND);
    let kiwi = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let rk = s.search(&kiwi, &[kiwi.hash], SearchLimits { soft_ms: Some(40), hard_ms: Some(40), ..Default::default() });
    assert!(kiwi.is_legal_tt_move(rk.best_move));
    assert!(rk.nodes > 0);
}

#[test]
fn nnue_simd_matches_scalar_and_incremental_matches_full_refresh() {
    for &(ver, kb, hl, ob) in &[
        (VERSION_SCRELU, 1, 256, 8),
        (VERSION_SCRELU, 4, 256, 8),
        (VERSION_SCRELU, 8, 512, 8),
        (VERSION_SCRELU_ASYM, 16, 1024, 8),
        (VERSION_PAIRWISE, 8, 1024, 8),
        (VERSION_DEEP_THREATS, 16, 1024, 8),
    ] {
        let net = random_net(ver, kb, hl, ob, 12345 + kb as u64 + ver as u64 * 100);
        let mut stack = AccStack::new(hl, 128);
        let mut rng = Rng::new(999 + kb as u64);
        let mut ml = MoveList::new();

        for _game in 0..20 {
            let mut b = Board::startpos();
            stack.init_root(&net, &b);
            for ply in 0..60 {
                let inc_eval = stack.evaluate(&net, ply, &b);
                let full_eval = net.evaluate_board(&b);
                assert_eq!(
                    inc_eval,
                    full_eval,
                    "incremental != full at ver={ver} kb={kb} ply={ply} fen={}",
                    b.to_fen()
                );

                // Also verify SIMD == scalar on full refresh (including threats for v3)
                let mut w = vec![0i16; hl];
                let mut bl = vec![0i16; hl];
                net.refresh_perspective(&b, chatarunga::Color::White, &mut w);
                net.refresh_perspective(&b, chatarunga::Color::Black, &mut bl);
                if net.has_threats {
                    net.add_threat_features(&b, &mut w, &mut bl);
                }
                let bucket = chatarunga::nnue::output_bucket(&b, ob);
                let (s_acc, o_acc) = if b.side == chatarunga::Color::White { (&w, &bl) } else { (&bl, &w) };
                assert_eq!(net.eval_from_accs(s_acc, o_acc, bucket), net.eval_from_accs_scalar(s_acc, o_acc, bucket));

                b.legal_moves(&mut ml);
                if ml.is_empty() {
                    break;
                }
                let m = ml[rng.next_usize(ml.len())];
                let next = b.make_move(m);
                stack.push_move(ply, &b, &next, m);
                b = next;
            }
        }
    }
}

#[test]
fn nnue_save_load_roundtrip_and_packed_pos() {
    let b = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    for &(ver, kb, hl) in &[(VERSION_PAIRWISE, 8, 256), (VERSION_DEEP_THREATS, 16, 256)] {
        let net = random_net(ver, kb, hl, 8, 77 + ver as u64);
        let path = std::env::temp_dir().join(format!("chess_ai_test_roundtrip_v{ver}.nnue"));
        net.save(&path).unwrap();
        let loaded = NnueNet::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(net.evaluate_board(&b), loaded.evaluate_board(&b));
    }

    let packed = PackedPos::pack(&b, 123, 2);
    let bytes = packed.to_bytes();
    let unpacked = PackedPos::from_bytes(&bytes);
    assert_eq!(packed, unpacked);
    let mut pcs = [(0u8, chatarunga::Color::White, chatarunga::Piece::Pawn); 32];
    let (cnt, kings) = unpacked.decode_pieces(&mut pcs);
    assert_eq!(cnt, b.occ().count_ones() as usize);
    assert_eq!(kings[0], b.king_sq(chatarunga::Color::White));
    assert_eq!(kings[1], b.king_sq(chatarunga::Color::Black));
}
