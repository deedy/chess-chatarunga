//! Comprehensive chess rules, invariants, Zobrist consistency, FEN/SAN/UCI tests.

use chatarunga::{Board, Color, DrawReason, Game, MoveList, Outcome, Rng, STARTPOS_FEN};

#[test]
fn fen_roundtrip_positions() {
    let fens = [
        STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    ];
    for fen in fens {
        let b = Board::from_fen(fen).unwrap();
        assert_eq!(b.to_fen(), fen);
        assert_eq!(b.hash, b.compute_hash());
    }
}

#[test]
fn en_passant_horizontal_pin_disallowed() {
    // White king a5, white pawn e5, black pawn d5 (ep d6), black rook h5.
    // exd6 would remove both pawns from rank 5, exposing white king to rook on h5.
    let b = Board::from_fen("8/8/8/K2pP2r/8/8/8/7k w - d6 0 1").unwrap();
    assert!(
        b.parse_uci_move("e5d6").is_none(),
        "horizontally pinned EP must be illegal"
    );
    // Same position without rook: EP must be legal.
    let b2 = Board::from_fen("8/8/8/K2pP3/8/8/8/7k w - d6 0 1").unwrap();
    assert!(
        b2.parse_uci_move("e5d6").is_some(),
        "unpinned EP must be legal"
    );
}

#[test]
fn en_passant_escapes_check() {
    // Black pawn on d5 just double-pushed giving check to White king on e4; White pawn on c5 can capture cxd6 ep.
    let b = Board::from_fen("4k3/8/8/2Pp4/4K3/8/8/8 w - d6 0 1").unwrap();
    assert!(b.in_check());
    let ep = b
        .parse_uci_move("c5d6")
        .expect("EP capturing checking pawn must be legal");
    let next = b.make_move(ep);
    assert!(!next.is_sq_attacked(next.king_sq(Color::White), next.occ(), Color::Black));
}

#[test]
fn castling_rules_and_rights_loss() {
    // Cannot castle through attacked square (f1 attacked by black rook on f8).
    let b = Board::from_fen("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
    assert!(b.parse_uci_move("e1g1").is_none());
    assert!(b.parse_uci_move("e1c1").is_some());

    // Capturing opponent's rook on h8 removes black's kingside castling right.
    let b2 = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    let cap = b2.parse_uci_move("h1h8").unwrap();
    let after = b2.make_move(cap);
    assert_eq!(after.to_fen().split_whitespace().nth(2).unwrap(), "Qq");
}

#[test]
fn fools_mate_and_stalemate() {
    let mut g = Game::new();
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"] {
        let m = g.board.parse_uci_move(uci).unwrap();
        g.apply_move(m);
    }
    assert_eq!(g.outcome(), Some(Outcome::Win(Color::Black)));

    let g_stale = Game::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    assert_eq!(
        g_stale.outcome(),
        Some(Outcome::Draw(DrawReason::Stalemate))
    );
}

#[test]
fn threefold_repetition_and_fifty_move() {
    let mut g = Game::new();
    for uci in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        let m = g.board.parse_uci_move(uci).unwrap();
        g.apply_move(m);
    }
    assert_eq!(g.outcome(), Some(Outcome::Draw(DrawReason::Repetition)));

    let g50 = Game::from_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 100 60").unwrap();
    assert_eq!(g50.outcome(), Some(Outcome::Draw(DrawReason::FiftyMove)));
}

#[test]
fn insufficient_material_cases() {
    assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        .unwrap()
        .is_insufficient_material());
    assert!(Board::from_fen("4k3/8/8/8/8/8/4N3/4K3 w - - 0 1")
        .unwrap()
        .is_insufficient_material());
    assert!(Board::from_fen("4k3/8/8/8/8/8/4B3/4K3 w - - 0 1")
        .unwrap()
        .is_insufficient_material());
    // Same-color bishops (c1 dark, f8 dark -> wait: c1=(2,0) even=dark, f8=(5,7) even=dark)
    assert!(Board::from_fen("4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1")
        .unwrap()
        .is_insufficient_material());
    // Opposite-color bishops (c1 dark, c8 light)
    assert!(!Board::from_fen("2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1")
        .unwrap()
        .is_insufficient_material());
}

#[test]
fn san_formatting_and_disambiguation() {
    let b = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
    assert_eq!(b.move_to_san(b.parse_uci_move("e1g1").unwrap()), "O-O");
    assert_eq!(b.move_to_san(b.parse_uci_move("e1c1").unwrap()), "O-O-O");

    let b2 = Board::from_fen("4k3/8/8/8/8/8/1K6/R5NR w - - 0 1").unwrap();
    let m = b2.parse_uci_move("a1d1").unwrap();
    assert_eq!(b2.move_to_san(m), "Rd1");

    let b3 = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w - - 0 1").unwrap();
    let m3 = b3.parse_uci_move("a1d1").unwrap();
    assert_eq!(b3.move_to_san(m3), "Rd1");

    let b4 = Board::from_fen("4k3/8/8/8/8/8/1K6/R6R w - - 0 1").unwrap();
    assert_eq!(b4.move_to_san(b4.parse_uci_move("a1d1").unwrap()), "Rad1");
    assert_eq!(b4.move_to_san(b4.parse_uci_move("h1d1").unwrap()), "Rhd1");
}

#[test]
fn random_playouts_zobrist_and_tactical_subset_invariants() {
    let mut rng = Rng::new(42);
    let mut ml = MoveList::new();
    let mut tml = MoveList::new();
    let mut qml = MoveList::new();
    for _ in 0..500 {
        let mut g = Game::new();
        for _ in 0..160 {
            assert_eq!(
                g.board.hash,
                g.board.compute_hash(),
                "Zobrist drift at {}",
                g.board.to_fen()
            );
            assert_eq!(g.board.colors[0] & g.board.colors[1], 0);
            let piece_union = g.board.pieces.iter().copied().fold(0u64, |a, b| a | b);
            assert_eq!(piece_union, g.board.occ());

            g.board.legal_moves(&mut ml);
            g.board.tactical_moves(&mut tml);
            // Every tactical move must be a legal move.
            for &tm in tml.as_slice() {
                assert!(
                    ml.as_slice().contains(&tm),
                    "tactical move {tm} not in legal moves at {}",
                    g.board.to_fen()
                );
            }
            // When not in check, tactical_moves and quiet_moves must be a disjoint exact partition of legal_moves.
            if !g.board.in_check() {
                g.board.quiet_moves(&mut qml);
                assert_eq!(
                    tml.len() + qml.len(),
                    ml.len(),
                    "tactical ({}) + quiet ({}) != legal ({}) at {}",
                    tml.len(),
                    qml.len(),
                    ml.len(),
                    g.board.to_fen()
                );
                for &qm in qml.as_slice() {
                    assert!(!qm.is_tactical(), "quiet_moves emitted tactical {qm}");
                    assert!(ml.as_slice().contains(&qm), "quiet move {qm} not in legal");
                    assert!(
                        !tml.as_slice().contains(&qm),
                        "overlap {qm} between tactical and quiet"
                    );
                }
            }
            // Verify attacks_by matches is_sq_attacked for every square.
            let occ = g.board.occ();
            let enemy = g.board.side.flip();
            let att_bb = g.board.enemy_attacks();
            for sq in 0..64u8 {
                let bit = (att_bb & (1u64 << sq)) != 0;
                assert_eq!(
                    bit,
                    g.board.is_sq_attacked(sq, occ, enemy),
                    "attacks_by mismatch sq={sq}"
                );
            }
            if ml.is_empty() || g.outcome().is_some() {
                break;
            }
            let m = ml[rng.next_usize(ml.len())];
            let next = g.board.make_move(m);
            // Our own king must never be left in check after our move.
            assert!(!next.is_sq_attacked(next.king_sq(g.board.side), next.occ(), next.side));
            g.apply_move(m);
        }
    }
}
