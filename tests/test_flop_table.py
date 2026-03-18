from __future__ import annotations

import json
import struct

import numpy as np

from submission.flop_table import (
    FlopDiscardTable,
    apply_suit_permutation,
    canonicalize_flop_state,
    quantize_u16,
)


def test_canonicalization_invariant_under_suit_renaming() -> None:
    my5 = (0, 10, 20, 5, 14)
    flop3 = (2, 11, 24)
    perm = (1, 2, 0)

    my5_perm = tuple(apply_suit_permutation(card, perm) for card in my5)
    flop_perm = tuple(apply_suit_permutation(card, perm) for card in flop3)

    canon_a = canonicalize_flop_state(my5, flop3)
    canon_b = canonicalize_flop_state(my5_perm, flop_perm)

    assert canon_a.canonical_hole == canon_b.canonical_hole
    assert canon_a.canonical_flop == canon_b.canonical_flop


def test_canonical_position_mapping_roundtrip() -> None:
    my5 = (18, 0, 9, 1, 10)
    flop3 = (2, 11, 20)
    canon = canonicalize_flop_state(my5, flop3)

    for original_pos in range(5):
        canonical_pos = canon.original_to_canonical_hole_pos[original_pos]
        roundtrip = canon.canonical_to_original_hole_pos[canonical_pos]
        assert roundtrip == original_pos


def test_table_lookup_maps_keep_positions_back_to_original(tmp_path) -> None:
    my5 = (18, 0, 9, 1, 10)
    flop3 = (2, 11, 20)
    canon = canonicalize_flop_state(my5, flop3)
    canonical_key = canon.canonical_hole + canon.canonical_flop

    best_keep_idx = 0  # canonical positions (0, 1)
    second_keep_idx = 1
    best_ev = 0.75
    gap = 0.10

    table_path = tmp_path / "bb_table.bin"
    index_path = tmp_path / "bb_index.npy"
    meta_path = tmp_path / "bb_meta.json"

    with table_path.open("wb") as f:
        f.write(
            struct.pack(
                "<BBHH",
                best_keep_idx,
                second_keep_idx,
                quantize_u16(best_ev),
                quantize_u16(gap),
            )
        )

    np.save(index_path, {canonical_key: 0}, allow_pickle=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"version": 1, "generated_states": 1}, f)

    table = FlopDiscardTable(table_path, index_path, meta_path)
    picked = table.choose_keep_positions(my5, flop3)
    assert picked is not None

    expected_a = canon.canonical_to_original_hole_pos[0]
    expected_b = canon.canonical_to_original_hole_pos[1]
    got_a, got_b, got_ev = picked
    assert {got_a, got_b} == {expected_a, expected_b}
    assert abs(got_ev - best_ev) < 1e-4
