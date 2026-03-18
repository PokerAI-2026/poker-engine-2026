from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path

import numpy as np

from submission.flop_table import canonicalize_flop_state, quantize_u16
from submission.lut_store import KEEP_INDEX_PAIRS, LUTStore, N_CARDS


def estimate_keep_equity(
    luts: LUTStore,
    keep_pair: tuple[int, int],
    flop3: tuple[int, int, int],
    samples: int,
    rng: random.Random,
) -> float:
    blocked = {int(keep_pair[0]), int(keep_pair[1]), int(flop3[0]), int(flop3[1]), int(flop3[2])}
    available = [card for card in range(N_CARDS) if card not in blocked]

    wins = 0.0
    ties = 0.0
    for _ in range(samples):
        draw = rng.sample(available, 4)  # opp(2) + turn/river(2)
        opp_hole = (int(draw[0]), int(draw[1]))
        board = [int(flop3[0]), int(flop3[1]), int(flop3[2]), int(draw[2]), int(draw[3])]

        my_score = luts.evaluate_7card_score(keep_pair, board)
        opp_score = luts.evaluate_7card_score(opp_hole, board)
        if my_score < opp_score:
            wins += 1.0
        elif my_score == opp_score:
            ties += 1.0
    return (wins + 0.5 * ties) / float(samples)


def evaluate_flop_state(
    luts: LUTStore,
    canonical_hole: tuple[int, int, int, int, int],
    canonical_flop: tuple[int, int, int],
    samples_per_keep: int,
    rng: random.Random,
) -> tuple[int, int, float, float]:
    scored: list[tuple[float, int]] = []
    for keep_idx, (i, j) in enumerate(KEEP_INDEX_PAIRS):
        keep_pair = (int(canonical_hole[i]), int(canonical_hole[j]))
        ev = estimate_keep_equity(luts, keep_pair, canonical_flop, samples_per_keep, rng)
        scored.append((float(ev), keep_idx))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    best_ev, best_idx = scored[0]
    second_ev, second_idx = scored[1]
    gap = max(0.0, best_ev - second_ev)
    return best_idx, second_idx, best_ev, gap


def write_records(path: Path, records: list[tuple[int, int, float, float]]) -> None:
    with path.open("wb") as f:
        for best_keep_idx, second_keep_idx, best_ev, gap in records:
            packed = struct.pack(
                "<BBHH",
                int(best_keep_idx),
                int(second_keep_idx),
                quantize_u16(best_ev),
                quantize_u16(gap),
            )
            f.write(packed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sampled BB flop discard table artifacts."
    )
    parser.add_argument(
        "--output-dir",
        default="submission/data",
        help="Directory for bb_table.bin / bb_index.npy / bb_meta.json",
    )
    parser.add_argument(
        "--target-states",
        type=int,
        default=20000,
        help="Number of unique canonical flop states to store",
    )
    parser.add_argument(
        "--samples-per-keep",
        type=int,
        default=120,
        help="Monte Carlo samples per keep pair",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic RNG seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N generated states",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "bb_table.bin"
    index_path = out_dir / "bb_index.npy"
    meta_path = out_dir / "bb_meta.json"
    if not args.overwrite and (table_path.exists() or index_path.exists() or meta_path.exists()):
        print("BB table files already exist. Use --overwrite to regenerate.")
        return

    t0 = time.time()
    luts = LUTStore(out_dir)
    rng = random.Random(args.seed)

    all_cards = list(range(N_CARDS))
    index_map: dict[tuple[int, ...], int] = {}
    records: list[tuple[int, int, float, float]] = []

    attempts = 0
    max_attempts = max(10000, args.target_states * 20)
    while len(records) < args.target_states and attempts < max_attempts:
        attempts += 1
        hole = tuple(rng.sample(all_cards, 5))
        remaining = [c for c in all_cards if c not in hole]
        flop = tuple(rng.sample(remaining, 3))

        canon = canonicalize_flop_state(hole, flop)
        canonical_key = canon.canonical_hole + canon.canonical_flop
        if canonical_key in index_map:
            continue

        record = evaluate_flop_state(
            luts=luts,
            canonical_hole=canon.canonical_hole,
            canonical_flop=canon.canonical_flop,
            samples_per_keep=args.samples_per_keep,
            rng=rng,
        )
        state_id = len(records)
        index_map[canonical_key] = state_id
        records.append(record)

        if len(records) % max(1, args.progress_every) == 0:
            elapsed = time.time() - t0
            print(
                f"Generated {len(records):,}/{args.target_states:,} states "
                f"(attempts={attempts:,}, elapsed={elapsed:.1f}s)"
            )

    if len(records) < args.target_states:
        print(
            f"Warning: generated {len(records):,} states after {attempts:,} attempts "
            f"(target={args.target_states:,})"
        )

    write_records(table_path, records)
    np.save(index_path, index_map, allow_pickle=True)

    meta = {
        "version": 1,
        "format": "bb-discard-table-v1",
        "record_size_bytes": 6,
        "endianness": "little",
        "target_states": int(args.target_states),
        "generated_states": int(len(records)),
        "samples_per_keep": int(args.samples_per_keep),
        "seed": int(args.seed),
        "keep_choices": [list(pair) for pair in KEEP_INDEX_PAIRS],
        "card_encoding": {"rank": "card % 9", "suit": "card // 9"},
        "canonicalization": "best lexicographic state over all 6 global suit permutations",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"Saved: {table_path}")
    print(f"Saved: {index_path}")
    print(f"Saved: {meta_path}")
    print(f"Finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
