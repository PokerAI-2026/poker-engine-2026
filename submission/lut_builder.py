from __future__ import annotations

import argparse
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from gym_env import PokerEnv

try:
    from submission.lut_store import (
        HAND5_SIZE,
        N_CARDS,
        PAIR_SIZE,
        SEVEN_TO_FIVE_SUBSETS,
        combo_to_index,
        pack_flop_key,
        pair_to_index,
    )
except ImportError:
    from lut_store import (  # type: ignore
        HAND5_SIZE,
        N_CARDS,
        PAIR_SIZE,
        SEVEN_TO_FIVE_SUBSETS,
        combo_to_index,
        pack_flop_key,
        pair_to_index,
    )


def evaluate_7_from_lut(hand5_strength: np.ndarray, hole2: tuple[int, int], board5: list[int]) -> int:
    cards7 = [hole2[0], hole2[1], *board5]
    best = 10**9
    for subset in SEVEN_TO_FIVE_SUBSETS:
        score = int(hand5_strength[combo_to_index([cards7[i] for i in subset])])
        if score < best:
            best = score
    return best


def generate_hand5_strength() -> np.ndarray:
    env = PokerEnv()
    evaluator = env.evaluator
    arr = np.zeros(HAND5_SIZE, dtype=np.int32)

    for cards in combinations(range(N_CARDS), 5):
        idx = combo_to_index(cards)
        treys_cards = [env.int_to_card(c) for c in cards]
        arr[idx] = evaluator.evaluate(treys_cards[:2], treys_cards[2:])
    return arr


def generate_pair_equity(hand5_strength: np.ndarray, samples_per_pair: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    pair_equity = np.zeros(PAIR_SIZE, dtype=np.float32)

    all_cards = list(range(N_CARDS))
    for pair in combinations(all_cards, 2):
        idx = pair_to_index(pair[0], pair[1])
        available = [c for c in all_cards if c not in pair]
        wins = ties = 0.0
        for _ in range(samples_per_pair):
            draw = rng.sample(available, 7)  # opp(2) + board(5)
            opp = (draw[0], draw[1])
            board5 = draw[2:7]
            my_score = evaluate_7_from_lut(hand5_strength, pair, board5)
            opp_score = evaluate_7_from_lut(hand5_strength, opp, board5)
            if my_score < opp_score:
                wins += 1.0
            elif my_score == opp_score:
                ties += 1.0
        pair_equity[idx] = (wins + 0.5 * ties) / samples_per_pair
    return pair_equity


def build_preflop_equity_from_pairs(pair_equity: np.ndarray) -> np.ndarray:
    return build_preflop_equity_from_pairs_topk(pair_equity, top_k=1)


def build_preflop_equity_from_pairs_topk(pair_equity: np.ndarray, top_k: int) -> np.ndarray:
    effective_top_k = max(1, min(10, int(top_k)))
    preflop = np.zeros(HAND5_SIZE, dtype=np.float32)
    for hand5 in combinations(range(N_CARDS), 5):
        idx5 = combo_to_index(hand5)
        pair_values: list[float] = []
        for a, b in combinations(hand5, 2):
            pair_values.append(float(pair_equity[pair_to_index(a, b)]))
        pair_values.sort(reverse=True)
        preflop[idx5] = float(sum(pair_values[:effective_top_k]) / effective_top_k)
    return preflop


def estimate_flop_ev(
    hand5_strength: np.ndarray,
    keep_pair: tuple[int, int],
    flop: tuple[int, int, int],
    samples: int,
    rng: random.Random,
) -> float:
    blocked = {keep_pair[0], keep_pair[1], flop[0], flop[1], flop[2]}
    available = [c for c in range(N_CARDS) if c not in blocked]
    wins = ties = 0.0
    for _ in range(samples):
        draw = rng.sample(available, 4)  # opp(2) + turn/river(2)
        opp = (draw[0], draw[1])
        board5 = [flop[0], flop[1], flop[2], draw[2], draw[3]]
        my_score = evaluate_7_from_lut(hand5_strength, keep_pair, board5)
        opp_score = evaluate_7_from_lut(hand5_strength, opp, board5)
        if my_score < opp_score:
            wins += 1.0
        elif my_score == opp_score:
            ties += 1.0
    return (wins + 0.5 * ties) / samples


def generate_flop_seed_table(
    hand5_strength: np.ndarray,
    target_states: int,
    flop_samples: int,
    seed: int,
) -> dict[int, float]:
    rng = random.Random(seed)
    table: dict[int, float] = {}
    pairs = list(combinations(range(N_CARDS), 2))

    while len(table) < target_states:
        keep = pairs[rng.randrange(len(pairs))]
        remaining = [c for c in range(N_CARDS) if c not in keep]
        flop = tuple(sorted(rng.sample(remaining, 3)))
        key = pack_flop_key(keep, flop)
        if key in table:
            continue
        table[key] = estimate_flop_ev(hand5_strength, keep, flop, flop_samples, rng)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline LUT artifacts for submission bot.")
    parser.add_argument("--output-dir", default="submission/data", help="Directory for .npy artifacts")
    parser.add_argument("--pair-samples", type=int, default=500, help="Samples per preflop pair equity")
    parser.add_argument("--flop-seed-states", type=int, default=3000, help="Number of seeded flop states")
    parser.add_argument("--flop-samples", type=int, default=260, help="Samples per seeded flop state")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic RNG seed")
    parser.add_argument(
        "--preflop-top-k",
        type=int,
        default=1,
        help="Aggregate each 5-card preflop hand using the top-k pair equities (1..10).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hand5_path = out_dir / "hand5_strength.npy"
    preflop_path = out_dir / "preflop_equity.npy"
    flop_path = out_dir / "flop_seed_table.npy"
    if not args.overwrite and hand5_path.exists() and preflop_path.exists() and flop_path.exists():
        print("All LUT files already exist. Use --overwrite to regenerate.")
        return

    t0 = time.time()
    print("Generating hand5 strength LUT...")
    hand5_strength = generate_hand5_strength()
    np.save(hand5_path, hand5_strength)
    print(f"Saved: {hand5_path}")

    print("Generating preflop pair equities...")
    pair_equity = generate_pair_equity(hand5_strength, samples_per_pair=args.pair_samples, seed=args.seed)
    print("Generating preflop 5-card equities...")
    preflop_equity = build_preflop_equity_from_pairs_topk(
        pair_equity, top_k=args.preflop_top_k
    )
    np.save(preflop_path, preflop_equity)
    print(f"Saved: {preflop_path}")

    print("Generating seeded flop transition table...")
    flop_seed = generate_flop_seed_table(
        hand5_strength,
        target_states=args.flop_seed_states,
        flop_samples=args.flop_samples,
        seed=args.seed ^ 0xBEEF,
    )
    np.save(flop_path, flop_seed, allow_pickle=True)
    print(f"Saved: {flop_path}")

    elapsed = time.time() - t0
    print(f"Completed LUT generation in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
