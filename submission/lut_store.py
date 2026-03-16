from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

N_CARDS = 27
HAND5_SIZE = 80730  # C(27, 5)
PAIR_SIZE = 351  # C(27, 2)

_NCK = np.zeros((N_CARDS + 1, 6), dtype=np.int64)
for n in range(N_CARDS + 1):
    _NCK[n, 0] = 1
    for k in range(1, min(5, n) + 1):
        _NCK[n, k] = _NCK[n - 1, k - 1] + _NCK[n - 1, k]

SEVEN_TO_FIVE_SUBSETS = tuple(combinations(range(7), 5))
KEEP_INDEX_PAIRS = tuple(combinations(range(5), 2))


def combo_to_index(cards: Sequence[int]) -> int:
    """
    Combinadic index (colex) for sorted card combinations.
    """
    sorted_cards = sorted(int(c) for c in cards)
    idx = 0
    for i, card in enumerate(sorted_cards):
        idx += int(_NCK[card, i + 1])
    return idx


def pair_to_index(card_a: int, card_b: int) -> int:
    lo, hi = (card_a, card_b) if card_a < card_b else (card_b, card_a)
    return int(_NCK[lo, 1] + _NCK[hi, 2])


def pack_flop_key(keep_cards: Sequence[int], flop_cards: Sequence[int]) -> int:
    """
    Packs keep/flop cards into a compact integer key.
    Every card id fits in 5 bits (0..26).
    """
    k0, k1 = sorted((int(keep_cards[0]), int(keep_cards[1])))
    f0, f1, f2 = sorted((int(flop_cards[0]), int(flop_cards[1]), int(flop_cards[2])))
    return k0 | (k1 << 5) | (f0 << 10) | (f1 << 15) | (f2 << 20)


class LUTStore:
    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.data_dir = Path(data_dir)
        self.hand5_strength = self._load_hand5_strength()
        self.preflop_equity = self._load_preflop_equity()
        self.flop_cache: Dict[int, float] = self._load_flop_seed_table()
        self._premium_threshold = float(np.quantile(self.preflop_equity, 0.90))

    def _load_hand5_strength(self) -> np.ndarray:
        path = self.data_dir / "hand5_strength.npy"
        if path.exists():
            arr = np.load(path)
            return arr.astype(np.int32, copy=False)

        # Safe fallback for local tests if artifacts are missing.
        from gym_env import PokerEnv

        env = PokerEnv()
        evaluator = env.evaluator
        arr = np.zeros(HAND5_SIZE, dtype=np.int32)
        for cards in combinations(range(N_CARDS), 5):
            idx = combo_to_index(cards)
            treys_cards = [env.int_to_card(c) for c in cards]
            arr[idx] = evaluator.evaluate(treys_cards[:2], treys_cards[2:])
        return arr

    def _load_preflop_equity(self) -> np.ndarray:
        path = self.data_dir / "preflop_equity.npy"
        if path.exists():
            arr = np.load(path)
            return arr.astype(np.float32, copy=False)

        # Conservative fallback: derive from 5-card absolute rank percentile.
        scores = self.hand5_strength.astype(np.float64)
        inv = (scores.max() - scores) / max(1.0, scores.max() - scores.min())
        return inv.astype(np.float32)

    def _load_flop_seed_table(self) -> Dict[int, float]:
        path = self.data_dir / "flop_seed_table.npy"
        if not path.exists():
            return {}
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            maybe_dict = loaded.item()
            if isinstance(maybe_dict, dict):
                return {int(k): float(v) for k, v in maybe_dict.items()}
        if isinstance(loaded, dict):
            return {int(k): float(v) for k, v in loaded.items()}
        return {}

    def hand5_score(self, cards5: Sequence[int]) -> int:
        return int(self.hand5_strength[combo_to_index(cards5)])

    def evaluate_7card_score(self, hole2: Sequence[int], board5: Sequence[int]) -> int:
        cards7 = [int(hole2[0]), int(hole2[1])] + [int(c) for c in board5]
        best = 10**9
        for subset in SEVEN_TO_FIVE_SUBSETS:
            score = self.hand5_score([cards7[i] for i in subset])
            if score < best:
                best = score
        return best

    def get_preflop_equity(self, cards5: Sequence[int]) -> float:
        if len(cards5) < 5:
            return 0.5
        return float(self.preflop_equity[combo_to_index(cards5[:5])])

    def is_premium_preflop(self, cards5: Sequence[int]) -> bool:
        return self.get_preflop_equity(cards5) >= self._premium_threshold

    def get_flop_ev(self, keep_cards: Sequence[int], flop_cards: Sequence[int]) -> float | None:
        return self.flop_cache.get(pack_flop_key(keep_cards, flop_cards))

    def set_flop_ev(self, keep_cards: Sequence[int], flop_cards: Sequence[int], ev: float) -> None:
        self.flop_cache[pack_flop_key(keep_cards, flop_cards)] = float(ev)

    def get_flop_ev_by_key(self, key: int) -> float | None:
        return self.flop_cache.get(int(key))

    def set_flop_ev_by_key(self, key: int, ev: float) -> None:
        self.flop_cache[int(key)] = float(ev)

    @staticmethod
    def available_cards(excluded: Iterable[int]) -> list[int]:
        excluded_set = {int(c) for c in excluded}
        return [c for c in range(N_CARDS) if c not in excluded_set]
