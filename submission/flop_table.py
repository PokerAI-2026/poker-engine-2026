from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Sequence

import numpy as np

from submission.lut_store import KEEP_INDEX_PAIRS

N_RANKS = 9
N_SUITS = 3
SUIT_PERMUTATIONS = tuple(permutations(range(N_SUITS), N_SUITS))


def card_rank(card_id: int) -> int:
    return int(card_id) % N_RANKS


def card_suit(card_id: int) -> int:
    return int(card_id) // N_RANKS


def apply_suit_permutation(card_id: int, suit_perm: Sequence[int]) -> int:
    """
    Applies a global suit remapping to a card id.
    """
    rank = card_rank(card_id)
    suit = card_suit(card_id)
    return int(suit_perm[suit]) * N_RANKS + rank


@dataclass(frozen=True)
class CanonicalizedState:
    canonical_hole: tuple[int, int, int, int, int]
    canonical_flop: tuple[int, int, int]
    original_to_canonical_hole_pos: tuple[int, int, int, int, int]
    canonical_to_original_hole_pos: tuple[int, int, int, int, int]


def canonicalize_flop_state(my5: Sequence[int], flop3: Sequence[int]) -> CanonicalizedState:
    if len(my5) != 5:
        raise ValueError(f"Expected 5 hole cards, got {len(my5)}")
    if len(flop3) != 3:
        raise ValueError(f"Expected 3 flop cards, got {len(flop3)}")

    best_key: tuple[int, ...] | None = None
    best_state: CanonicalizedState | None = None

    for suit_perm in SUIT_PERMUTATIONS:
        remapped_hole = [
            (apply_suit_permutation(int(card), suit_perm), original_idx)
            for original_idx, card in enumerate(my5)
        ]
        remapped_hole.sort(key=lambda x: (x[0], x[1]))

        canonical_hole = tuple(int(card) for (card, _) in remapped_hole)
        canonical_to_original = tuple(int(original_idx) for (_, original_idx) in remapped_hole)
        original_to_canonical = [0] * 5
        for canonical_pos, original_pos in enumerate(canonical_to_original):
            original_to_canonical[original_pos] = canonical_pos

        canonical_flop = tuple(
            sorted(apply_suit_permutation(int(card), suit_perm) for card in flop3)
        )

        key = canonical_hole + canonical_flop
        if best_key is None or key < best_key:
            best_key = key
            best_state = CanonicalizedState(
                canonical_hole=canonical_hole,
                canonical_flop=canonical_flop,
                original_to_canonical_hole_pos=tuple(original_to_canonical),
                canonical_to_original_hole_pos=canonical_to_original,
            )

    if best_state is None:
        raise RuntimeError("Failed to canonicalize flop state")
    return best_state


def state_key(my5: Sequence[int], flop3: Sequence[int]) -> tuple[int, ...]:
    canon = canonicalize_flop_state(my5, flop3)
    return canon.canonical_hole + canon.canonical_flop


def quantize_u16(value: float) -> int:
    clipped = min(1.0, max(0.0, float(value)))
    return int(round(clipped * 65535.0))


def dequantize_u16(value: int) -> float:
    return int(value) / 65535.0


class FlopDiscardTable:
    RECORD_DTYPE = np.dtype(
        [
            ("best_keep_idx", np.uint8),
            ("second_keep_idx", np.uint8),
            ("best_equity_q", "<u2"),
            ("gap_q", "<u2"),
        ]
    )

    def __init__(
        self,
        table_path: str | Path,
        index_path: str | Path,
        meta_path: str | Path,
    ) -> None:
        self.table_path = Path(table_path)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        self.meta = self._load_meta()
        self.index = self._load_index()
        self.table = np.memmap(self.table_path, mode="r", dtype=self.RECORD_DTYPE)

    @classmethod
    def from_data_dir(
        cls, data_dir: str | Path, strict: bool = False
    ) -> FlopDiscardTable | None:
        base = Path(data_dir)
        table_path = base / "bb_table.bin"
        index_path = base / "bb_index.npy"
        meta_path = base / "bb_meta.json"

        if not (table_path.exists() and index_path.exists() and meta_path.exists()):
            if strict:
                raise FileNotFoundError(
                    "Missing one or more BB table files: bb_table.bin, bb_index.npy, bb_meta.json"
                )
            return None

        try:
            return cls(table_path, index_path, meta_path)
        except Exception:
            if strict:
                raise
            return None

    def _load_meta(self) -> dict[str, object]:
        with self.meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("bb_meta.json must contain an object")
        return data

    def _load_index(self) -> dict[tuple[int, ...], int]:
        loaded = np.load(self.index_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            maybe_dict = loaded.item()
        else:
            maybe_dict = loaded

        if not isinstance(maybe_dict, dict):
            raise ValueError("bb_index.npy must contain a dictionary")

        out: dict[tuple[int, ...], int] = {}
        for key, value in maybe_dict.items():
            if isinstance(key, str):
                parts = tuple(int(piece) for piece in key.split(","))
            else:
                parts = tuple(int(x) for x in key)
            if len(parts) != 8:
                raise ValueError(f"Invalid canonical key length: {len(parts)} (expected 8)")
            out[parts] = int(value)
        return out

    def lookup(self, my5: Sequence[int], flop3: Sequence[int]) -> tuple[np.void, CanonicalizedState] | None:
        if len(my5) != 5 or len(flop3) != 3:
            return None

        canon = canonicalize_flop_state(my5, flop3)
        key = canon.canonical_hole + canon.canonical_flop
        state_id = self.index.get(key)
        if state_id is None or state_id < 0 or state_id >= len(self.table):
            return None
        return self.table[state_id], canon

    def choose_keep_positions(
        self, my5: Sequence[int], flop3: Sequence[int]
    ) -> tuple[int, int, float] | None:
        looked_up = self.lookup(my5, flop3)
        if looked_up is None:
            return None
        record, canon = looked_up

        keep_idx = int(record["best_keep_idx"])
        if keep_idx < 0 or keep_idx >= len(KEEP_INDEX_PAIRS):
            return None
        c0, c1 = KEEP_INDEX_PAIRS[keep_idx]
        orig0 = int(canon.canonical_to_original_hole_pos[c0])
        orig1 = int(canon.canonical_to_original_hole_pos[c1])
        best_equity = dequantize_u16(int(record["best_equity_q"]))
        return (orig0, orig1, best_equity)

    def choose_keep(self, my5: Sequence[int], flop3: Sequence[int]) -> tuple[int, int] | None:
        chosen = self.choose_keep_positions(my5, flop3)
        if chosen is None:
            return None
        keep0_pos, keep1_pos, _ = chosen
        return (int(my5[keep0_pos]), int(my5[keep1_pos]))

    def choose_discard(self, my5: Sequence[int], flop3: Sequence[int]) -> tuple[int, int, int] | None:
        chosen = self.choose_keep_positions(my5, flop3)
        if chosen is None:
            return None
        keep_positions = {chosen[0], chosen[1]}
        discards = tuple(int(my5[i]) for i in range(5) if i not in keep_positions)
        if len(discards) != 3:
            return None
        return discards
