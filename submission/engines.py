from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np

from submission.lut_store import KEEP_INDEX_PAIRS, LUTStore, N_CARDS, pack_flop_key

_ALL_CARDS = np.arange(N_CARDS, dtype=np.intp)


@dataclass(frozen=True)
class ParsedState:
    street: int
    hand_number: int
    my_cards: tuple[int, ...]
    community_cards: tuple[int, ...]
    my_bet: int
    opp_bet: int
    min_raise: int
    max_raise: int
    pot_size: int
    valid_actions: tuple[int, ...]
    time_used: float
    time_left: float
    opp_last_action: str
    opp_discarded_cards: tuple[int, ...]
    can_discard: bool
    blind_position: int

    @property
    def continue_cost(self) -> int:
        return max(0, self.opp_bet - self.my_bet)


class StateManager:
    def __init__(self, history_size: int = 256) -> None:
        self._opp_action_history = deque(maxlen=history_size)
        self._last_action_token: tuple[int, int, str, int] | None = None

    def parse(
        self, observation: dict[str, Any], info: dict[str, Any] | None
    ) -> ParsedState:
        my_cards = tuple(
            int(c) for c in observation.get("my_cards", []) if int(c) != -1
        )
        community_cards = tuple(
            int(c) for c in observation.get("community_cards", []) if int(c) != -1
        )
        valid_actions = tuple(
            int(v) for v in observation.get("valid_actions", [0, 0, 0, 0, 0])
        )
        opp_discarded = tuple(
            int(c) for c in observation.get("opp_discarded_cards", []) if int(c) != -1
        )
        street = int(observation.get("street", 0))
        hand_number = int((info or {}).get("hand_number", -1))
        opp_last_action = str(observation.get("opp_last_action", "None"))
        opp_bet = int(observation.get("opp_bet", 0))
        token = (hand_number, street, opp_last_action, opp_bet)
        if opp_last_action != "None" and token != self._last_action_token:
            self._opp_action_history.append(token)
            self._last_action_token = token

        can_discard = bool(valid_actions[4]) if len(valid_actions) > 4 else False
        return ParsedState(
            street=street,
            hand_number=hand_number,
            my_cards=my_cards,
            community_cards=community_cards,
            my_bet=int(observation.get("my_bet", 0)),
            opp_bet=opp_bet,
            min_raise=int(observation.get("min_raise", 0)),
            max_raise=int(observation.get("max_raise", 0)),
            pot_size=int(observation.get("pot_size", 0)),
            valid_actions=valid_actions,
            time_used=float(observation.get("time_used", 0.0)),
            time_left=float(observation.get("time_left", 0.0)),
            opp_last_action=opp_last_action,
            opp_discarded_cards=opp_discarded,
            can_discard=can_discard,
            blind_position=int(observation.get("blind_position", 0)),
        )


class TimeSupervisor:
    def __init__(
        self,
        total_hands: int = 1000,
        full_threshold: float = 1.2,
        survival_threshold: float = 0.30,
    ) -> None:
        self.total_hands = total_hands
        self.full_threshold = full_threshold
        self.survival_threshold = survival_threshold

    def select_mode(self, state: ParsedState) -> tuple[str, float]:
        hand_num = state.hand_number if state.hand_number >= 0 else 0
        remaining_hands = max(1, self.total_hands - hand_num)
        tavg = state.time_left / remaining_hands
        if tavg <= self.survival_threshold:
            return "survival", tavg
        if tavg <= self.full_threshold:
            return "fast", tavg
        return "full", tavg


class OpponentModel:
    def __init__(self) -> None:
        self.hands_observed = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.cbet_count = 0
        self.cbet_opportunities = 0
        self.showdown_count = 0
        self.showdown_aggressive_count = 0
        self.fold_count = 0
        self.fold_opportunities = 0

        self._current_hand = -1
        self._hand_vpip = False
        self._hand_pfr = False
        self._hand_opp_preflop_aggressor = False
        self._hand_cbet = False
        self._hand_fold = False
        self._hand_fold_opportunity = False
        self._last_event_token: tuple[int, int, str, int, int] | None = None
        self._finalized_hands: set[int] = set()

    def _ensure_hand(self, hand_number: int) -> None:
        if hand_number == self._current_hand:
            return
        self._current_hand = hand_number
        self._hand_vpip = False
        self._hand_pfr = False
        self._hand_opp_preflop_aggressor = False
        self._hand_cbet = False
        self._hand_fold = False
        self._hand_fold_opportunity = False
        self._last_event_token = None

    def record_state(self, state: ParsedState) -> None:
        self._ensure_hand(state.hand_number)
        action = state.opp_last_action
        if action == "None":
            return

        token = (state.hand_number, state.street, action, state.opp_bet, state.my_bet)
        if token == self._last_event_token:
            return
        self._last_event_token = token

        if state.street == 0 and action in ("CALL", "RAISE"):
            self._hand_vpip = True
        if state.street == 0 and action == "RAISE":
            self._hand_pfr = True
            self._hand_opp_preflop_aggressor = True
        if state.street == 1 and self._hand_opp_preflop_aggressor and action == "RAISE":
            self._hand_cbet = True
        if action == "FOLD":
            self._hand_fold = True
        if action in ("FOLD", "CALL", "RAISE"):
            self._hand_fold_opportunity = True

    def finalize_hand(self, state: ParsedState, info: dict[str, Any] | None) -> None:
        hand_number = state.hand_number
        if hand_number in self._finalized_hands:
            return
        self._finalized_hands.add(hand_number)
        self.hands_observed += 1
        self.vpip_count += int(self._hand_vpip)
        self.pfr_count += int(self._hand_pfr)
        if self._hand_opp_preflop_aggressor:
            self.cbet_opportunities += 1
            self.cbet_count += int(self._hand_cbet)
        if self._hand_fold_opportunity:
            self.fold_opportunities += 1
            self.fold_count += int(self._hand_fold)

        info = info or {}
        if "player_0_cards" in info and "player_1_cards" in info:
            self.showdown_count += 1
            if self._hand_pfr or self._hand_cbet:
                self.showdown_aggressive_count += 1

    def get_fold_rate(self) -> float:
        if self.fold_opportunities < 10:
            return 0.0
        return self.fold_count / self.fold_opportunities

    def get_margin(self, base_margin: float = 0.05) -> float:
        if self.hands_observed < 5:
            return base_margin

        vpip = self.vpip_count / max(1, self.hands_observed)
        pfr = self.pfr_count / max(1, self.hands_observed)
        cbet = (
            self.cbet_count / max(1, self.cbet_opportunities)
            if self.cbet_opportunities
            else 0.0
        )
        fold_rate = self.get_fold_rate()

        margin = base_margin

        if vpip > 0.80 and pfr > 0.45:
            margin -= 0.03
        elif vpip > 0.80 and pfr < 0.25:
            margin += 0.03

        if pfr < 0.15:
            margin += 0.04
        elif pfr < 0.25:
            margin += 0.02

        if cbet > 0.75:
            margin += 0.02

        if fold_rate < 0.15 and self.fold_opportunities >= 20:
            margin += 0.03
        elif fold_rate > 0.50:
            margin -= 0.03

        return max(0.02, min(0.15, margin))


class DiscardEngine:
    def __init__(
        self, luts: LUTStore, full_samples: int = 100, fast_samples: int = 40
    ) -> None:
        self.luts = luts
        self.full_samples = full_samples
        self.fast_samples = fast_samples

    def _sample_size(self, mode: str) -> int:
        if mode == "full":
            return self.full_samples
        if mode == "fast":
            return self.fast_samples
        return max(32, self.fast_samples // 2)

    @staticmethod
    def _heuristic_flop_ev(keep_cards: Sequence[int], flop_cards: Sequence[int]) -> float:
        """
        Cheap deterministic estimate for fast/survival paths.
        Ranks: 0-7 = 2-9, 8 = Ace.  Suits: 0-2 (only 3 suits in 27-card deck).
        """
        cards = [int(c) for c in keep_cards] + [int(c) for c in flop_cards[:3]]
        ranks = [c % 9 for c in cards]
        suits = [c // 9 for c in cards]
        keep_ranks = [c % 9 for c in (int(keep_cards[0]), int(keep_cards[1]))]
        keep_suits = [c // 9 for c in (int(keep_cards[0]), int(keep_cards[1]))]

        rank_counts: dict[int, int] = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        max_same_rank = max(rank_counts.values()) if rank_counts else 1

        suit_counts: dict[int, int] = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_same_suit = max(suit_counts.values()) if suit_counts else 1

        ev = 0.42

        if max_same_rank >= 3:
            ev += 0.20
        elif max_same_rank >= 2:
            n_pairs = sum(1 for v in rank_counts.values() if v >= 2)
            if n_pairs >= 2:
                ev += 0.15
            else:
                ev += 0.10

        if max_same_suit >= 5:
            ev += 0.28
        elif max_same_suit >= 4:
            ev += 0.10
        elif max_same_suit >= 3:
            if keep_suits[0] == keep_suits[1]:
                hero_suit = keep_suits[0]
                if suit_counts.get(hero_suit, 0) >= 3:
                    ev += 0.05

        rank_set = set(ranks)
        straight_windows = [
            (0, 1, 2, 3, 4),
            (1, 2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (3, 4, 5, 6, 7),
            (4, 5, 6, 7, 8),
            (8, 0, 1, 2, 3),
        ]
        best_consec = 0
        for window in straight_windows:
            match = sum(1 for r in window if r in rank_set)
            if match > best_consec:
                best_consec = match
        if best_consec >= 5:
            ev += 0.25
        elif best_consec >= 4:
            ev += 0.06
        elif best_consec >= 3:
            ev += 0.02

        if keep_suits[0] == keep_suits[1]:
            ev += 0.03

        top_keep = max(keep_ranks)
        ev += (top_keep / 8.0) * 0.06

        return max(0.15, min(0.90, ev))

    def _sample_equity(
        self,
        hero_cards: Sequence[int],
        known_board: Sequence[int],
        opponent_range: list[tuple[int, int]] | None,
        samples: int,
        seed: int,
        dead_cards: Sequence[int] = (),
    ) -> float:
        used_hero_board = (
            set(int(c) for c in hero_cards)
            | set(int(c) for c in known_board)
            | set(int(c) for c in dead_cards)
        )

        if opponent_range is None:
            available = [c for c in range(N_CARDS) if c not in used_hero_board]
            candidate_range = list(combinations(available, 2))
        else:
            candidate_range = [
                (int(a), int(b))
                for (a, b) in opponent_range
                if int(a) not in used_hero_board
                and int(b) not in used_hero_board
                and int(a) != int(b)
            ]
        if not candidate_range:
            return 0.5

        board_known = [int(c) for c in known_board]
        need_board = 5 - len(board_known)
        if need_board < 0:
            board_known = board_known[:5]
            need_board = 0

        if need_board == 0 and len(candidate_range) <= 200:
            wins = ties = 0.0
            my_score = self.luts.evaluate_7card_score(hero_cards, board_known)
            for opp_cards in candidate_range:
                opp_score = self.luts.evaluate_7card_score(opp_cards, board_known)
                if my_score < opp_score:
                    wins += 1.0
                elif my_score == opp_score:
                    ties += 1.0
            return (wins + 0.5 * ties) / len(candidate_range)

        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        base_remaining = np.array(
            [c for c in range(N_CARDS) if c not in used_hero_board], dtype=np.intp
        )
        range_arr = np.array(candidate_range, dtype=np.intp)
        n_range = len(candidate_range)

        wins = 0.0
        ties = 0.0
        valid = 0
        for _ in range(samples):
            idx = rng.integers(n_range)
            opp_c0 = range_arr[idx, 0]
            opp_c1 = range_arr[idx, 1]
            mask = (base_remaining != opp_c0) & (base_remaining != opp_c1)
            remaining = base_remaining[mask]
            if need_board > len(remaining):
                continue
            if need_board > 0:
                board_tail = rng.choice(remaining, size=need_board, replace=False)
                board5 = board_known + board_tail.tolist()
            else:
                board5 = board_known

            my_score = self.luts.evaluate_7card_score(hero_cards, board5)
            opp_score = self.luts.evaluate_7card_score((opp_c0, opp_c1), board5)
            if my_score < opp_score:
                wins += 1.0
            elif my_score == opp_score:
                ties += 1.0
            valid += 1

        return (wins + 0.5 * ties) / valid if valid > 0 else 0.5

    def get_or_estimate_flop_ev(
        self,
        keep_cards: Sequence[int],
        flop_cards: Sequence[int],
        mode: str,
    ) -> float:
        key = pack_flop_key(keep_cards, flop_cards)
        cached = self.luts.get_flop_ev_by_key(key)
        if cached is not None:
            return cached
        if mode != "full":
            ev = self._heuristic_flop_ev(keep_cards, flop_cards)
            self.luts.set_flop_ev_by_key(key, ev)
            return ev
        discard_samples = min(self._sample_size(mode), 300)
        ev = self._sample_equity(
            hero_cards=keep_cards,
            known_board=flop_cards,
            opponent_range=None,
            samples=discard_samples,
            seed=0xA5A5A5 ^ key,
        )
        self.luts.set_flop_ev_by_key(key, ev)
        return ev

    def choose_discard(
        self, state: ParsedState, mode: str
    ) -> tuple[tuple[int, int, int, int], float]:
        cards = state.my_cards
        if len(cards) != 5 or len(state.community_cards) < 3:
            return (4, 0, 0, 1), 0.5

        flop = state.community_cards[:3]
        best_pair = (0, 1)
        best_ev = -1.0
        for i, j in KEEP_INDEX_PAIRS:
            keep_cards = (cards[i], cards[j])
            ev = self.get_or_estimate_flop_ev(keep_cards, flop, mode)
            if ev > best_ev:
                best_ev = ev
                best_pair = (i, j)
        return (4, 0, best_pair[0], best_pair[1]), best_ev

    def opponent_lower_bound(
        self, opp_discarded: Sequence[int], flop_cards: Sequence[int], mode: str
    ) -> float | None:
        if len(opp_discarded) != 3 or len(flop_cards) < 3:
            return None
        flop = flop_cards[:3]
        best = -1.0
        for pair in combinations(opp_discarded, 2):
            ev = self.get_or_estimate_flop_ev(pair, flop, mode)
            if ev > best:
                best = ev
        return best if best >= 0.0 else None

    def _quick_flop_ev(
        self, keep_cards: Sequence[int], flop_cards: Sequence[int]
    ) -> float:
        key = pack_flop_key(keep_cards, flop_cards)
        cached = self.luts.get_flop_ev_by_key(key)
        if cached is not None:
            return cached
        ev = self._heuristic_flop_ev(keep_cards, flop_cards)
        self.luts.set_flop_ev_by_key(key, ev)
        return ev

    def narrowed_opponent_range(
        self, state: ParsedState, mode: str
    ) -> list[tuple[int, int]] | None:
        # Range narrowing is expensive and not worth it on early streets.
        if mode != "full" or state.street < 2 or len(state.community_cards) < 3:
            return None
        known = (
            set(state.my_cards)
            | set(state.community_cards)
            | set(state.opp_discarded_cards)
        )
        candidates = [
            (a, b)
            for (a, b) in combinations(range(N_CARDS), 2)
            if a not in known and b not in known
        ]
        if not candidates:
            return None

        lower_bound = self.opponent_lower_bound(
            state.opp_discarded_cards, state.community_cards[:3], mode
        )
        if lower_bound is None:
            return candidates

        filtered: list[tuple[int, int]] = []
        flop = state.community_cards[:3]
        for combo in candidates:
            if self._quick_flop_ev(combo, flop) >= lower_bound:
                filtered.append(combo)
        return filtered if filtered else candidates

    def estimate_hero_equity(
        self,
        hero_cards: Sequence[int],
        board_cards: Sequence[int],
        mode: str,
        opponent_range: list[tuple[int, int]] | None = None,
        dead_cards: Sequence[int] = (),
    ) -> float:
        board = list(board_cards[:5])
        seed = 0x5AA55A ^ pack_flop_key(
            hero_cards, board[:3] if len(board) >= 3 else [0, 1, 2]
        )
        return self._sample_equity(
            hero_cards=hero_cards,
            known_board=board,
            opponent_range=opponent_range,
            samples=self._sample_size(mode),
            seed=seed,
            dead_cards=dead_cards,
        )


class DecisionEngine:
    def __init__(self, action_types: Any) -> None:
        self.action_types = action_types
        self._seed = random.randint(0, 0xFFFFFFFF)
        self._rng = random.Random(self._seed)

    def _is_valid(self, state: ParsedState, action_idx: int) -> bool:
        return 0 <= action_idx < len(state.valid_actions) and bool(
            state.valid_actions[action_idx]
        )

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        if hi < lo:
            return lo
        return max(lo, min(hi, value))

    def _jitter(self) -> float:
        return self._rng.uniform(-0.03, 0.03)

    @staticmethod
    def _preflop_tightness(
        my_bet: int, pot_odds: float, margin: float, is_premium: bool
    ) -> tuple[float, float]:
        if is_premium:
            return (pot_odds + margin, max(0.75, pot_odds + 0.30))
        if my_bet <= 2:
            return (pot_odds + margin, max(0.75, pot_odds + 0.30))
        if my_bet <= 8:
            return (0.58, 0.78)
        if my_bet <= 16:
            return (0.65, 0.82)
        return (0.72, 0.88)

    def _raise_amount(self, state: ParsedState, edge: float, mode: str) -> int:
        if state.max_raise <= 0:
            return 0
        noise = self._rng.uniform(0.80, 1.25)
        if state.street == 0:
            if edge > 0.15:
                raw = 6
            elif edge > 0.05:
                raw = 4
            else:
                raw = state.min_raise
            raw = min(raw, 6)
        else:
            if edge > 0.30:
                factor = 0.65
            elif edge > 0.15:
                factor = 0.50
            else:
                factor = 0.35
            raw = int(round(max(2, state.pot_size) * factor))
        raw = int(round(raw * noise))
        return self._clamp(raw, state.min_raise, state.max_raise)

    def _should_bluff(self, opp_fold_rate: float) -> bool:
        if opp_fold_rate < 0.25:
            return False
        if opp_fold_rate > 0.50:
            base_freq = 0.15
        elif opp_fold_rate > 0.35:
            base_freq = 0.10
        else:
            base_freq = 0.06
        return self._rng.random() < base_freq

    def decide(
        self,
        state: ParsedState,
        my_ev: float,
        margin: float,
        mode: str,
        opp_fold_rate: float = 0.0,
        is_premium: bool = False,
    ) -> tuple[int, int, int, int]:
        fold = self.action_types.FOLD.value
        raise_action = self.action_types.RAISE.value
        check = self.action_types.CHECK.value
        call = self.action_types.CALL.value

        continue_cost = state.continue_cost
        pot = max(1, state.pot_size)
        pot_odds = (continue_cost / (pot + continue_cost)) if continue_cost > 0 else 0.0
        jitter = self._jitter()

        street_margin = margin
        if state.street >= 3:
            street_margin = max(margin, 0.12)
        elif state.street >= 2:
            street_margin = max(margin, 0.08)

        # --- No continue cost: check or bet ---
        if continue_cost == 0:
            if state.street >= 1:
                value_threshold = 0.58 if opp_fold_rate >= 0.25 else 0.63
            else:
                value_threshold = 0.65 if opp_fold_rate >= 0.25 else 0.70
            value_threshold += jitter

            if my_ev > value_threshold and self._is_valid(state, raise_action):
                if self._rng.random() < 0.12 and self._is_valid(state, check):
                    return (check, 0, 0, 0)
                return (
                    raise_action,
                    self._raise_amount(state, my_ev - 0.5, mode),
                    0,
                    0,
                )
            if (
                0.45 <= my_ev <= 0.65
                and self._is_valid(state, raise_action)
                and self._rng.random() < 0.08
            ):
                return (
                    raise_action,
                    self._clamp(
                        int(round(pot * 0.40)),
                        state.min_raise,
                        state.max_raise,
                    ),
                    0,
                    0,
                )
            if (
                my_ev > 0.35
                and self._is_valid(state, raise_action)
                and self._should_bluff(opp_fold_rate)
            ):
                return (
                    raise_action,
                    self._clamp(
                        int(round(pot * 0.40)),
                        state.min_raise,
                        state.max_raise,
                    ),
                    0,
                    0,
                )
            if self._is_valid(state, check):
                return (check, 0, 0, 0)
            if self._is_valid(state, call):
                return (call, 0, 0, 0)
            if self._is_valid(state, raise_action):
                return (
                    raise_action,
                    self._clamp(state.min_raise, state.min_raise, state.max_raise),
                    0,
                    0,
                )
            return (fold, 0, 0, 0)

        # --- Facing a bet: preflop escalation tiers ---
        if state.street == 0:
            call_thresh, raise_thresh = self._preflop_tightness(
                state.my_bet, pot_odds, street_margin, is_premium
            )
            call_thresh += jitter
            raise_thresh += jitter

            if state.my_bet >= 20 and not is_premium:
                if my_ev >= call_thresh and self._is_valid(state, call):
                    return (call, 0, 0, 0)
                if self._is_valid(state, fold):
                    return (fold, 0, 0, 0)
                if self._is_valid(state, check):
                    return (check, 0, 0, 0)
                return (call if self._is_valid(state, call) else fold, 0, 0, 0)

            if my_ev > raise_thresh and self._is_valid(state, raise_action):
                return (raise_action, self._raise_amount(state, my_ev - 0.5, mode), 0, 0)
            if my_ev >= call_thresh and self._is_valid(state, call):
                return (call, 0, 0, 0)
            if self._is_valid(state, fold):
                return (fold, 0, 0, 0)
            if self._is_valid(state, check):
                return (check, 0, 0, 0)
            return (call if self._is_valid(state, call) else fold, 0, 0, 0)

        # --- Facing a bet: post-flop ---
        raise_threshold = max(0.75, pot_odds + 0.30) + jitter
        call_threshold = pot_odds + street_margin + jitter

        if my_ev > raise_threshold and self._is_valid(state, raise_action):
            return (raise_action, self._raise_amount(state, my_ev - 0.5, mode), 0, 0)
        if my_ev >= call_threshold and self._is_valid(state, call):
            return (call, 0, 0, 0)
        if self._is_valid(state, fold):
            return (fold, 0, 0, 0)
        if self._is_valid(state, check):
            return (check, 0, 0, 0)
        return (call if self._is_valid(state, call) else fold, 0, 0, 0)
