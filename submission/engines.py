from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

from submission.lut_store import KEEP_INDEX_PAIRS, LUTStore, N_CARDS, pack_flop_key


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


@dataclass(frozen=True)
class OpponentStats:
    hands_observed: int
    vpip_rate: float
    pfr_rate: float
    cbet_rate: float
    preflop_raise_rate: float
    fold_to_3bet_rate: float
    river_raise_rate: float


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
        full_threshold: float = 0.8,
        survival_threshold: float = 0.20,
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
        self.preflop_raise_count = 0
        self.preflop_raise_opportunities = 0
        self.fold_to_3bet_count = 0
        self.fold_to_3bet_opportunities = 0
        self.river_raise_count = 0
        self.river_raise_opportunities = 0
        self.showdown_count = 0
        self.showdown_aggressive_count = 0

        self._current_hand = -1
        self._hand_vpip = False
        self._hand_pfr = False
        self._hand_opp_preflop_aggressor = False
        self._hand_cbet = False
        self._hand_hero_3bet = False
        self._hand_opp_folded_to_3bet = False
        self._hand_saw_river_action = False
        self._hand_river_raise = False
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
        self._hand_hero_3bet = False
        self._hand_opp_folded_to_3bet = False
        self._hand_saw_river_action = False
        self._hand_river_raise = False
        self._last_event_token = None

    def record_hero_action(
        self, state: ParsedState, action: tuple[int, int, int, int], raise_idx: int
    ) -> None:
        self._ensure_hand(state.hand_number)
        action_type = int(action[0])
        if state.street != 0 or action_type != raise_idx:
            return
        # Hero raising while facing an existing preflop bet is our 3-bet marker.
        if state.opp_bet > state.my_bet:
            self._hand_hero_3bet = True

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
        if state.street == 0 and action == "FOLD" and self._hand_hero_3bet:
            self._hand_opp_folded_to_3bet = True
        if state.street == 3:
            self._hand_saw_river_action = True
            if action == "RAISE":
                self._hand_river_raise = True

    def finalize_hand(self, state: ParsedState, info: dict[str, Any] | None) -> None:
        hand_number = state.hand_number
        if hand_number in self._finalized_hands:
            return
        self._finalized_hands.add(hand_number)
        self.hands_observed += 1
        self.vpip_count += int(self._hand_vpip)
        self.pfr_count += int(self._hand_pfr)
        self.preflop_raise_opportunities += 1
        self.preflop_raise_count += int(self._hand_pfr)
        if self._hand_opp_preflop_aggressor:
            self.cbet_opportunities += 1
            self.cbet_count += int(self._hand_cbet)
        if self._hand_hero_3bet:
            self.fold_to_3bet_opportunities += 1
            self.fold_to_3bet_count += int(self._hand_opp_folded_to_3bet)
        if self._hand_saw_river_action:
            self.river_raise_opportunities += 1
            self.river_raise_count += int(self._hand_river_raise)

        info = info or {}
        if "player_0_cards" in info and "player_1_cards" in info:
            self.showdown_count += 1
            if self._hand_pfr or self._hand_cbet:
                self.showdown_aggressive_count += 1

    def get_stats(self) -> OpponentStats:
        hands = max(1, self.hands_observed)
        cbet_rate = (
            self.cbet_count / max(1, self.cbet_opportunities)
            if self.cbet_opportunities
            else 0.0
        )
        return OpponentStats(
            hands_observed=self.hands_observed,
            vpip_rate=self.vpip_count / hands,
            pfr_rate=self.pfr_count / hands,
            cbet_rate=cbet_rate,
            preflop_raise_rate=self.preflop_raise_count
            / max(1, self.preflop_raise_opportunities),
            fold_to_3bet_rate=self.fold_to_3bet_count
            / max(1, self.fold_to_3bet_opportunities),
            river_raise_rate=self.river_raise_count
            / max(1, self.river_raise_opportunities),
        )

    def get_margin(self, base_margin: float = 0.05) -> float:
        if self.hands_observed == 0:
            return base_margin

        stats = self.get_stats()

        margin = base_margin
        if stats.vpip_rate > 0.80 and stats.pfr_rate > 0.45:
            margin -= 0.02
        if stats.pfr_rate < 0.20:
            margin += 0.03
        if stats.cbet_rate > 0.70:
            margin += 0.01
        return max(0.02, min(0.15, margin))


class DiscardEngine:
    def __init__(
        self, luts: LUTStore, full_samples: int = 500, fast_samples: int = 250
    ) -> None:
        self.luts = luts
        self.full_samples = full_samples
        self.fast_samples = fast_samples

    def _sample_size(self, mode: str) -> int:
        if mode == "full":
            return self.full_samples
        if mode == "fast":
            return self.fast_samples
        return max(120, self.fast_samples // 2)

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

        # Exact compare when no future board cards remain and range is moderate.
        if need_board == 0 and len(candidate_range) <= 200:
            wins = ties = 0.0
            for opp_cards in candidate_range:
                my_score = self.luts.evaluate_7card_score(hero_cards, board_known)
                opp_score = self.luts.evaluate_7card_score(opp_cards, board_known)
                if my_score < opp_score:
                    wins += 1.0
                elif my_score == opp_score:
                    ties += 1.0
            return (wins + 0.5 * ties) / len(candidate_range)

        rng = random.Random(seed)
        wins = 0.0
        ties = 0.0
        valid = 0
        for _ in range(samples):
            opp_cards = candidate_range[rng.randrange(len(candidate_range))]
            blocked = used_hero_board | {opp_cards[0], opp_cards[1]}
            remaining = [c for c in range(N_CARDS) if c not in blocked]
            if need_board > len(remaining):
                continue
            board_tail = rng.sample(remaining, need_board) if need_board > 0 else []
            board5 = board_known + board_tail

            my_score = self.luts.evaluate_7card_score(hero_cards, board5)
            opp_score = self.luts.evaluate_7card_score(opp_cards, board5)
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
        ev = self._sample_equity(
            hero_cards=keep_cards,
            known_board=flop_cards,
            opponent_range=None,
            samples=80,
            seed=0xA5A5A5 ^ key,
        )
        self.luts.set_flop_ev_by_key(key, ev)
        return ev

    def narrowed_opponent_range(
        self, state: ParsedState, mode: str
    ) -> list[tuple[int, int]] | None:
        if len(state.community_cards) < 3:
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
    def __init__(
        self,
        action_types: Any,
        *,
        preflop_policy_v2: bool = True,
        adaptive_model_v2: bool = True,
        street_margin_v2: bool = True,
        rng_seed: int = 2026,
    ) -> None:
        self.action_types = action_types
        self.preflop_policy_v2 = preflop_policy_v2
        self.adaptive_model_v2 = adaptive_model_v2
        self.street_margin_v2 = street_margin_v2
        self._rng = random.Random(rng_seed)

    def _is_valid(self, state: ParsedState, action_idx: int) -> bool:
        return 0 <= action_idx < len(state.valid_actions) and bool(
            state.valid_actions[action_idx]
        )

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        if hi < lo:
            return lo
        return max(lo, min(hi, value))

    def _raise_amount(
        self,
        state: ParsedState,
        edge: float,
        mode: str,
        *,
        preflop_aggressive: bool = False,
    ) -> int:
        if state.max_raise <= 0:
            return 0
        if state.street == 0:
            factor = 0.80 if preflop_aggressive else 0.60
        elif mode == "full":
            factor = 0.48
        elif mode == "fast":
            factor = 0.38
        else:
            factor = 0.30
        if edge > 0.35:
            factor += 0.15
        elif edge > 0.20:
            factor += 0.05
        raw = int(round(max(2, state.pot_size) * factor))
        return self._clamp(raw, state.min_raise, state.max_raise)

    @staticmethod
    def _cost_bucket(continue_cost: int, pot: int) -> str:
        ratio = continue_cost / max(1, pot)
        if ratio <= 0.35:
            return "small"
        if ratio <= 0.75:
            return "medium"
        return "large"

    def _roll(self, probability: float) -> bool:
        return self._rng.random() < max(0.0, min(1.0, probability))

    def _street_margin(
        self,
        state: ParsedState,
        base_margin: float,
        stats: OpponentStats | None,
    ) -> float:
        if not self.street_margin_v2:
            return base_margin

        if state.street == 0:
            margin = base_margin - 0.012
        elif state.street == 1:
            margin = base_margin
        elif state.street == 2:
            margin = base_margin + 0.004
        else:
            margin = base_margin + 0.009

        if self.adaptive_model_v2 and stats is not None and stats.hands_observed >= 40:
            if stats.preflop_raise_rate > 0.62 and state.street == 0:
                margin -= 0.010
            if stats.cbet_rate > 0.68 and state.street == 1:
                margin += 0.008
            if stats.river_raise_rate > 0.32 and state.street == 3:
                margin += 0.010

        return max(0.02, min(0.18, margin))

    def _preflop_decide(
        self,
        state: ParsedState,
        my_ev: float,
        margin: float,
        mode: str,
        stats: OpponentStats | None,
    ) -> tuple[int, int, int, int] | None:
        if state.street != 0 or not self.preflop_policy_v2:
            return None

        fold = self.action_types.FOLD.value
        raise_action = self.action_types.RAISE.value
        check = self.action_types.CHECK.value
        call = self.action_types.CALL.value

        continue_cost = state.continue_cost
        pot = max(1, state.pot_size)
        can_raise = self._is_valid(state, raise_action)
        can_call = self._is_valid(state, call)
        can_check = self._is_valid(state, check)
        can_fold = self._is_valid(state, fold)

        # Blind position in env observations: 1 -> big blind, 0 -> small blind.
        is_big_blind = state.blind_position == 1
        preflop_raise_rate = stats.preflop_raise_rate if stats is not None else 0.5
        fold_to_3bet = stats.fold_to_3bet_rate if stats is not None else 0.35

        if continue_cost == 0:
            open_threshold = 0.54 if is_big_blind else 0.56
            if self.adaptive_model_v2 and preflop_raise_rate > 0.60:
                open_threshold -= 0.02

            if can_raise and my_ev >= open_threshold:
                if my_ev >= 0.76:
                    raise_freq = 1.0
                elif my_ev >= 0.68:
                    raise_freq = 0.80
                elif my_ev >= 0.60:
                    raise_freq = 0.58
                else:
                    raise_freq = 0.34

                if self._roll(raise_freq):
                    return (
                        raise_action,
                        self._raise_amount(
                            state, my_ev - 0.5, mode, preflop_aggressive=(my_ev >= 0.70)
                        ),
                        0,
                        0,
                    )
            if can_check:
                return (check, 0, 0, 0)
            if can_call:
                return (call, 0, 0, 0)
            if can_fold:
                return (fold, 0, 0, 0)
            return None

        pot_odds = continue_cost / (pot + continue_cost)
        cost_bucket = self._cost_bucket(continue_cost, pot)
        defend_floor = {
            True: {"small": 0.45, "medium": 0.35, "large": 0.24},
            False: {"small": 0.34, "medium": 0.26, "large": 0.18},
        }[is_big_blind][cost_bucket]
        if self.adaptive_model_v2:
            if preflop_raise_rate > 0.62:
                defend_floor += 0.08
            if preflop_raise_rate > 0.72:
                defend_floor += 0.05
        defend_floor = min(0.72, defend_floor)

        value_3bet = can_raise and my_ev >= (0.73 if cost_bucket == "large" else 0.70)
        bluff_3bet_freq = 0.02
        if self.adaptive_model_v2:
            if fold_to_3bet > 0.52:
                bluff_3bet_freq += 0.06
            if preflop_raise_rate > 0.60:
                bluff_3bet_freq += 0.03

        if can_raise and (value_3bet or (my_ev >= 0.56 and self._roll(bluff_3bet_freq))):
            return (
                raise_action,
                self._raise_amount(
                    state,
                    my_ev - 0.5,
                    mode,
                    preflop_aggressive=True,
                ),
                0,
                0,
            )

        call_edge = pot_odds + margin
        call_ok = my_ev >= call_edge
        # Defend floor: allow marginal continues with bounded randomization.
        if not call_ok and my_ev >= max(0.36, pot_odds - 0.10):
            call_ok = self._roll(defend_floor * 0.50)

        if call_ok and can_call:
            return (call, 0, 0, 0)
        if can_fold:
            return (fold, 0, 0, 0)
        if can_check:
            return (check, 0, 0, 0)
        if can_call:
            return (call, 0, 0, 0)
        return None

    def decide(
        self,
        state: ParsedState,
        my_ev: float,
        base_margin: float,
        mode: str,
        opponent_stats: OpponentStats | None = None,
    ) -> tuple[int, int, int, int]:
        fold = self.action_types.FOLD.value
        raise_action = self.action_types.RAISE.value
        check = self.action_types.CHECK.value
        call = self.action_types.CALL.value

        margin = self._street_margin(state, base_margin, opponent_stats)

        preflop_action = self._preflop_decide(
            state, my_ev, margin, mode, opponent_stats
        )
        if preflop_action is not None:
            return preflop_action

        continue_cost = state.continue_cost
        pot = max(1, state.pot_size)
        pot_odds = (continue_cost / (pot + continue_cost)) if continue_cost > 0 else 0.0

        if continue_cost == 0:
            if my_ev > 0.65 and self._is_valid(state, raise_action):
                return (
                    raise_action,
                    self._raise_amount(state, my_ev - 0.5, mode),
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

        bet_ratio = min(2.0, continue_cost / pot)
        discount = max(0.65, 1.0 - 0.25 * bet_ratio)
        effective_ev = my_ev * discount

        if my_ev > 0.79 and self._is_valid(state, raise_action):
            return (raise_action, self._raise_amount(state, my_ev - 0.5, mode), 0, 0)
        if effective_ev >= (pot_odds + margin) and self._is_valid(state, call):
            return (call, 0, 0, 0)
        if self._is_valid(state, fold):
            return (fold, 0, 0, 0)
        if self._is_valid(state, check):
            return (check, 0, 0, 0)
        return (call if self._is_valid(state, call) else fold, 0, 0, 0)
