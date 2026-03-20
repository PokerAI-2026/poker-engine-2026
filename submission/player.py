from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

from agents.agent import Agent, Observation
from gym_env import PokerEnv
from submission.engines import (
    DecisionEngine,
    DiscardEngine,
    OpponentModel,
    StateManager,
    TimeSupervisor,
)
from submission.lut_store import LUTStore


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True) -> None:
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.base_margin = 0.05

        data_dir = Path(__file__).resolve().parent / "data"
        self.luts = LUTStore(data_dir)
        self.state_manager = StateManager()
        self.time_supervisor = TimeSupervisor(total_hands=1000)
        self.opponent_model = OpponentModel()
        self.discard_engine = DiscardEngine(self.luts)
        self.decision_engine = DecisionEngine(self.action_types)

        player_id = os.getenv("PLAYER_ID", "0")
        self.player_id = int(player_id) if player_id.isdigit() else 0
        self.current_mode = "full"

    def __name__(self) -> str:
        return "PlayerAgent"

    def _is_valid(self, valid_actions: tuple[int, ...], action_idx: int) -> bool:
        return 0 <= action_idx < len(valid_actions) and bool(valid_actions[action_idx])

    def _safe_action(self, valid_actions: tuple[int, ...]) -> Tuple[int, int, int, int]:
        fold = self.action_types.FOLD.value
        raise_action = self.action_types.RAISE.value
        check = self.action_types.CHECK.value
        call = self.action_types.CALL.value
        discard = self.action_types.DISCARD.value
        if self._is_valid(valid_actions, check):
            return (check, 0, 0, 0)
        if self._is_valid(valid_actions, call):
            return (call, 0, 0, 0)
        if self._is_valid(valid_actions, fold):
            return (fold, 0, 0, 0)
        if self._is_valid(valid_actions, raise_action):
            return (raise_action, 1, 0, 0)
        if self._is_valid(valid_actions, discard):
            return (discard, 0, 0, 1)
        return (fold, 0, 0, 0)

    def _validated_action(
        self,
        action: Tuple[int, int, int, int],
        state_valid_actions: tuple[int, ...],
        min_raise: int,
        max_raise: int,
    ) -> Tuple[int, int, int, int]:
        action_type, raise_amount, keep_1, keep_2 = (int(x) for x in action)
        if not self._is_valid(state_valid_actions, action_type):
            return self._safe_action(state_valid_actions)

        if action_type == self.action_types.RAISE.value:
            if max_raise < min_raise:
                return self._safe_action(state_valid_actions)
            raise_amount = max(min_raise, min(max_raise, raise_amount))
            return (action_type, raise_amount, 0, 0)

        if action_type == self.action_types.DISCARD.value:
            if keep_1 == keep_2 or not (0 <= keep_1 <= 4) or not (0 <= keep_2 <= 4):
                return (action_type, 0, 0, 1)
            return (action_type, 0, keep_1, keep_2)

        return (action_type, 0, 0, 0)

    def _survival_action(self, state) -> Tuple[int, int, int, int]:
        check = self.action_types.CHECK.value
        call = self.action_types.CALL.value
        fold = self.action_types.FOLD.value
        raise_action = self.action_types.RAISE.value

        if state.street == 0 and len(state.my_cards) >= 5:
            premium = self.luts.is_premium_preflop(state.my_cards[:5])
            if premium:
                if state.continue_cost > 0 and self._is_valid(
                    state.valid_actions, call
                ):
                    return (call, 0, 0, 0)
                if self._is_valid(state.valid_actions, check):
                    return (check, 0, 0, 0)
                if self._is_valid(state.valid_actions, raise_action):
                    return (raise_action, state.min_raise, 0, 0)
            else:
                if self._is_valid(state.valid_actions, check):
                    return (check, 0, 0, 0)
                if self._is_valid(state.valid_actions, fold):
                    return (fold, 0, 0, 0)
                if (
                    self._is_valid(state.valid_actions, call)
                    and state.continue_cost == 0
                ):
                    return (call, 0, 0, 0)
        else:
            if self._is_valid(state.valid_actions, check):
                return (check, 0, 0, 0)
            if self._is_valid(state.valid_actions, fold):
                return (fold, 0, 0, 0)
        return self._safe_action(state.valid_actions)

    def _preflop_aggression_action(self, state) -> Tuple[int, int, int, int] | None:
        if state.street != 0 or len(state.my_cards) < 5:
            return None
        if not self.luts.is_aggressive_preflop(state.my_cards[:5]):
            return None

        raise_action = self.action_types.RAISE.value
        call = self.action_types.CALL.value
        check = self.action_types.CHECK.value

        # Force initiative mainly in open spots; avoid auto 3-bet wars.
        if state.continue_cost == 0 and self._is_valid(state.valid_actions, raise_action):
            target = max(state.min_raise, min(state.max_raise, max(2, state.min_raise)))
            return (raise_action, target, 0, 0)
        if (
            state.continue_cost > 0
            and self.luts.is_premium_preflop(state.my_cards[:5])
            and state.continue_cost <= max(2, state.pot_size // 5)
            and state.min_raise <= 4
            and self._is_valid(state.valid_actions, raise_action)
        ):
            target = max(state.min_raise, min(state.max_raise, max(2, state.min_raise)))
            return (raise_action, target, 0, 0)
        if state.continue_cost > 0 and self._is_valid(state.valid_actions, call):
            return (call, 0, 0, 0)
        if self._is_valid(state.valid_actions, check):
            return (check, 0, 0, 0)
        return None

    def act(
        self,
        observation: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Any,
    ) -> Tuple[int, int, int, int]:
        try:
            state = self.state_manager.parse(
                observation, info if isinstance(info, dict) else {}
            )
            self.opponent_model.record_state(state)

            mode, tavg = self.time_supervisor.select_mode(state)
            self.current_mode = mode

            if state.can_discard:
                discard_mode = "survival" if mode == "survival" else mode
                action, discard_ev = self.discard_engine.choose_discard(
                    state, discard_mode
                )
                self.logger.debug(
                    "Hand %s discard mode=%s tavg=%.3f ev=%.3f action=%s",
                    state.hand_number,
                    mode,
                    tavg,
                    discard_ev,
                    action,
                )
                return self._validated_action(
                    action, state.valid_actions, state.min_raise, state.max_raise
                )

            preflop_action = self._preflop_aggression_action(state)
            if preflop_action is not None:
                self.logger.debug(
                    "Hand %s street=0 forced preflop aggression mode=%s action=%s",
                    state.hand_number,
                    mode,
                    preflop_action,
                )
                return self._validated_action(
                    preflop_action,
                    state.valid_actions,
                    state.min_raise,
                    state.max_raise,
                )

            if mode == "survival":
                return self._validated_action(
                    self._survival_action(state),
                    state.valid_actions,
                    state.min_raise,
                    state.max_raise,
                )

            opponent_range = None
            if mode == "full" and state.street >= 2 and state.continue_cost > 0:
                opponent_range = self.discard_engine.narrowed_opponent_range(
                    state, mode
                )

            if state.street == 0 and len(state.my_cards) >= 5:
                my_ev = self.luts.get_preflop_equity(state.my_cards[:5])
            elif len(state.my_cards) >= 2:
                my_ev = self.discard_engine.estimate_hero_equity(
                    hero_cards=state.my_cards[:2],
                    board_cards=state.community_cards,
                    mode=mode,
                    opponent_range=opponent_range,
                    dead_cards=state.opp_discarded_cards,
                )
            else:
                my_ev = 0.5

            margin = self.opponent_model.get_margin(self.base_margin)
            action = self.decision_engine.decide(state, my_ev, margin, mode)
            self.logger.debug(
                "Hand %s street=%s mode=%s tavg=%.3f ev=%.3f margin=%.3f action=%s",
                state.hand_number,
                state.street,
                mode,
                tavg,
                my_ev,
                margin,
                action,
            )
            return self._validated_action(
                action, state.valid_actions, state.min_raise, state.max_raise
            )
        except Exception as exc:
            self.logger.exception("act() failure: %s", exc)
            valid_actions = tuple(
                int(v) for v in observation.get("valid_actions", [1, 0, 0, 0, 0])
            )
            return self._safe_action(valid_actions)

    def observe(
        self,
        observation: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Any,
    ) -> None:
        try:
            state = self.state_manager.parse(
                observation, info if isinstance(info, dict) else {}
            )
            self.opponent_model.record_state(state)
            if terminated:
                self.opponent_model.finalize_hand(
                    state, info if isinstance(info, dict) else {}
                )
        except Exception as exc:
            self.logger.exception("observe() failure: %s", exc)
