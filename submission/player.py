from typing import Any, Tuple

from agents.agent import Agent, Observation
from gym_env import PokerEnv


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True) -> None:
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

    def __name__(self) -> str:
        return "PlayerAgent"

    def act(
        self,
        observation: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Any,
    ) -> Tuple[int, int, int, int]:
        """
        Starter: Folds whenever possible. On the flop discard round (mandatory), keeps first two cards.

        Args (Gym-style step callback):
          observation: Dict with game state for this player (street, my_cards, community_cards,
                       my_bet, opp_bet, valid_actions, my_discarded_cards, opp_discarded_cards, etc.).
          reward: Chip reward from the previous step (e.g. +2 if you won the pot).
          terminated: True if the hand has ended (fold or showdown).
          truncated: True if the episode was cut off (e.g. time limit); usually False per hand.
          info: Extra dict (e.g. hand_number, and at showdown player_0_cards, player_1_cards, community_cards).
        """
        # Example usage of logger
        self.logger.info(f"Hand {info.get('hand_number', '?')} street {observation['street']}")

        valid_actions = observation["valid_actions"]

        if valid_actions[self.action_types.DISCARD.value]:
            return self.action_types.DISCARD.value, 0, 0, 1

        return self.action_types.FOLD.value, 0, 0, 0

