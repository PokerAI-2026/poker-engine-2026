"""
Graphical 1v1 poker GUI for manual play or bot-vs-bot watching.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

from agents.test_agents import CallingStationAgent, FoldAgent, RandomAgent
from gym_env import PokerEnv

# Suppress noisy Streamlit bare-mode warning spam.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

try:
    from submission.player import PlayerAgent
    PLAYER_AGENT_AVAILABLE = True
except ImportError:
    PlayerAgent = None  # type: ignore
    PLAYER_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

HUMAN_KEY = "Human"
SUIT_SYMBOLS = {"d": "♦", "h": "♥", "s": "♠"}
RED_SUITS = {"♦", "♥"}

AGENT_CHOICES: Dict[str, Any] = {
    HUMAN_KEY: None,
    "FoldAgent": FoldAgent,
    "CallingStationAgent": CallingStationAgent,
    "RandomAgent": RandomAgent,
}
if PLAYER_AGENT_AVAILABLE:
    AGENT_CHOICES["PlayerAgent (submission)"] = PlayerAgent


def _inject_css() -> None:
    st.markdown(
        """
<style>
.table-wrap {
  background: radial-gradient(ellipse at center, #195f3d 0%, #0d3b26 70%);
  border: 2px solid #244f3a;
  border-radius: 18px;
  padding: 16px;
  margin-bottom: 12px;
}
.seat-title {
  color: #f4f4f4;
  font-weight: 700;
  margin-bottom: 6px;
}
.card-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.card {
  width: 56px;
  height: 76px;
  border-radius: 8px;
  border: 1px solid #e2e2e2;
  background: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  font-weight: 700;
  box-shadow: 0 2px 4px rgba(0,0,0,0.35);
}
.card.red {
  color: #c62828;
}
.card.black {
  color: #111111;
}
.card.back {
  background: repeating-linear-gradient(
    45deg,
    #1e3a8a,
    #1e3a8a 8px,
    #3b82f6 8px,
    #3b82f6 16px
  );
  border-color: #bcd1ff;
  color: transparent;
}
.table-meta {
  color: #f4f4f4;
  font-weight: 600;
  margin: 8px 0;
}
.small-note {
  color: #d9d9d9;
  font-size: 0.9rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _convert_numpy(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(x) for x in obj]
    return obj


def _card_from_int(env: PokerEnv, card_int: int) -> Tuple[str, str]:
    # Returns (rank, suit_symbol)
    code = env.int_card_to_str(card_int)
    return code[0], SUIT_SYMBOLS.get(code[1], code[1])


def _cards_html_from_ints(env: PokerEnv, cards: List[int], hidden: bool = False) -> str:
    visible = [c for c in cards if c != -1]
    n = max(1, len(visible))
    chunks: List[str] = []
    if hidden:
        for _ in range(n):
            chunks.append('<div class="card back">?</div>')
    else:
        for c in visible:
            rank, suit = _card_from_int(env, c)
            color_cls = "red" if suit in RED_SUITS else "black"
            chunks.append(f'<div class="card {color_cls}">{rank}{suit}</div>')
    return '<div class="card-row">' + "".join(chunks) + "</div>"


def _cards_html_from_strs(cards: List[str]) -> str:
    chunks: List[str] = []
    for c in cards:
        if len(c) < 2:
            continue
        rank, suit = c[0], SUIT_SYMBOLS.get(c[1], c[1])
        color_cls = "red" if suit in RED_SUITS else "black"
        chunks.append(f'<div class="card {color_cls}">{rank}{suit}</div>')
    if not chunks:
        chunks.append('<div class="card back">?</div>')
    return '<div class="card-row">' + "".join(chunks) + "</div>"


def _ensure_obs_extra(obs0: Dict[str, Any], obs1: Dict[str, Any]) -> None:
    # Keep compatibility with agents that expect these keys.
    obs0.setdefault("time_used", 0.0)
    obs1.setdefault("time_used", 0.0)
    obs0.setdefault("time_left", 500.0)
    obs1.setdefault("time_left", 500.0)
    obs0.setdefault("opp_last_action", "None")
    obs1.setdefault("opp_last_action", "None")


def _init_state() -> None:
    defaults = {
        "env": None,
        "obs0": None,
        "obs1": None,
        "reward0": 0.0,
        "reward1": 0.0,
        "terminated": True,
        "truncated": False,
        "info": {},
        "history": [],
        "agent0": None,
        "agent1": None,
        "agent0_name": "",
        "agent1_name": "",
        "show_opponent_cards": False,
        "match_total_0": 0.0,
        "match_total_1": 0.0,
        "hand_no": 1,
        "hand_scored": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _new_hand(name0: str, name1: str, seed: int | None, reset_match: bool = False) -> None:
    if reset_match:
        st.session_state.match_total_0 = 0.0
        st.session_state.match_total_1 = 0.0
        st.session_state.hand_no = 1

    env = PokerEnv(logger=logger)
    sb = (st.session_state.hand_no - 1) % 2
    (obs0, obs1), info = env.reset(seed=seed, options={"small_blind_player": sb})
    _ensure_obs_extra(obs0, obs1)

    cls0 = AGENT_CHOICES[name0]
    cls1 = AGENT_CHOICES[name1]
    agent0 = cls0(stream=False) if cls0 is not None else None
    agent1 = cls1(stream=False) if cls1 is not None else None

    st.session_state.env = env
    st.session_state.obs0 = obs0
    st.session_state.obs1 = obs1
    st.session_state.reward0 = 0.0
    st.session_state.reward1 = 0.0
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.info = info
    st.session_state.history = []
    st.session_state.agent0 = agent0
    st.session_state.agent1 = agent1
    st.session_state.agent0_name = name0
    st.session_state.agent1_name = name1
    st.session_state.hand_scored = False


def _apply_action(action: Tuple[int, int, int, int], actor_label: str, action_name: str) -> None:
    env = st.session_state.env
    obs0 = st.session_state.obs0
    obs1 = st.session_state.obs1
    reward0 = st.session_state.reward0
    reward1 = st.session_state.reward1

    (obs0, obs1), (reward0, reward1), term, trunc, info = env.step(action)
    _ensure_obs_extra(obs0, obs1)

    st.session_state.obs0 = obs0
    st.session_state.obs1 = obs1
    st.session_state.reward0 = reward0
    st.session_state.reward1 = reward1
    st.session_state.terminated = term
    st.session_state.truncated = trunc
    st.session_state.info = info
    st.session_state.history.append(
        {
            "street": obs0.get("street", "?"),
            "actor": actor_label,
            "action_name": action_name,
            "action": action,
        }
    )

    if term and not st.session_state.hand_scored:
        st.session_state.match_total_0 += float(reward0)
        st.session_state.match_total_1 += float(reward1)
        st.session_state.hand_scored = True


def _bot_step() -> None:
    obs0 = st.session_state.obs0
    obs1 = st.session_state.obs1
    acting = obs0["acting_agent"]
    obs_acting = _convert_numpy(obs0 if acting == 0 else obs1)

    reward = st.session_state.reward0 if acting == 0 else st.session_state.reward1
    agent = st.session_state.agent0 if acting == 0 else st.session_state.agent1
    if agent is None:
        return

    action = agent.act(
        obs_acting,
        reward,
        st.session_state.terminated,
        st.session_state.truncated,
        st.session_state.info,
    )
    action_type = PokerEnv.ActionType(action[0]).name
    actor_label = "P0" if acting == 0 else "P1"
    _apply_action(tuple(action), actor_label, action_type)


def main() -> None:
    st.set_page_config(page_title="Poker GUI", layout="wide")
    _init_state()
    _inject_css()

    st.title("Poker GUI (1v1)")

    with st.sidebar:
        st.subheader("Setup")
        name0 = st.selectbox("Player 0 (SB)", list(AGENT_CHOICES.keys()), key="select_agent0")
        name1 = st.selectbox("Player 1 (BB)", list(AGENT_CHOICES.keys()), key="select_agent1")

        use_seed = st.checkbox("Use fixed seed", value=False)
        seed_val = st.number_input("Seed", min_value=0, value=0, step=1)
        seed = int(seed_val) if use_seed else None

        c1, c2 = st.columns(2)
        with c1:
            if st.button("New hand", use_container_width=True):
                _new_hand(name0, name1, seed=seed, reset_match=False)
                st.rerun()
        with c2:
            if st.button("Reset match", use_container_width=True):
                _new_hand(name0, name1, seed=seed, reset_match=True)
                st.rerun()

        st.divider()
        st.checkbox("Show opponent cards", key="show_opponent_cards")

        with st.expander("Tournament rules", expanded=False):
            st.markdown(
                """
- **Format:** 1v1, 1,000 hands; each hand starts at 100 chips.
- **Deck:** 27 cards = suits (♦, ♥, ♠) and ranks (2-9, A).
- **Ranking:** Straight Flush > Full House > Flush > Straight > Trips > Two Pair > Pair > High Card.
- **Flow:** Pre-flop betting -> Flop + discard (keep exactly 2 cards) -> Turn betting -> River betting -> showdown/fold.
- **Discard twist:** discarded hole cards are revealed.
                """
            )

    # Start once on first load.
    if st.session_state.env is None:
        _new_hand(name0, name1, seed=seed, reset_match=False)

    env: PokerEnv = st.session_state.env
    obs0 = st.session_state.obs0
    obs1 = st.session_state.obs1
    terminated = st.session_state.terminated

    # Determine who is human (if anyone)
    human_seat = None
    if st.session_state.agent0 is None and st.session_state.agent1 is not None:
        human_seat = 0
    elif st.session_state.agent1 is None and st.session_state.agent0 is not None:
        human_seat = 1

    # Table ordering: opponent top, you bottom when human is present.
    if human_seat == 0:
        top_seat, bottom_seat = 1, 0
    elif human_seat == 1:
        top_seat, bottom_seat = 0, 1
    else:
        top_seat, bottom_seat = 0, 1

    names = {
        0: f"Player 0 (SB) - {st.session_state.agent0_name}",
        1: f"Player 1 (BB) - {st.session_state.agent1_name}",
    }
    if human_seat is not None:
        names[human_seat] = "You"

    bets = {0: obs0["my_bet"], 1: obs1["my_bet"]}
    cards = {0: list(env.player_cards[0]), 1: list(env.player_cards[1])}

    # Opponent card visibility toggle
    reveal_opp = st.session_state.show_opponent_cards or terminated or human_seat is None

    top_hidden = (human_seat is not None and top_seat != human_seat and not reveal_opp)
    bottom_hidden = False  # always show "you" (or both in bot mode)

    # Board cards revealed by street
    street = int(obs0["street"])
    board_n = 0 if street == 0 else street + 2
    board_cards = list(env.community_cards[:board_n])

    street_names = {0: "Pre-flop", 1: "Flop", 2: "Turn", 3: "River"}
    to_act = int(obs0["acting_agent"])

    # Header metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hand #", st.session_state.hand_no)
    m2.metric("Street", street_names.get(street, str(street)))
    m3.metric("Pot", int(obs0["pot_size"]))
    m4.metric("To act", names[to_act])

    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)

    # Top seat
    st.markdown(f'<div class="seat-title">{names[top_seat]} · Bet: {bets[top_seat]}</div>', unsafe_allow_html=True)
    st.markdown(_cards_html_from_ints(env, cards[top_seat], hidden=top_hidden), unsafe_allow_html=True)

    st.markdown('<div class="table-meta">Board</div>', unsafe_allow_html=True)
    st.markdown(_cards_html_from_ints(env, board_cards, hidden=False), unsafe_allow_html=True)

    # Discard visibility after flop-discard phase starts
    disc0 = [c for c in env.discarded_cards[0] if c != -1]
    disc1 = [c for c in env.discarded_cards[1] if c != -1]
    if disc0 or disc1:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"<div class='small-note'>P0 discarded</div>", unsafe_allow_html=True)
            st.markdown(_cards_html_from_ints(env, disc0, hidden=False), unsafe_allow_html=True)
        with d2:
            st.markdown(f"<div class='small-note'>P1 discarded</div>", unsafe_allow_html=True)
            st.markdown(_cards_html_from_ints(env, disc1, hidden=False), unsafe_allow_html=True)

    st.markdown(f'<div class="seat-title" style="margin-top:10px;">{names[bottom_seat]} · Bet: {bets[bottom_seat]}</div>', unsafe_allow_html=True)
    st.markdown(_cards_html_from_ints(env, cards[bottom_seat], hidden=bottom_hidden), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Hand completion / scoreboard
    if terminated:
        st.success(f"Hand finished: P0 {st.session_state.reward0:+.0f}, P1 {st.session_state.reward1:+.0f}")
        info = st.session_state.info
        if "player_0_cards" in info:
            st.write("Showdown")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown("**P0 hole cards**")
                st.markdown(_cards_html_from_strs(info.get("player_0_cards", [])), unsafe_allow_html=True)
            with s2:
                st.markdown("**Board**")
                st.markdown(_cards_html_from_strs(info.get("community_cards", [])), unsafe_allow_html=True)
            with s3:
                st.markdown("**P1 hole cards**")
                st.markdown(_cards_html_from_strs(info.get("player_1_cards", [])), unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        s1.metric("Match total P0", f"{st.session_state.match_total_0:+.0f}")
        s2.metric("Match total P1", f"{st.session_state.match_total_1:+.0f}")

        if st.button("Next hand", use_container_width=True):
            st.session_state.hand_no += 1
            _new_hand(name0, name1, seed=seed, reset_match=False)
            st.rerun()

    else:
        # Manual input when it's the human seat's turn.
        human_turn = human_seat is not None and to_act == human_seat
        if human_turn:
            st.subheader("Your action")
            obs_human = obs0 if human_seat == 0 else obs1
            valid = [int(x) for x in list(obs_human["valid_actions"])]
            At = PokerEnv.ActionType

            if valid[At.DISCARD.value]:
                my_cards = [c for c in obs_human["my_cards"] if c != -1]
                labels = [f"{_card_from_int(env, c)[0]}{_card_from_int(env, c)[1]} (index {i})" for i, c in enumerate(my_cards)]
                k1 = st.selectbox("Keep card 1", list(range(len(my_cards))), format_func=lambda i: labels[i], key="keep_1")
                k2_choices = [i for i in range(len(my_cards)) if i != k1]
                k2 = st.selectbox("Keep card 2", k2_choices, format_func=lambda i: labels[i], key="keep_2")
                if st.button("Submit discard", use_container_width=True):
                    _apply_action((At.DISCARD.value, 0, int(k1), int(k2)), f"P{human_seat}", "DISCARD")
                    st.rerun()
            else:
                min_raise = int(obs_human["min_raise"])
                max_raise = int(obs_human["max_raise"])
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    if valid[At.FOLD.value] and st.button("Fold", use_container_width=True):
                        _apply_action((At.FOLD.value, 0, 0, 0), f"P{human_seat}", "FOLD")
                        st.rerun()
                with c2:
                    if valid[At.CALL.value] and st.button("Call", use_container_width=True):
                        _apply_action((At.CALL.value, 0, 0, 0), f"P{human_seat}", "CALL")
                        st.rerun()
                with c3:
                    if valid[At.CHECK.value] and st.button("Check", use_container_width=True):
                        _apply_action((At.CHECK.value, 0, 0, 0), f"P{human_seat}", "CHECK")
                        st.rerun()
                with c4:
                    raise_to = st.number_input("Raise to", min_value=min_raise, max_value=max_raise, value=min_raise, step=1)
                    if valid[At.RAISE.value] and st.button("Raise", use_container_width=True):
                        _apply_action((At.RAISE.value, int(raise_to), 0, 0), f"P{human_seat}", "RAISE")
                        st.rerun()
        else:
            # Bot's turn or bot-vs-bot watching controls.
            both_bots = st.session_state.agent0 is not None and st.session_state.agent1 is not None
            if both_bots:
                a, b = st.columns(2)
                with a:
                    if st.button("Step one action", use_container_width=True):
                        _bot_step()
                        st.rerun()
                with b:
                    if st.button("Play hand to end", use_container_width=True):
                        for _ in range(250):
                            if st.session_state.terminated:
                                break
                            _bot_step()
                        st.rerun()
            else:
                if st.button("Opponent acts", use_container_width=True):
                    _bot_step()
                    st.rerun()

    if st.session_state.history:
        with st.expander("Action history"):
            for i, h in enumerate(st.session_state.history, start=1):
                st.write(f"{i}. {h['actor']} -> {h['action_name']} {h['action']}")


if __name__ == "__main__":
    main()
