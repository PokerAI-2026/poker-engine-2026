# Submission agent: logic and structure

This document describes the poker agent under `submission/`: how it is organized, how decisions flow, and how equities and actions are computed.

## High-level overview

The agent is a **rule-based bot** (no online learning) that:

1. **Parses** each gym observation into a typed `ParsedState`.
2. **Adapts compute** via a **time budget** (`full` / `fast` / `survival` modes).
3. On the **flop discard phase**, picks **two hole cards to keep** by maximizing estimated equity vs a random opponent (with LUT + cache).
4. On **betting streets**, estimates **hero equity** (preflop 5-card LUT vs postflop Monte Carlo), adjusts a **call threshold** from simple opponent stats, then applies **pot-odds-style** fold/call/raise rules.

Core dependencies: `agents.agent.Agent`, `gym_env.PokerEnv` (action types, 27-card deck).

---

## Directory layout

| Path | Role |
|------|------|
| `player.py` | `PlayerAgent`: orchestrates `act()` / `observe()`, wires all engines. |
| `engines.py` | `ParsedState`, `StateManager`, `TimeSupervisor`, `OpponentModel`, `DiscardEngine`, `DecisionEngine`. |
| `lut_store.py` | `LUTStore`: loads `.npy` tables, 7-card best-5 evaluation, preflop equity, flop EV cache. |
| `lut_builder.py` | **Offline** script to generate `hand5_strength.npy`, `preflop_equity.npy`, `flop_seed_table.npy`. |
| `data/*.npy` | Precomputed artifacts used at runtime. |

---

## Entry point: `PlayerAgent`

- Subclasses `Agent` from `agents.agent`.
- **Constructs** (in order):
  - `LUTStore(submission/data)` — strength + preflop equity + flop seed cache.
  - `StateManager()` — observation → `ParsedState`; also appends opponent action tokens to an internal deque (currently **not read** elsewhere).
  - `TimeSupervisor(total_hands=1000)` — mode selection from clock.
  - `OpponentModel()` — VPIP/PFR/c-bet and showdown aggression counters.
  - `DiscardEngine(luts)` — discard choice + equity sampling + opponent range narrowing.
  - `DecisionEngine(PokerEnv.ActionType)` — fold / check / call / raise sizing.
- `PLAYER_ID` env var is read into `self.player_id` but **not used** in the current decision logic.
- `base_margin = 0.05` — baseline extra edge required to call when facing a bet (then adjusted by `OpponentModel`).

### `act(observation, reward, terminated, truncated, info)`

1. `state = state_manager.parse(observation, info)`
2. `opponent_model.record_state(state)`
3. `mode, tavg = time_supervisor.select_mode(state)` → `"full"`, `"fast"`, or `"survival"`
4. **If `state.can_discard`** (flop discard legal):
   - Uses `discard_mode = "survival"` when mode is survival, else same as `mode`.
   - `choose_discard(state, discard_mode)` → validated `(DISCARD, 0, keep_i, keep_j)`.
5. **If `mode == "survival"`** (and not discarding): `_survival_action(state)` — conservative preflop / check-fold postflop.
6. **Else** (normal betting):
   - If `mode == "full"`: `opponent_range = narrowed_opponent_range(state, mode)` (may be full random pairs).
   - **Equity:**
     - Street 0 with 5 hole cards: `luts.get_preflop_equity(first five cards)`.
     - Else if ≥2 hole cards: `discard_engine.estimate_hero_equity(...)` with optional narrowed range and opponent discards as dead cards.
     - Else: `my_ev = 0.5`.
   - `margin = opponent_model.get_margin(base_margin)`
   - `decision_engine.decide(state, my_ev, margin, mode)` → fold/check/call/raise.
7. All actions pass through `_validated_action` (clamp raise, valid discard indices, fallback `_safe_action` on invalidity).
8. On any exception: log and `_safe_action` from observation.

### `observe(...)`

Re-parses state, records opponent model, and on `terminated` calls `opponent_model.finalize_hand` to update VPIP/PFR/c-bet and showdown stats (when `info` exposes both players’ cards).

---

## `ParsedState` (engines)

Immutable snapshot used everywhere:

- Cards: `my_cards`, `community_cards`, `opp_discarded_cards` (ints, -1 stripped).
- Betting: `my_bet`, `opp_bet`, `min_raise`, `max_raise`, `pot_size`.
- `continue_cost = max(0, opp_bet - my_bet)`.
- `valid_actions` bit/mask vector from env (index 4 = discard when present).
- `can_discard`, `street`, `hand_number`, `time_used`, `time_left`, `blind_position`, `opp_last_action`.

---

## Time modes: `TimeSupervisor`

Assumes **1000 hands** total. Let `remaining_hands = max(1, 1000 - hand_number)`, `tavg = time_left / remaining_hands`:

| Condition | Mode | Effect |
|-----------|------|--------|
| `tavg ≤ 0.20` | `survival` | Minimal risk: special preflop rules; postflop check/fold bias; discard uses fewer samples. |
| `0.20 < tavg ≤ 0.80` | `fast` | Medium MC samples; slightly smaller raises. |
| `tavg > 0.80` | `full` | More samples; opponent range narrowing on postflop; larger default raise factors. |

---

## `OpponentModel`

Tracks per-hand flags from opponent line:

- **VPIP**: preflop CALL or RAISE.
- **PFR**: preflop RAISE.
- **C-bet**: preflop aggressor raises again on flop (street 1).

On hand end (`finalize_hand`): increments aggregates; if showdown info exists, counts showdowns and how often opponent was preflop/ flop aggressive.

**`get_margin(base)`** (clamped to `[0.02, 0.15]`):

- Tighten (call more): high VPIP (>0.80) and high PFR (>0.45) → `margin -= 0.02`.
- Widen (fold more): low PFR (<0.20) → `margin += 0.03`.
- High c-bet freq (>0.70) → `margin += 0.01`.

This margin is added to **pot odds** when deciding whether to call facing a bet.

---

## `DiscardEngine`

### Flop discard (`choose_discard`)

- Requires 5 hole cards and flop (≥3 board cards). Otherwise returns a default discard action `(4, 0, 0, 1)`.
- Enumerates all **10 pairs** of indices `KEEP_INDEX_PAIRS` (which two cards to keep).
- For each pair, `get_or_estimate_flop_ev(keep, flop, mode)`:
  - Lookup `pack_flop_key` in `LUTStore.flop_cache` (seed table + runtime fills).
  - If miss: Monte Carlo `_sample_equity` with **uniform random opponent hole** and random turn/river, sample count capped by mode (`full` 375, `fast` 200, survival ~100).
- Picks the pair with **maximum EV**; returns `(DISCARD, 0, i, j)` and best EV.

### Postflop hero equity (`estimate_hero_equity`)

- Builds known board (up to 5), seeds RNG from hero + flop key.
- `_sample_size(mode)` samples as above.
- If `opponent_range` is `None`: all remaining 2-card combos vs hero.
- If range provided: only those combos; can do **exact enumeration** when board complete and ≤200 combos.
- **Dead cards**: opponent’s discarded cards excluded from deck.

### Opponent range narrowing (`narrowed_opponent_range`, `full` mode only)

- After flop, builds all unknown 2-card combos.
- From opponent’s **three discarded** cards, computes **lower bound** = max over pairs of those cards of their flop EV (proxy for “strength they threw away”).
- Filters opponent combos to those whose **quick** flop EV (80-sample MC, cached) ≥ that lower bound.
- If filter empties, falls back to full range.

---

## `DecisionEngine.decide`

Uses `my_ev` ∈ [0,1] as win-equity proxy vs pot odds.

### Free to act (`continue_cost == 0`)

- **Raise** if `my_ev > 0.65` and raise legal; size ≈ `factor * max(2, pot)` with factor 0.50 / 0.40 / 0.30 by mode, +0.15 if edge > 0.35, clamped to `[min_raise, max_raise]`.
- Else **check** if legal, else **call**, else min **raise**, else **fold**.

### Facing a bet

- `pot_odds = continue_cost / (pot + continue_cost)`.
- `effective_ev = my_ev * discount`, `discount = max(0.65, 1 - 0.25 * min(2, continue_cost/pot))` (penalize large calls).
- **Raise** if `my_ev > 0.80` and raise legal.
- **Call** if `effective_ev >= pot_odds + margin`.
- Else **fold** (or check/call fallbacks if fold illegal).

---

## `LUTStore` and data artifacts

- **Deck**: `N_CARDS = 27`. Combinations indexed by **combinadic** `combo_to_index` (5-card and 2-card).
- **`hand5_strength.npy`**: For each 5-card combo, evaluator score (lower = stronger in treys convention used by env). Used to score any 5 of 7 for showdown comparison.
- **`preflop_equity.npy`**: Per **5-card** starting hand, average of pairwise 2-card equities vs random opponent (built in `lut_builder`). Runtime **premium** = equity ≥ 90th percentile → used in survival preflop.
- **`flop_seed_table.npy`**: Pickled dict `int key → float EV` for random (keep_pair, flop) states; warm-starts flop discard / cache.
- **Runtime `flop_cache`**: Fills in missing keys during play.

If `.npy` files are missing, code falls back to building strength on the fly or deriving crude preflop numbers from strength percentiles.

---

## `lut_builder.py` (offline)

CLI to regenerate artifacts:

- `generate_hand5_strength()` — full enumeration vs env evaluator.
- `generate_pair_equity()` — MC per 2-card hand vs random opp + board.
- `build_preflop_equity_from_pairs()` — average of C(5,2) pair equities per 5-card hand.
- `generate_flop_seed_table()` — random (keep, flop) keys with MC EV for turn/river.

---

## Design summary

| Concern | Mechanism |
|---------|-----------|
| Strength / showdown | Precomputed 5-card LUT + best-of-21 five-card subsets for 7 cards. |
| Preflop (5 cards) | LUT equity + survival premium gate. |
| Postflop equity | Monte Carlo vs full or narrowed opponent range. |
| Discard | Max EV over 10 keep pairs; cached/seeding for flop states. |
| Bet sizing | Pot-fraction raises scaled by mode and edge. |
| Exploitability | Light stat-based margin on calls; range narrow from discards in full mode. |
| Time safety | Survival mode shrinks sampling and plays passively. |

---

## Minor implementation notes

- `StateManager._opp_action_history` is populated but not consumed by current logic.
- `self.player_id` is unused in decisions.
- Treys-style scores: **lower** numeric score = stronger hand in `evaluate_7card_score` comparisons (`my_score < opp_score` ⇒ win).
