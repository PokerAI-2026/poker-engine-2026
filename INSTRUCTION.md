````markdown
# Flop Discard Tables README

## Goal

Implement a **fast flop discard strategy** for the CMU AI Poker bot using a precomputed **big blind (BB) lookup table** and a simple **small blind (SB) fallback strategy**.

The game rules relevant to this project are:

- 27-card deck: suits `{♦, ♥, ♠}` and ranks `{2,3,4,5,6,7,8,9,A}`
- Each player starts with **5 hole cards**
- On the flop, **3 community cards** are dealt
- Then each player must **discard 3** and **keep 2**
- **Big blind discards first**
- Discarded cards are revealed to the opponent
- After discards, the hand continues with turn, river, and betting

This implementation only targets the **flop discard decision**.

---

# High-level design

We will implement the following:

1. A **BB lookup table** keyed by `(my 5 hole cards, flop 3 cards)`
2. Each BB table entry stores:
   - best keep choice among the 10 possible keep-2 subsets
   - second-best keep choice
   - best equity
   - equity gap between best and second-best
3. A **simple SB strategy**:
   - use the same BB table as a baseline
   - ignore opponent revealed discards for now
   - optionally re-evaluate top 2 candidates on the fly later
4. A compact binary format for storing/loading the table
5. Suit canonicalization to reduce table size

This first version is intentionally simple and robust.

---

# Version 1 strategy

## Big blind (BB)

When we are BB on the flop:

- we must discard **before** seeing opponent’s discard
- therefore our best discard is a function only of:
  - our 5 hole cards
  - the 3 flop cards

So the BB table is indexed by:

`(my_5_cards, flop_3_cards)`

At runtime:
- enumerate current flop state
- canonicalize it
- compute table index
- look up best keep
- map back to original card identities
- discard the other 3 cards

## Small blind (SB)

For the simple version:

- reuse the BB table
- choose discard based only on `(my_5_cards, flop_3_cards)`
- ignore opponent revealed discards for now

This is not optimal, but it is easy to implement and should still be useful.

Later, SB can be improved by:
- exact re-evaluation of top 2 BB candidates after seeing opponent discards
- or a small adjustment model

---

# State definition

A flop discard state consists of:

- 5 private hole cards
- 3 flop cards

Total visible cards for the decision: 8

The raw number of such states is:

`C(27,5) * C(22,3)`

Equivalently:

`27! / (19! * 5! * 3!)`

This is:

`124,324,200`

Each state has exactly:

`C(5,2) = 10`

possible keep choices.

---

# What is stored per state

For each state we store:

- `best_keep_idx` : `uint8`
- `second_keep_idx` : `uint8`
- `best_equity` : `uint16`
- `equity_gap` : `uint16`

Total:

- **6 bytes per stored state**

## Keep index convention

Let the 5 hole cards in canonical order be positions:

`0,1,2,3,4`

The 10 keep-2 choices are ordered as:

```python
KEEP_CHOICES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
]
````

Then:

* `best_keep_idx` is an integer in `[0, 9]`
* `second_keep_idx` is also in `[0, 9]`

## Equity encoding

Store equity as a `uint16` representing a number in `[0,1]`.

Encoding:

```python
q = round(equity * 65535)
```

Decoding:

```python
equity = q / 65535.0
```

## Gap encoding

Let:

```python
gap = best_equity - second_best_equity
```

Store this as `uint16` using the same quantization rule.

This gap can be used later to:

* decide whether BB best choice is clearly dominant
* identify close cases where SB re-evaluation would be valuable

---

# Table semantics

## BB table entry meaning

For a given `(my_5_cards, flop_3_cards)` state, for each keep choice:

* keep those 2 cards
* discard the other 3
* evaluate expected showdown equity under random compatible opponent holdings and future runouts

Then:

* `best_keep_idx` = keep choice with highest equity
* `second_keep_idx` = runner-up
* `best_equity` = equity of the best keep
* `equity_gap` = difference between best and second-best equity

## Important simplification

This version computes **showdown equity**, not full betting EV.

That means:

* no future betting strategy is modeled
* no fold equity is modeled
* no exploitative opponent model is used

This is acceptable for V1.

---

# Equity evaluator specification

This is the most important part of the implementation.

For each candidate keep:

1. We know:

   * my kept 2 cards
   * my discarded 3 cards
   * flop 3 cards

2. Unknown cards are:

   * opponent original 5 cards
   * opponent future kept 2 cards after their discard choice
   * turn
   * river

We need a tractable and consistent approximation.

## Required V1 evaluator

Use this approximation:

For each candidate keep:

1. Enumerate all possible opponent 5-card holdings from the remaining 19 unseen cards
2. For each opponent holding:

   * assume opponent also chooses their best keep-2 by showdown equity under the same evaluator
3. Enumerate all possible turn/river pairs from the remaining deck
4. Evaluate final showdown result
5. Average over all compatible opponent holdings and turn/river runouts

This yields an approximate self-consistent showdown equity.

## Why this is acceptable

It is expensive offline but very cheap online after tabulation.

The deck is small enough that this should be feasible with good engineering and batching.

---

# Hand evaluator requirements

Implement a custom hand evaluator for this game variant.

## Deck

* suits: 3
* ranks: 9 (`2,3,4,5,6,7,8,9,A`)
* ace can be low or high in straights

## Hand rankings from strongest to weakest

1. straight flush
2. full house
3. flush
4. straight
5. three of a kind
6. two pair
7. one pair
8. high card

## Important note

Four of a kind is impossible in this deck and should never be considered.

## Evaluator API

Implement:

```python
def evaluate_5card_hand(cards) -> tuple:
    """
    Returns a comparable rank tuple.
    Larger tuple means stronger hand.
    """
```

And:

```python
def best_5_from_7(cards7) -> tuple:
    """
    cards7 is length 7.
    Returns the best 5-card hand rank tuple.
    """
```

Since final hands after discards use:

* 2 private cards
* 5 community cards

we need best 5 from 7.

## Comparison rule

The returned tuple must be directly comparable with Python tuple comparison.

Example shape:

```python
(category, tiebreak1, tiebreak2, ...)
```

Where larger is better.

You may choose any consistent encoding, but it must correctly resolve all ties.

---

# Card representation

Use a compact integer representation for cards.

Recommended:

* rank ids: `0..8` for `2,3,4,5,6,7,8,9,A`
* suit ids: `0..2` for `♦,♥,♠`

Then encode card as:

```python
card_id = rank * 3 + suit
```

So total cards are `0..26`.

Helper functions:

```python
def card_rank(card_id) -> int:
    return card_id // 3

def card_suit(card_id) -> int:
    return card_id % 3
```

This should be the canonical internal representation everywhere.

---

# Suit canonicalization

## Purpose

Many raw states are equivalent up to a global renaming of suits.

Example:

* swapping all hearts and spades everywhere does not change strategic structure

So instead of storing all raw states, map them to a **canonical suit representative**.

This reduces storage substantially.

## How to canonicalize

There are exactly `3! = 6` suit permutations.

Represent each as a mapping from old suit id to new suit id.

For a given state `(my5, flop3)`:

1. apply each of the 6 suit permutations to all 8 cards
2. sort:

   * hole cards into a canonical hole order
   * flop cards into a canonical board order
3. encode transformed state as a tuple
4. choose the lexicographically smallest encoding
5. return:

   * canonical state encoding
   * permutation used
   * mapping from original hole positions to canonical hole positions
   * inverse mapping if needed

## Canonical ordering convention

Within a transformed state:

* sort hole cards ascending by `(rank, suit)` or by integer `card_id`
* sort flop cards ascending by `card_id`

Be consistent everywhere.

## Important implementation note

Because keep indices refer to hole-card positions, canonicalization must preserve a mapping so that:

* stored best keep in canonical hole positions
* can be translated back to original cards at runtime

Recommended return type:

```python
@dataclass
class CanonicalizedState:
    canonical_hole: tuple[int, int, int, int, int]
    canonical_flop: tuple[int, int, int]
    original_to_canonical_hole_pos: tuple[int, int, int, int, int]
    canonical_to_original_hole_pos: tuple[int, int, int, int, int]
```

---

# State indexing

We need a deterministic integer index for each canonical state.

There are two ways to do this.

## Preferred method for V1: dictionary-built dense indexing

During offline generation:

1. enumerate all raw states
2. canonicalize each state
3. assign each distinct canonical state a dense integer id
4. write records into an array in that dense order
5. also persist the canonical-state -> dense-id mapping

This is simplest to implement.

### Files

* `bb_table.bin` : fixed-width records
* `bb_index.pkl` or `bb_index.bin` : canonical state -> dense id

This is acceptable offline, but runtime should avoid Python-heavy structures if possible.

## Better method later: combinadic/ranking index

Later, replace dictionary mapping with a direct perfect index:

* rank canonical hole 5-set
* rank canonical flop 3-set conditioned on hole
* combine into dense id

This is harder; do not block V1 on it.

---

# Offline generation pipeline

Implement a generator script:

`generate_bb_table.py`

## Inputs

* no user input required
* maybe CLI flags for:

  * output path
  * number of workers
  * checkpoint frequency
  * chunk size
  * whether to use sampling instead of exhaustive evaluation

## Outputs

* `bb_table.bin`
* `bb_meta.json`
* `bb_index.pkl` or better binary index
* optional progress checkpoints

## Steps

### Step 1: enumerate raw states

Enumerate all:

* hole 5-card combinations from 27-card deck
* flop 3-card combinations from remaining 22 cards

### Step 2: canonicalize

For each raw state:

* canonicalize
* skip if canonical state already processed

### Step 3: evaluate the 10 keep choices

For each canonical state:

* enumerate all 10 keep choices
* compute equity for each
* sort results descending

### Step 4: store record

Write a 6-byte record:

* best keep idx
* second keep idx
* best equity uint16
* equity gap uint16

### Step 5: serialize metadata

Write metadata including:

* version
* card encoding convention
* keep choice ordering
* quantization scheme
* canonicalization rule
* record size

---

# Runtime lookup API

Implement:

```python
class FlopDiscardTable:
    def __init__(self, table_path, index_path, meta_path):
        ...
    
    def choose_keep(self, my5, flop3):
        """
        Returns the 2 cards to keep.
        my5 and flop3 are iterables of card_ids.
        """
```

## Runtime procedure

1. canonicalize `(my5, flop3)`
2. find dense state id
3. read record
4. get `best_keep_idx`
5. map canonical keep positions back to original hole positions
6. return original 2 cards

Also implement:

```python
def choose_discard(self, my5, flop3):
    """
    Returns the 3 cards to discard.
    """
```

---

# Binary file format

Use fixed-width packed records.

## Record layout (6 bytes)

Byte layout:

* byte 0: `best_keep_idx` (`uint8`)
* byte 1: `second_keep_idx` (`uint8`)
* bytes 2-3: `best_equity_q` (`uint16`, little-endian)
* bytes 4-5: `gap_q` (`uint16`, little-endian)

Use little-endian everywhere.

## Why binary instead of torch pickle

Do **not** store this as:

* Python dict of objects
* Python pickle of lists of tuples
* heavy PyTorch object graphs

That wastes a lot of space and load time.

Use:

* raw binary
* optionally `numpy.memmap` for loading

---

# Recommended loader implementation

Use `numpy.memmap` or `mmap`.

Example:

```python
import numpy as np

table = np.memmap(table_path, mode="r", dtype=np.uint8)
```

Then parse records manually or define a structured dtype.

Example structured dtype:

```python
record_dtype = np.dtype([
    ("best_keep_idx", np.uint8),
    ("second_keep_idx", np.uint8),
    ("best_equity_q", "<u2"),
    ("gap_q", "<u2"),
])
```

Then:

```python
table = np.memmap(table_path, mode="r", dtype=record_dtype)
record = table[state_id]
```

This keeps RAM usage modest and lookup fast.

---

# Milestone plan

## Milestone 1: card + hand evaluator

Implement and test:

* card encoding
* 5-card evaluator
* best-5-from-7 evaluator

### Required tests

* known category examples for each hand class
* ace-low straight
* ace-high straight
* full house ordering
* flush ordering
* tie cases

## Milestone 2: keep choice enumeration

Implement:

* `KEEP_CHOICES`
* helper to convert keep idx to discard idx
* helper to select kept cards from hole cards

## Milestone 3: canonicalization

Implement:

* all 6 suit permutations
* canonical state selection
* hole-position mapping back and forth

### Required tests

* suit-renamed equivalent states canonicalize identically
* keep mapping round-trips correctly

## Milestone 4: equity evaluator

Implement flop equity evaluation for one state.

### First acceptable version

You may initially use:

* exact turn/river enumeration
* exhaustive opponent 5-card enumeration
* opponent best-keep response by same showdown-equity evaluator

If this is too slow, temporarily use sampled opponent holdings and runouts to validate pipeline.

## Milestone 5: offline generator

Implement chunked table generation with:

* resumable checkpoints
* multiprocessing
* progress logging

## Milestone 6: runtime integration

Implement:

* binary loader
* `choose_keep`
* `choose_discard`

Integrate into bot flop decision logic.

---

# Testing requirements

## Unit tests

Create tests for:

* card encode/decode
* 5-card hand ranking
* best-5-of-7
* keep choice ordering
* canonicalization equivalence
* mapping canonical keep back to original cards
* binary record pack/unpack

## Property tests

Suggested:

* canonicalization is idempotent
* applying any suit permutation then canonicalizing gives same canonical result
* equity quantize/dequantize stays within expected error tolerance

## Integration tests

For random states:

1. compute best keep directly with evaluator
2. store/load table entry
3. verify runtime lookup reproduces same keep

---

# Performance notes

## Table generation is offline

Offline generation may be expensive. That is okay.

Prioritize:

* correctness
* resumability
* deterministic output

## Runtime must be cheap

Runtime lookup should be:

* one canonicalization
* one index lookup
* one record read
* one keep mapping

No expensive rollout should happen during live BB decision.

## Parallelism

Use multiprocessing for offline generation.

Good pattern:

* producer enumerates canonical states
* worker pool evaluates states
* writer process serializes records/checkpoints

---

# Suggested project structure

```text
poker_tables/
    README.md
    cards.py
    hand_eval.py
    canonicalize.py
    equity.py
    keep_choices.py
    state_index.py
    generate_bb_table.py
    flop_table.py
    tests/
        test_cards.py
        test_hand_eval.py
        test_canonicalize.py
        test_keep_choices.py
        test_equity.py
        test_lookup.py
```

---

# Concrete API suggestions

## cards.py

```python
ALL_CARDS = list(range(27))

def card_rank(card_id: int) -> int: ...
def card_suit(card_id: int) -> int: ...
def make_card(rank: int, suit: int) -> int: ...
def card_to_str(card_id: int) -> str: ...
def str_to_card(s: str) -> int: ...
```

## keep_choices.py

```python
KEEP_CHOICES = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4),
    (3, 4),
]

def keep_idx_to_positions(idx: int) -> tuple[int, int]: ...
def discard_positions_from_keep_idx(idx: int) -> tuple[int, int, int]: ...
```

## canonicalize.py

```python
def canonicalize_flop_state(my5: tuple[int, ...], flop3: tuple[int, ...]) -> CanonicalizedState:
    ...
```

## equity.py

```python
def keep_equity(my5, flop3, keep_idx) -> float:
    ...
```

## flop_table.py

```python
class FlopDiscardTable:
    def __init__(self, table_path, index_path, meta_path): ...
    def choose_keep(self, my5, flop3): ...
    def choose_discard(self, my5, flop3): ...
```

---

# Simple SB behavior for now

In live play:

## BB

* use table directly

## SB

* also use the same table directly
* ignore opponent revealed discards in V1

Optional tiny improvement:

* if `equity_gap` is large, trust BB best immediately
* if `equity_gap` is small, log the state for future SB-improvement work

This helps identify where SB conditioning matters most.

---

# Future improvements (do not implement now unless easy)

1. SB top-2 re-evaluation using opponent revealed discards
2. direct dense ranking index without dictionary
3. stronger compression
4. stronger equity model including future betting
5. Monte Carlo fallback for missing states
6. training a small SB correction model

---

# Implementation priorities

If time is limited, do things in this order:

1. correct hand evaluator
2. correct canonicalization
3. correct keep-choice indexing
4. correct table lookup path
5. approximate/offline equity generation
6. speed optimization

Correctness matters more than squeezing maximum offline speed immediately.

---

# Final deliverables

Codex should produce:

1. source code implementing all modules above
2. tests
3. generation script
4. runtime lookup class
5. metadata spec
6. example script showing:

   * table generation on a small subset
   * loading the table
   * querying a random flop state

---

# Acceptance criteria

This project is complete when:

1. given `(my5, flop3)`, the code returns a legal keep-2 decision
2. the decision is obtained by table lookup, not live rollout
3. table records match offline evaluator outputs
4. canonicalization is correct and stable
5. binary format is compact and documented
6. loader works in the bot environment

---

# Notes to Codex

* Prefer clean, deterministic Python over premature micro-optimization
* Use type hints
* Add docstrings
* Add tests early
* Build a small-subset generation mode first for validation
* Avoid giant Python object structures for final storage
* Use raw integer card representations end-to-end
* Be very careful with keep-index mapping after canonicalization

The most error-prone parts are:

* hand ranking correctness
* ace-low / ace-high straight handling
* canonicalization mappings
* interpreting stored keep indices correctly

```

If you want, I can also turn this into a **shorter, more Codex-task-oriented version** with explicit TODO checklists and file-by-file implementation instructions.
```
