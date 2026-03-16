"""
Evaluate match logs and compare baseline vs candidate strategy runs.

Usage examples:

  python evaluate_engine.py --candidate-glob "logs/candidate/*.csv"
  python evaluate_engine.py --baseline-glob "logs/base/*.csv" --candidate-glob "logs/candidate/*.csv" --check-thresholds
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MatchMetrics:
    path: str
    hands: int
    final_bankroll_t0: int
    chips_per_hand: float
    fold_to_preflop_raise: float
    preflop_raise_faced: int
    preflop_3bet_rate: float
    preflop_terminal_net: int
    showdown_net: int
    nonshowdown_net: int
    preflop_terminal_hands: int
    showdown_hands: int
    nonshowdown_hands: int


def _read_rows(path: str) -> list[dict[str, str]]:
    raw = Path(path).read_text().splitlines()
    rows = list(csv.DictReader(raw[1:])) if raw and raw[0].startswith("# Team ") else list(csv.DictReader(raw))
    return rows


def _per_hand_rows(rows: Iterable[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    result: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        hand_number = int(row["hand_number"])
        result[hand_number].append(row)
    return result


def compute_metrics(path: str) -> MatchMetrics:
    rows = _read_rows(path)
    hand_rows = _per_hand_rows(rows)
    if not hand_rows:
        raise ValueError(f"No hand rows found in {path}")

    sorted_hands = sorted(hand_rows)
    final_bankroll = int(hand_rows[sorted_hands[-1]][-1]["team_0_bankroll"])

    per_hand_delta: dict[int, int] = {}
    prev = 0
    for hand in sorted_hands:
        current = int(hand_rows[hand][-1]["team_0_bankroll"])
        per_hand_delta[hand] = current - prev
        prev = current

    faced_preflop_raise = 0
    defended_with_raise = 0
    folded_to_preflop_raise = 0

    preflop_terminal_net = 0
    preflop_terminal_hands = 0
    showdown_net = 0
    showdown_hands = 0
    nonshowdown_net = 0
    nonshowdown_hands = 0

    for hand in sorted_hands:
        hrows = hand_rows[hand]
        delta = per_hand_delta[hand]

        # Track how often Team 0 faces preflop raises and how it responds.
        preflop_rows = [r for r in hrows if r["street"] == "Pre-Flop"]
        first_raise_idx = None
        for idx, row in enumerate(preflop_rows):
            if int(row["active_team"]) == 1 and row["action_type"] == "RAISE":
                first_raise_idx = idx
                break
        if first_raise_idx is not None:
            for row in preflop_rows[first_raise_idx + 1 :]:
                if int(row["active_team"]) == 0:
                    faced_preflop_raise += 1
                    if row["action_type"] == "FOLD":
                        folded_to_preflop_raise += 1
                    if row["action_type"] == "RAISE":
                        defended_with_raise += 1
                    break

        folds = [r for r in hrows if r["action_type"] == "FOLD"]
        if folds and folds[-1]["street"] == "Pre-Flop":
            preflop_terminal_hands += 1
            preflop_terminal_net += delta
        if folds:
            nonshowdown_hands += 1
            nonshowdown_net += delta
        else:
            showdown_hands += 1
            showdown_net += delta

    hands = len(sorted_hands)
    return MatchMetrics(
        path=path,
        hands=hands,
        final_bankroll_t0=final_bankroll,
        chips_per_hand=final_bankroll / hands,
        fold_to_preflop_raise=(
            folded_to_preflop_raise / max(1, faced_preflop_raise)
        ),
        preflop_raise_faced=faced_preflop_raise,
        preflop_3bet_rate=(defended_with_raise / max(1, faced_preflop_raise)),
        preflop_terminal_net=preflop_terminal_net,
        showdown_net=showdown_net,
        nonshowdown_net=nonshowdown_net,
        preflop_terminal_hands=preflop_terminal_hands,
        showdown_hands=showdown_hands,
        nonshowdown_hands=nonshowdown_hands,
    )


def aggregate(metrics: list[MatchMetrics]) -> MatchMetrics:
    if not metrics:
        raise ValueError("No match logs to aggregate.")
    hands = sum(m.hands for m in metrics)
    faced = sum(m.preflop_raise_faced for m in metrics)
    weighted_fold = sum(
        m.fold_to_preflop_raise * m.preflop_raise_faced for m in metrics
    ) / max(1, faced)
    weighted_3bet = sum(
        m.preflop_3bet_rate * m.preflop_raise_faced for m in metrics
    ) / max(1, faced)
    final_bankroll = sum(m.final_bankroll_t0 for m in metrics)
    return MatchMetrics(
        path=f"{len(metrics)} files",
        hands=hands,
        final_bankroll_t0=final_bankroll,
        chips_per_hand=final_bankroll / max(1, hands),
        fold_to_preflop_raise=weighted_fold,
        preflop_raise_faced=faced,
        preflop_3bet_rate=weighted_3bet,
        preflop_terminal_net=sum(m.preflop_terminal_net for m in metrics),
        showdown_net=sum(m.showdown_net for m in metrics),
        nonshowdown_net=sum(m.nonshowdown_net for m in metrics),
        preflop_terminal_hands=sum(m.preflop_terminal_hands for m in metrics),
        showdown_hands=sum(m.showdown_hands for m in metrics),
        nonshowdown_hands=sum(m.nonshowdown_hands for m in metrics),
    )


def print_metrics(title: str, metric: MatchMetrics) -> None:
    print(f"\n== {title} ==")
    print(f"source: {metric.path}")
    print(f"hands: {metric.hands}")
    print(f"final_bankroll_t0: {metric.final_bankroll_t0}")
    print(f"chips_per_hand: {metric.chips_per_hand:.3f}")
    print(
        f"fold_to_preflop_raise: {metric.fold_to_preflop_raise:.3%} (faced={metric.preflop_raise_faced})"
    )
    print(f"preflop_3bet_rate: {metric.preflop_3bet_rate:.3%}")
    print(
        "net_by_path: "
        f"preflop_terminal={metric.preflop_terminal_net}, "
        f"showdown={metric.showdown_net}, "
        f"nonshowdown={metric.nonshowdown_net}"
    )


def compare_and_check(
    baseline: MatchMetrics,
    candidate: MatchMetrics,
    check_thresholds: bool,
) -> int:
    chips_improvement = candidate.chips_per_hand - baseline.chips_per_hand
    if baseline.preflop_terminal_net < 0:
        preflop_improvement_ratio = (
            (candidate.preflop_terminal_net - baseline.preflop_terminal_net)
            / abs(baseline.preflop_terminal_net)
        )
    else:
        preflop_improvement_ratio = 0.0

    print("\n== Delta Candidate - Baseline ==")
    print(f"chips_per_hand_delta: {chips_improvement:.3f}")
    print(
        "fold_to_preflop_raise_delta: "
        f"{candidate.fold_to_preflop_raise - baseline.fold_to_preflop_raise:+.3%}"
    )
    print(
        "preflop_terminal_net_delta: "
        f"{candidate.preflop_terminal_net - baseline.preflop_terminal_net:+d}"
    )
    print(f"preflop_terminal_improvement_ratio: {preflop_improvement_ratio:.3f}")

    if not check_thresholds:
        return 0

    checks = [
        (
            "fold_to_preflop_raise <= 70%",
            candidate.fold_to_preflop_raise <= 0.70,
        ),
        (
            "preflop_terminal_net improvement >= 40%",
            preflop_improvement_ratio >= 0.40,
        ),
        (
            "chips_per_hand improvement >= 1.0",
            chips_improvement >= 1.0,
        ),
    ]
    failed = [label for label, ok in checks if not ok]
    print("\n== Threshold Check ==")
    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if failed:
        print(f"\nthreshold_status: FAIL ({len(failed)} failed checks)")
        return 2
    print("\nthreshold_status: PASS")
    return 0


def expand_glob(pattern: str | None) -> list[str]:
    if not pattern:
        return []
    return sorted(glob.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate poker match logs and compare baseline/candidate."
    )
    parser.add_argument(
        "--candidate-glob",
        required=True,
        help="Glob for candidate CSV logs (e.g. logs/candidate/*.csv)",
    )
    parser.add_argument(
        "--baseline-glob",
        default=None,
        help="Optional glob for baseline CSV logs (e.g. logs/baseline/*.csv)",
    )
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Apply plan thresholds when baseline and candidate are provided.",
    )
    args = parser.parse_args()

    candidate_files = expand_glob(args.candidate_glob)
    if not candidate_files:
        raise SystemExit("No candidate files matched.")
    candidate_metrics = [compute_metrics(path) for path in candidate_files]
    candidate_agg = aggregate(candidate_metrics)
    print_metrics("Candidate Aggregate", candidate_agg)

    baseline_files = expand_glob(args.baseline_glob)
    if not baseline_files:
        return 0
    baseline_metrics = [compute_metrics(path) for path in baseline_files]
    baseline_agg = aggregate(baseline_metrics)
    print_metrics("Baseline Aggregate", baseline_agg)

    return compare_and_check(
        baseline=baseline_agg,
        candidate=candidate_agg,
        check_thresholds=args.check_thresholds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
