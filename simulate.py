"""Monte Carlo experiment for the interactive NPS threshold paper.

The simulation is deliberately synthetic: it tests the arithmetic incentives created
by NPS, not whether any particular real-world intervention will work. Re-run with:

    python simulate.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np


SEED = 20_260_809
SCENARIOS = 12_000
CUSTOMERS = 20_000
CATEGORIES = 12
SCORES = np.arange(11)

STRATEGIES = (
    "edge_count",
    "edge_mean_volume_narrow",
    "edge_mean_volume",
    "edge_mean_volume_wide",
    "category_volume",
    "lower_edge_count",
    "upper_edge_count",
    "detractor_volume",
    "severe_tail",
    "worst_mean",
    "tight_edge",
    "loose_edge",
    "lowest_sd",
    "highest_sd",
    "random",
)

REGIMES = (
    "equal_response",
    "hidden_response",
    "tight_easier",
    "loose_easier",
)

ENHANCEMENT_RATES = {
    "small": 0.15,
    "moderate": 0.35,
    "strong": 0.60,
}

EDGE_STRATEGIES = ("edge_count", "lower_edge_count", "upper_edge_count")
EFFORT_BUDGETS = tuple(range(1, 7))
MAX_EFFORT_BUDGET = max(EFFORT_BUDGETS)
EFFORT_RESPONSE_RATE = ENHANCEMENT_RATES["moderate"]
EFFORT_REGIMES = REGIMES


def discrete_normal(mu: float, sigma: float) -> np.ndarray:
    """Probability mass over integer scores 0..10, with clipped-tail behavior."""
    weights = np.exp(-0.5 * ((SCORES - mu) / sigma) ** 2)
    return weights / weights.sum()


def observed_stats(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sizes = counts.sum(axis=1)
    means = (counts * SCORES).sum(axis=1) / np.maximum(sizes, 1)
    variances = (counts * (SCORES - means[:, None]) ** 2).sum(axis=1) / np.maximum(sizes, 1)
    return means, np.sqrt(variances)


def select_categories(
    counts: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, int]:
    sizes = counts.sum(axis=1)
    edge_distance = np.minimum(np.abs(means - 6), np.abs(means - 8))
    edge_closeness_narrow = np.exp(-0.5 * (edge_distance / 0.35) ** 2)
    edge_closeness = np.exp(-0.5 * (edge_distance / 0.65) ** 2)
    edge_closeness_wide = np.exp(-0.5 * (edge_distance / 1.00) ** 2)

    return {
        "edge_count": int(np.argmax(counts[:, 6] + counts[:, 8])),
        # A plausible shortcut: favor a large category whose average score sits
        # near either NPS edge. Unlike edge_count, this can hide the actual score
        # distribution behind the average.
        "edge_mean_volume_narrow": int(np.argmax(sizes * edge_closeness_narrow)),
        "edge_mean_volume": int(np.argmax(sizes * edge_closeness)),
        "edge_mean_volume_wide": int(np.argmax(sizes * edge_closeness_wide)),
        "category_volume": int(np.argmax(sizes)),
        "lower_edge_count": int(np.argmax(counts[:, 6])),
        "upper_edge_count": int(np.argmax(counts[:, 8])),
        "detractor_volume": int(np.argmax(counts[:, :7].sum(axis=1))),
        "severe_tail": int(np.argmax(counts[:, :3].sum(axis=1))),
        "worst_mean": int(np.argmin(means)),
        "tight_edge": int(np.argmax(sizes * edge_closeness / (stds + 0.25))),
        "loose_edge": int(np.argmax(sizes * edge_closeness * (stds + 0.25))),
        "lowest_sd": int(np.argmin(stds)),
        "highest_sd": int(np.argmax(stds)),
        "random": int(rng.integers(0, CATEGORIES)),
    }


def response_probabilities(
    regime: str,
    stds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if regime == "equal_response":
        return np.full(CATEGORIES, ENHANCEMENT_RATES["moderate"])
    if regime == "hidden_response":
        return np.clip(rng.beta(4, 7, size=CATEGORIES), 0.08, 0.78)

    z = (stds - stds.mean()) / (stds.std() + 1e-9)
    noise = rng.normal(0, 0.035, size=CATEGORIES)
    direction = -1 if regime == "tight_easier" else 1
    return np.clip(ENHANCEMENT_RATES["moderate"] + direction * 0.105 * z + noise, 0.08, 0.78)


def baseline_band(nps: float) -> str:
    if nps < 0:
        return "below_0"
    if nps < 40:
        return "0_to_40"
    if nps < 70:
        return "40_to_70"
    if nps < 85:
        return "70_to_85"
    return "85_plus"


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p05": round(float(np.quantile(a, 0.05)), 4),
        "p95": round(float(np.quantile(a, 0.95)), 4),
    }


def summarize_paired_difference(
    first: list[float],
    second: list[float],
    comparisons: int = 1,
) -> dict[str, float]:
    """Paired mean difference with pointwise and simultaneous 95% intervals."""
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if (
        first_values.ndim != 1
        or first_values.shape != second_values.shape
        or first_values.size < 2
        or comparisons < 1
    ):
        raise ValueError(
            "Paired results must be one-dimensional, equally sized, and contain at least two scenarios."
        )
    difference = first_values - second_values
    standard_error = float(difference.std(ddof=1) / math.sqrt(len(difference)))
    mean = float(difference.mean())
    pointwise_z = NormalDist().inv_cdf(0.975)
    simultaneous_z = NormalDist().inv_cdf(1 - 0.05 / (2 * comparisons))
    return {
        "mean_difference": round(mean, 4),
        "standard_error": round(standard_error, 4),
        "pointwise_ci95_low": round(mean - pointwise_z * standard_error, 4),
        "pointwise_ci95_high": round(mean + pointwise_z * standard_error, 4),
        "simultaneous_ci95_low": round(mean - simultaneous_z * standard_error, 4),
        "simultaneous_ci95_high": round(mean + simultaneous_z * standard_error, 4),
        "comparisons_adjusted_for": comparisons,
        "first_higher_pct": round(100 * float((difference > 0).mean()), 3),
        "equal_pct": round(100 * float((difference == 0).mean()), 3),
        "paired_scenarios": int(len(difference)),
    }


def nps_class_values() -> np.ndarray:
    """NPS contribution per customer: detractor=-1, passive=0, promoter=1."""
    return np.where(SCORES <= 6, -1, np.where(SCORES <= 8, 0, 1))


def effort_gain_table(counts: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Expected NPS-point gain for every supported effort level and category."""
    before = nps_class_values()
    gains = np.zeros((CATEGORIES, MAX_EFFORT_BUDGET + 1), dtype=float)
    for effort in range(1, MAX_EFFORT_BUDGET + 1):
        after_scores = np.minimum(SCORES + effort, 10)
        after = np.where(after_scores <= 6, -1, np.where(after_scores <= 8, 0, 1))
        gains[:, effort] = (
            100
            * response
            * (counts * (after - before)).sum(axis=1)
            / CUSTOMERS
        )
    return gains


def optimize_effort_plans(
    gains: np.ndarray,
    forced_category: int | None = None,
) -> dict[int, np.ndarray]:
    """Find the best allocation for every budget in one dynamic-programming pass."""
    states: dict[int, tuple[float, np.ndarray]] = {
        0: (0.0, np.zeros(CATEGORIES, dtype=np.int8))
    }
    for category in range(CATEGORIES):
        minimum = 1 if category == forced_category else 0
        next_states: dict[int, tuple[float, np.ndarray]] = {}
        for used, (value, plan) in states.items():
            for effort in range(minimum, MAX_EFFORT_BUDGET - used + 1):
                total_used = used + effort
                candidate = value + gains[category, effort]
                if total_used not in next_states or candidate > next_states[total_used][0]:
                    next_plan = plan.copy()
                    next_plan[category] = effort
                    next_states[total_used] = (float(candidate), next_plan)
        states = next_states
    return {
        budget: states[budget][1]
        for budget in EFFORT_BUDGETS
    }


def select_effort_plans_by_budget(
    counts: np.ndarray,
    response: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """Return the NPS-optimal and severe-protected plan for budgets 1..6."""
    gains = effort_gain_table(counts, response)
    severe_category = int(np.argmax(counts[:, :3].sum(axis=1)))
    quick = optimize_effort_plans(gains)
    balanced = optimize_effort_plans(gains, severe_category)
    return {
        budget: {
            "quick": quick[budget],
            "balanced": balanced[budget],
        }
        for budget in EFFORT_BUDGETS
    }


def allocation_pattern(plan: np.ndarray) -> str:
    """Human-readable split such as '2+1+1'."""
    positive = sorted((int(value) for value in plan if value > 0), reverse=True)
    return "+".join(str(value) for value in positive)


def realize_effort_plan(
    counts: np.ndarray,
    plan: np.ndarray,
    severe_total: int,
    response: np.ndarray,
) -> tuple[float, float]:
    """Expected result: one effort unit buys at most one point per responder."""
    before = nps_class_values()
    nps_units = 0.0
    severe_helped = 0.0
    for category, effort_value in enumerate(plan):
        effort = int(effort_value)
        if effort == 0:
            continue
        after_scores = np.minimum(SCORES + effort, 10)
        after = np.where(after_scores <= 6, -1, np.where(after_scores <= 8, 0, 1))
        nps_units += float(response[category] * (counts[category] * (after - before)).sum())
        severe_helped += float(response[category] * counts[category, :3].sum())
    return 100 * nps_units / CUSTOMERS, 100 * severe_helped / max(severe_total, 1)


def run() -> dict:
    rng = np.random.default_rng(SEED)
    trial_trace: list[dict[str, object]] = []
    metrics: dict[str, dict[str, dict[str, list[float]]]] = {
        regime: {
            strategy: defaultdict(list)  # type: ignore[arg-type]
            for strategy in STRATEGIES
        }
        for regime in REGIMES
    }
    bands: dict[str, dict[str, dict[str, list[float]]]] = {
        regime: {
            strategy: defaultdict(list)  # type: ignore[arg-type]
            for strategy in STRATEGIES
        }
        for regime in REGIMES
    }
    band_counts: dict[str, int] = defaultdict(int)
    baseline_values: list[float] = []
    edge_by_strength: dict[str, dict[str, dict[str, list[float]]]] = {
        strength: {
            strategy: defaultdict(list)  # type: ignore[arg-type]
            for strategy in EDGE_STRATEGIES
        }
        for strength in ENHANCEMENT_RATES
    }
    effort_metrics: dict[str, dict[int, dict[str, list[float]]]] = {
        regime: {
            budget: defaultdict(list)  # type: ignore[arg-type]
            for budget in EFFORT_BUDGETS
        }
        for regime in EFFORT_REGIMES
    }
    effort_pattern_counts: dict[str, dict[int, dict[str, int]]] = {
        regime: {
            budget: defaultdict(int)
            for budget in EFFORT_BUDGETS
        }
        for regime in EFFORT_REGIMES
    }

    for scenario_index in range(SCENARIOS):
        # A scenario is a company with twelve issue categories. The quality range is
        # intentionally broad so the results span low to ceiling-level NPS.
        global_mean = rng.uniform(4.4, 12.25)
        category_means = np.clip(global_mean + rng.normal(0, 0.82, CATEGORIES), -0.5, 13.0)
        category_stds = np.clip(
            np.exp(rng.normal(math.log(1.15), 0.43, CATEGORIES)), 0.42, 3.15
        )
        shares = rng.dirichlet(np.full(CATEGORIES, 2.0))
        sizes = rng.multinomial(CUSTOMERS, shares)

        counts = np.zeros((CATEGORIES, 11), dtype=np.int64)
        for category in range(CATEGORIES):
            probs = discrete_normal(category_means[category], category_stds[category])

            # Some categories are mixtures of two experiences. This makes category
            # means an imperfect shortcut and stress-tests standard-deviation rules.
            if rng.random() < 0.24:
                separation = rng.uniform(0.8, 2.4)
                weight = rng.uniform(0.28, 0.72)
                probs = (
                    weight * discrete_normal(category_means[category] - separation, category_stds[category])
                    + (1 - weight)
                    * discrete_normal(category_means[category] + separation, category_stds[category])
                )

            # A small common-noise component prevents perfectly clean synthetic tails.
            probs = 0.992 * probs + 0.008 / 11
            counts[category] = rng.multinomial(sizes[category], probs / probs.sum())

        means, stds = observed_stats(counts)
        chosen = select_categories(counts, means, stds, rng)

        total_by_score = counts.sum(axis=0)
        baseline_nps = 100 * (
            total_by_score[9:].sum() - total_by_score[:7].sum()
        ) / CUSTOMERS
        band = baseline_band(float(baseline_nps))
        baseline_values.append(float(baseline_nps))
        band_counts[band] += 1
        severe_total = max(int(total_by_score[:3].sum()), 1)

        # The enhancement experiment never "fully fixes" an issue. Depending on
        # strength, only 15%, 35%, or 60% of customers in the selected category
        # receive a one-point improvement. Lower- and upper-edge selection differ
        # only in how they choose the category; a category repair can create both
        # 6->7 and 8->9 crossings.
        for strength, rate in ENHANCEMENT_RATES.items():
            lower_crossings = rng.binomial(counts[:, 6], rate)
            upper_crossings = rng.binomial(counts[:, 8], rate)
            category_lift = 100 * (lower_crossings + upper_crossings) / CUSTOMERS
            for strategy in EDGE_STRATEGIES:
                category = chosen[strategy]
                edge_by_strength[strength][strategy][band].append(float(category_lift[category]))

        for regime in REGIMES:
            q = response_probabilities(regime, stds, rng)

            # Budget experiment under the same four ease-of-movement models.
            # For every budget from 1 to 6, dynamic programming checks every
            # possible integer allocation across the twelve categories.
            if regime in EFFORT_REGIMES:
                effort_plans = select_effort_plans_by_budget(counts, q)
                for budget, plans in effort_plans.items():
                    quick_nps, quick_severe = realize_effort_plan(
                        counts, plans["quick"], severe_total, q
                    )
                    balanced_nps, balanced_severe = realize_effort_plan(
                        counts, plans["balanced"], severe_total, q
                    )
                    effort_metrics[regime][budget]["quick_nps_lift"].append(quick_nps)
                    effort_metrics[regime][budget]["quick_severe_reach_pct"].append(quick_severe)
                    effort_metrics[regime][budget]["balanced_nps_lift"].append(balanced_nps)
                    effort_metrics[regime][budget]["balanced_severe_reach_pct"].append(balanced_severe)
                    effort_pattern_counts[regime][budget][allocation_pattern(plans["quick"])] += 1

            successes = np.zeros_like(counts)
            for category in range(CATEGORIES):
                successes[category, :10] = rng.binomial(counts[category, :10], q[category])

            category_nps_lift = 100 * (successes[:, 6] + successes[:, 8]) / CUSTOMERS
            category_mean_lift = successes.sum(axis=1) / CUSTOMERS
            category_severe_reach = 100 * successes[:, :3].sum(axis=1) / severe_total
            oracle_lift = max(float(category_nps_lift.max()), 1e-12)
            oracle_category = int(np.argmax(category_nps_lift))

            if regime == "equal_response" and scenario_index < 240:
                trial_trace.append({
                    "trial": scenario_index + 1,
                    "baseline_nps": round(float(baseline_nps), 3),
                    "oracle": round(float(category_nps_lift[oracle_category]), 3),
                    "lifts": {
                        strategy: round(float(category_nps_lift[category]), 3)
                        for strategy, category in chosen.items()
                        if strategy != "random"
                    },
                    "selected_categories": {
                        strategy: category + 1
                        for strategy, category in chosen.items()
                        if strategy != "random"
                    },
                })

            for strategy, category in chosen.items():
                m = metrics[regime][strategy]
                b = bands[regime][strategy]
                nps_lift = float(category_nps_lift[category])
                mean_lift = float(category_mean_lift[category])
                severe_reach = float(category_severe_reach[category])
                efficiency = nps_lift / oracle_lift
                oracle_match = float(category == oracle_category)

                m["nps_lift"].append(nps_lift)
                m["mean_score_lift"].append(mean_lift)
                m["severe_reach_pct"].append(severe_reach)
                m["oracle_efficiency"].append(efficiency)
                m["oracle_match"].append(oracle_match)

                b[f"{band}|nps_lift"].append(nps_lift)
                b[f"{band}|oracle_efficiency"].append(efficiency)

    result_metrics: dict[str, dict[str, dict]] = {}
    result_bands: dict[str, dict[str, dict]] = {}
    result_edge_by_strength: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for regime in REGIMES:
        result_metrics[regime] = {}
        result_bands[regime] = {}
        for strategy in STRATEGIES:
            result_metrics[regime][strategy] = {
                metric: summarize(values)
                for metric, values in metrics[regime][strategy].items()
            }
            result_bands[regime][strategy] = {}
            for key, values in bands[regime][strategy].items():
                band, metric = key.split("|")
                result_bands[regime][strategy].setdefault(band, {})[metric] = summarize(values)

    for strength in ENHANCEMENT_RATES:
        result_edge_by_strength[strength] = {}
        for strategy in EDGE_STRATEGIES:
            result_edge_by_strength[strength][strategy] = {
                band: summarize(values)
                for band, values in edge_by_strength[strength][strategy].items()
            }

    paired_comparisons = {
        "loose_easier_average_wide_minus_edge_count": summarize_paired_difference(
            metrics["loose_easier"]["edge_mean_volume_wide"]["nps_lift"],
            metrics["loose_easier"]["edge_count"]["nps_lift"],
            comparisons=len(STRATEGIES) - 1,
        )
    }

    result_effort_by_regime = {}
    for regime in EFFORT_REGIMES:
        budgets = {}
        for budget in EFFORT_BUDGETS:
            pattern_counts = effort_pattern_counts[regime][budget]
            pattern_shares = {
                pattern: round(100 * count / SCENARIOS, 3)
                for pattern, count in pattern_counts.items()
            }
            most_common_pattern = max(pattern_counts, key=pattern_counts.get)
            budgets[str(budget)] = {
                metric: summarize(values)
                for metric, values in effort_metrics[regime][budget].items()
            }
            budgets[str(budget)]["allocation_pattern_share_pct"] = pattern_shares
            budgets[str(budget)]["most_common_allocation"] = most_common_pattern
            budgets[str(budget)]["most_common_allocation_share_pct"] = round(
                100 * pattern_counts[most_common_pattern] / SCENARIOS, 3
            )
        result_effort_by_regime[regime] = {"budgets": budgets}

    # Exact arithmetic: one existing 6->7 or 8->9 conversion changes NPS by 100/n,
    # regardless of baseline. The fraction of remaining headroom consumed by +1 does
    # rise toward the ceiling.
    headroom = [
        {
            "nps": nps,
            "sample_share_for_plus_one_pct": 1.0,
            "remaining_opportunity_used_pct": round(100 / (100 - nps), 2),
        }
        for nps in (0, 40, 70, 80, 90, 95)
    ]

    return {
        "meta": {
            "seed": SEED,
            "scenarios": SCENARIOS,
            "customers_per_scenario": CUSTOMERS,
            "synthetic_customer_profiles": SCENARIOS * CUSTOMERS,
            "categories_per_scenario": CATEGORIES,
            "intervention": "Each affected customer has a category-specific probability of a one-point score increase; business effort per improvement step is held equal.",
            "information_sets": "One-step focus rules use only observed score summaries and do not receive generated hidden success chances. Multi-step sensitivity plans receive the exact generated category-specific chances and therefore represent perfect-information upper bounds.",
            "enhancement_rates": ENHANCEMENT_RATES,
            "effort_budgets": list(EFFORT_BUDGETS),
            "effort_response_rate": EFFORT_RESPONSE_RATE,
            "response_regime_assumptions": {
                "equal_response": "Every category is equally easy to move.",
                "hidden_response": "Ease of movement is drawn independently and randomly for every category.",
                "tight_easier": "Lower-SD categories are easier and higher-SD categories are harder to move.",
                "loose_easier": "Higher-SD categories are easier and lower-SD categories are harder to move.",
            },
            "strategy_definitions": {
                "edge_count": "Select the category with the largest observed count of scores 6 and 8 combined.",
                "edge_mean_volume_narrow": "Select a large category whose observed average is close to score 6 or 8; category size is weighted by a narrow closeness bandwidth of 0.35 score points.",
                "edge_mean_volume": "Select a large category whose observed average is close to score 6 or 8; category size is weighted by a middle closeness bandwidth of 0.65 score points.",
                "edge_mean_volume_wide": "Select a large category whose observed average is close to score 6 or 8; category size is weighted by a wide closeness bandwidth of 1.00 score point.",
                "category_volume": "Select the category linked to the largest total number of responses across scores 0 to 10.",
                "tight_edge": "Select an edge-near category while favoring lower observed standard deviation.",
                "loose_edge": "Select an edge-near category while favoring higher observed standard deviation.",
                "lowest_sd": "Select the category with the lowest observed score standard deviation, regardless of its average.",
                "highest_sd": "Select the category with the highest observed score standard deviation, regardless of its average.",
            },
        },
        "baseline_nps": summarize(baseline_values),
        "baseline_band_counts": dict(band_counts),
        "trial_trace": trial_trace,
        "metrics": result_metrics,
        "paired_comparisons": paired_comparisons,
        "by_baseline_band": result_bands,
        "edge_by_starting_nps_and_enhancement": result_edge_by_strength,
        "effort_capacity_experiment": {
            "assumption": "One improvement step creates at most one score-point increase for each responsive customer; totals from one to six steps are tested and business effort per step is equal.",
            "planner_information": "For the three category-specific sensitivity regimes, the optimizer receives each exact generated category success probability without estimation error. These are perfect-information upper bounds. The equal-response regime is a neutral baseline when category-specific probabilities are unknown, not evidence that real probabilities are equal.",
            "response_regimes": result_effort_by_regime,
        },
        "headroom": headroom,
        "exact_checks": {
            "one_conversion_nps_points_at_n_100": 1,
            "one_conversion_nps_points_at_n_1000": 0.1,
            "fifty_promoters_plus_one_detractor_nps": round(100 * 50 / 51 - 100 / 51, 4),
            "equal_cost_equal_response_rule": "For one-point moves with equal business effort per step and equal response probability, expected immediate NPS gain is maximized exactly by the largest count of scores 6 and 8 combined.",
            "category_gain_per_cost_formula": "(100 / total respondents) * response probability * (count at 6 + count at 8) / business effort per improvement step",
        },
    }


if __name__ == "__main__":
    output = run()
    path = Path(__file__).with_name("results.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    js_path = Path(__file__).with_name("results.js")
    js_path.write_text(
        "window.NPS_RESULTS = " + json.dumps(output, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {path}")
    print(f"Wrote {js_path}")
    print(json.dumps(output["meta"], indent=2))
