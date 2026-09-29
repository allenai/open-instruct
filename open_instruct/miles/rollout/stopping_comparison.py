"""CPU-only estimators and scheduling for the comparative stopping experiment."""

import math
import statistics


def comparison(stop_rewards, continue_rewards, stop_lengths, continue_lengths, *, tie_bonus=0.0, tie_min_accuracy=0.5):
    """An exact empirical tie is an experimental preference, not certified equality."""
    if not stop_rewards or len(stop_rewards) != len(continue_rewards):
        raise ValueError("Stopping comparisons require equally sized, nonempty groups")
    if len(stop_lengths) != len(stop_rewards) or len(continue_lengths) != len(continue_rewards):
        raise ValueError("Every comparison sample must have a generated length")
    if any(r not in (0, 1) for r in [*stop_rewards, *continue_rewards]):
        raise ValueError("Comparative stopping currently requires binary rewards")
    stop, continuation = statistics.mean(stop_rewards), statistics.mean(continue_rewards)
    saved = statistics.mean(continue_lengths) - statistics.mean(stop_lengths)
    tied = sum(stop_rewards) == sum(continue_rewards)
    bonus = tie_bonus if tied and stop >= tie_min_accuracy and saved > 0 else 0.0
    return {
        "stop_value": stop,
        "continue_value": continuation,
        "accuracy_advantage": stop - continuation,
        "tie_bonus": bonus,
        "advantage": stop - continuation + bonus,
        "tokens_saved": saved,
    }


def centered_rewards(rewards):
    """Within-action answer advantages; never mix these with natural prompt groups."""
    mean = statistics.mean(rewards)
    return [reward - mean for reward in rewards]


def parent_weights(lengths, uniform_share, length_cap):
    lengths = [min(max(1, length), length_cap) for length in lengths]
    total = sum(lengths)
    return [uniform_share / len(lengths) + (1 - uniform_share) * length / total for length in lengths]


def tapered_rate(step, initial, interval, floor_fraction):
    return initial * max(floor_fraction, 2.0 ** -(step // interval))


def risk_detected(deltas, *, minimum_parents, margin):
    """Parent-level one-sided normal screen; affects acquisition, never cut labels."""
    if len(deltas) < minimum_parents:
        return False
    error = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return statistics.mean(deltas) + 1.645 * error < -margin


def advance_rate(rate, completed_step, *, initial, interval, floor_fraction, risk):
    if risk:
        return min(initial, 2 * rate)
    if completed_step % interval == 0:
        return max(initial * floor_fraction, rate / 2)
    return rate
