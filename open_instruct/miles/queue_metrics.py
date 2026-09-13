"""Bounded, windowed accounting of completed-queue work; no retained sample data."""

from collections import Counter

LENGTH_BINS = (
    (256, "0_255"),
    (512, "256_511"),
    (1024, "512_1023"),
    (2048, "1024_2047"),
    (4096, "2048_4095"),
    (8192, "4096_8191"),
    (16384, "8192_16383"),
)
AGE_BINS = ((1, "0"), (2, "1"), (3, "2"), (4, "3"), (5, "4"), (9, "5_8"), (17, "9_16"))


def bucket(value, bins, overflow):
    return next((label for upper, label in bins if value < upper), overflow)


class QueueMetrics:
    def __init__(self):
        self.counts = Counter()

    def record(self, lengths, *, age, accepted):
        outcome = "delivered" if accepted else "dropped"
        self.counts[f"{outcome}_groups"] += 1
        age_bin = "unknown" if age is None else bucket(age, AGE_BINS, "17_plus")
        for length in lengths:
            length_bin = bucket(length, LENGTH_BINS, "16384_plus")
            self.counts[f"{outcome}_samples"] += 1
            self.counts[f"{outcome}_response_tokens"] += length
            self.counts[f"{outcome}_samples_by_length/{length_bin}"] += 1
            self.counts[f"{outcome}_samples_by_age/{age_bin}"] += 1

    def collect(self):
        counts, self.counts = self.counts, Counter()
        result = {}
        for unit in ("groups", "samples", "response_tokens"):
            dropped, delivered = counts[f"dropped_{unit}"], counts[f"delivered_{unit}"]
            result[f"dropped_{unit}"] = dropped
            result[f"delivered_{unit}"] = delivered
            result[f"dropped_{unit}_fraction"] = dropped / max(1, dropped + delivered)
        for axis, labels in (
            ("length", [label for _, label in LENGTH_BINS] + ["16384_plus"]),
            ("age", [label for _, label in AGE_BINS] + ["17_plus", "unknown"]),
        ):
            for label in labels:
                suffix = f"samples_by_{axis}/{label}"
                dropped, delivered = counts[f"dropped_{suffix}"], counts[f"delivered_{suffix}"]
                result[f"dropped_{suffix}"] = dropped
                result[f"delivered_{suffix}"] = delivered
                result[f"dropped_samples_fraction_by_{axis}/{label}"] = dropped / max(1, dropped + delivered)
        return {f"rollout/fully_async/completed_queue/{key}": value for key, value in result.items()}
