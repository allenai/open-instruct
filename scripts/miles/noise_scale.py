"""Exploratory Euclidean gradient-noise diagnostics, not calibrated critical batch sizes.

B_simple = tr(Sigma) / |G|^2 compares raw gradient noise and signal. Its relationship
to useful training batch size depends on curvature, optimizer and data distribution;
this module does not establish that relationship for Adam or changing-policy RL.

Prompt groups, rather than individual responses, are the sampling units. Independence
between groups is an assumption to check, not a consequence of grouping. The signal
and covariance-trace estimators are unbiased under iid sampling; their ratio is not.
Intervals and null-based diagnostics below are exploratory, especially near zero signal.
"""

import dataclasses
import math

import numpy as np

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@dataclasses.dataclass(frozen=True)
class Accumulation:
    """Sufficient statistics for one independent panel of prompt groups.

    `sum_squared_norms` is sum_i |g_i|^2 and `squared_sum_norm` is |sum_i g_i|^2, both
    over the same `groups` per-group gradients. Keeping them separate is what makes the
    signal and the noise separable: the first carries |G|^2 + tr(Sigma) per group, the
    second grows with |G|^2 alone once divided by the group count squared.
    """

    groups: int
    sum_squared_norms: float
    squared_sum_norm: float

    def __post_init__(self):
        if self.groups < 2:
            raise ValueError(f"A panel needs at least two groups to separate signal from noise; got {self.groups}")
        if self.sum_squared_norms < 0 or self.squared_sum_norm < 0:
            raise ValueError("Squared norms cannot be negative")

    def __add__(self, other):
        raise TypeError("Panels cannot be added from norms alone; pool with `pool` over the raw gradient sums")


def estimate(accumulation: Accumulation) -> dict:
    """Unbiased signal, noise and noise scale from one panel.

    With g_i independent, E|mean g|^2 = |G|^2 + tr(Sigma)/n, so the naive squared norm
    of the mean gradient overstates the signal by exactly one batch's worth of noise.
    Subtracting it is what stops the noise scale from collapsing as the panel grows.
    """
    n = accumulation.groups
    mean_squared = accumulation.squared_sum_norm / (n * n)
    trace_sigma = (accumulation.sum_squared_norms - accumulation.squared_sum_norm / n) / (n - 1)
    gradient_squared = mean_squared - trace_sigma / n
    scale = trace_sigma / gradient_squared if gradient_squared > 0 else math.inf
    return dict(
        groups=n,
        mean_squared=mean_squared,
        gradient_squared=gradient_squared,
        trace_sigma=trace_sigma,
        noise_scale=scale,
        signal_to_noise=gradient_squared * n / trace_sigma if trace_sigma > 0 else math.inf,
    )


def sketch_estimate(sketches: np.ndarray, *, dimension_ratio: float = 1.0) -> dict:
    """Noise scale from per-group gradients projected onto a coordinate subsample.

    `sketches` is [groups, coordinates] in float64-compatible dtype, one row per group.
    Sampling coordinates uniformly scales both tr(Sigma) and |G|^2 by the same factor,
    so their ratio is unbiased in the limit and `dimension_ratio` only restores the
    absolute magnitudes for comparison against the exact full-dimensional totals.
    """
    array = np.asarray(sketches, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Sketches must be [groups, coordinates]; found shape {array.shape}")
    if dimension_ratio <= 0:
        raise ValueError("The coordinate sampling ratio must be positive")
    accumulation = Accumulation(
        groups=array.shape[0],
        sum_squared_norms=float((array * array).sum()),
        squared_sum_norm=float((array.sum(axis=0) ** 2).sum()),
    )
    result = estimate(accumulation)
    for key in ("mean_squared", "gradient_squared", "trace_sigma"):
        result[key] = result[key] / dimension_ratio
    return result


def gram(vectors: np.ndarray) -> np.ndarray:
    """All pairwise inner products of the per-group gradients.

    Every quantity the noise scale needs from a subset of groups is a sum of entries of
    this matrix: the squared norm of their sum is the sum over the selected pairs, and the
    sum of their squared norms is the sum over the selected diagonal. Resampling therefore
    costs a few hundred thousand additions rather than a copy of the whole panel, which is
    what makes a thousand bootstrap draws affordable at this width.
    """
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Vectors must be [groups, coordinates]; found shape {array.shape}")
    return array @ array.T


def estimate_from_gram(matrix: np.ndarray, counts=None) -> dict:
    """Signal, noise and noise scale for a multiset of groups given by `counts`.

    `counts` is how many times each group appears, so a bootstrap draw is a count vector
    and the full panel is a vector of ones. Duplicated groups are counted as separate
    draws, which is exactly what resampling with replacement means here.
    """
    square = np.asarray(matrix, dtype=np.float64)
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError(f"The Gram matrix must be square; found shape {square.shape}")
    weights = np.ones(square.shape[0]) if counts is None else np.asarray(counts, dtype=np.float64)
    total = float(weights.sum())
    if total < 2:
        raise ValueError(f"At least two groups are needed to separate signal from noise; got {total}")
    return estimate(
        Accumulation(
            groups=int(round(total)),
            sum_squared_norms=float(weights @ np.diag(square)),
            squared_sum_norm=float(weights @ square @ weights),
        )
    )


def bootstrap(sketches: np.ndarray, *, draws: int = 2000, seed: int = 17) -> dict:
    """Resample groups with replacement to put an interval on the noise scale.

    Percentiles are conditional on finite positive ratios. Near zero signal, discarding
    unresolved draws can give misleading intervals; these are diagnostic percentiles,
    not a calibrated confidence interval. The resampling unit is the prompt group.
    """
    array = np.asarray(sketches, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Sketches must be [groups, coordinates]; found shape {array.shape}")
    generator = np.random.default_rng(seed)
    n = array.shape[0]
    matrix = gram(array)
    values = []
    for _ in range(draws):
        counts = np.bincount(generator.integers(0, n, size=n), minlength=n)
        try:
            values.append(estimate_from_gram(matrix, counts)["noise_scale"])
        except ValueError:
            continue
    finite = np.array([value for value in values if math.isfinite(value) and value > 0])
    if finite.size < draws // 2:
        logger.warning("Only %d of %d bootstrap draws gave a finite positive noise scale", finite.size, draws)
    if finite.size == 0:
        return dict(draws=0, low=math.nan, median=math.nan, high=math.nan)
    return dict(
        draws=int(finite.size),
        low=float(np.percentile(finite, 2.5)),
        median=float(np.percentile(finite, 50.0)),
        high=float(np.percentile(finite, 97.5)),
    )


def subsample_curve(sketches: np.ndarray, sizes, *, repeats: int = 64, seed: int = 17) -> list:
    """Noise scale re-estimated from fewer groups, to show the estimate has converged.

    The unbiased subtraction above needs a panel comparable to the noise scale itself:
    when the panel is much smaller, |G|^2 is a small difference of two large numbers and
    the estimate runs away. Plotting the estimate against panel size is how a reader
    tells a converged measurement from one that merely ran out of data.
    """
    array = np.asarray(sketches, dtype=np.float64)
    generator = np.random.default_rng(seed)
    matrix = gram(array)
    rows = []
    for size in sizes:
        if size < 2 or size > array.shape[0]:
            raise ValueError(f"Subsample size {size} is outside 2..{array.shape[0]}")
        values = []
        for _ in range(repeats):
            counts = np.zeros(array.shape[0])
            counts[generator.choice(array.shape[0], size=size, replace=False)] = 1.0
            value = estimate_from_gram(matrix, counts)["noise_scale"]
            if math.isfinite(value) and value > 0:
                values.append(value)
        rows.append(
            dict(
                groups=int(size),
                usable=len(values),
                median=float(np.median(values)) if values else math.nan,
                low=float(np.percentile(values, 10)) if values else math.nan,
                high=float(np.percentile(values, 90)) if values else math.nan,
            )
        )
    return rows


def efficiency(noise_scale: float, batch_sizes) -> list:
    """Predicted cost of each batch size, in updates and in examples.

    Under the model this noise scale comes from, training at batch B needs
    1 + B_simple/B times the minimum number of updates and 1 + B/B_simple times the
    minimum number of examples. The two curves cross at B_simple, which is why that
    point is the natural place to sit: either side trades one resource for the other at
    a worsening rate.
    """
    if not noise_scale > 0 or not math.isfinite(noise_scale):
        raise ValueError(f"The noise scale must be finite and positive; got {noise_scale}")
    rows = []
    for batch in batch_sizes:
        if batch <= 0:
            raise ValueError("Batch sizes must be positive")
        rows.append(
            dict(
                groups=batch,
                updates_vs_minimum=1.0 + noise_scale / batch,
                examples_vs_minimum=1.0 + batch / noise_scale,
            )
        )
    return rows


def before_filtering(noise_scale: float, drop_fraction: float) -> float:
    """The noise scale per sampled group, given the one measured on retained groups.

    An online filter that discards constant-reward groups removes groups whose gradient is
    exactly zero, so a sampled group is that zero with probability f and a retained group
    otherwise. Both the true gradient and the second moment scale by (1 - f), but the
    gradient enters squared, which leaves `(B_sampled + 1) = (B_retained + 1) / (1 - f)`.

    The distinction matters because the two answer different questions: how many groups to
    retain per update, and how many prompts to sample in order to retain them.
    """
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError(f"The discarded fraction must be in [0, 1); got {drop_fraction}")
    if not noise_scale > 0 or not math.isfinite(noise_scale):
        raise ValueError(f"The noise scale must be finite and positive; got {noise_scale}")
    return (noise_scale + 1.0) / (1.0 - drop_fraction) - 1.0


def batch_scaling(matrix: np.ndarray, sizes, *, repeats: int = 8, seed: int = 17) -> list:
    """How the squared norm of a batch gradient falls as the batch grows.

    For independent groups, E|mean gradient of B|^2 = |G|^2 + tr(Sigma)/B, so multiplying by
    B gives a straight line in B with intercept tr(Sigma) and slope |G|^2. Plotting that
    product is the assumption-light way to read the noise scale off a panel: while it is
    flat, the batch gradient is essentially all noise and the batch is far below the noise
    scale; where it starts to climb, the batch is approaching it.

    Batches are disjoint within a repeat, but repeats reuse the panel. Only a single
    partition with at least two batches gets a conventional sample standard error
    (assuming iid groups). Repeated partitions return no population standard error;
    averaging their values does not make them independent observations.
    """
    square = np.asarray(matrix, dtype=np.float64)
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError("Expected a square Gram matrix")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    generator = np.random.default_rng(seed)
    n = square.shape[0]
    rows = []
    for size in sizes:
        if size < 1 or size > n:
            continue
        values = []
        for _ in range(repeats):
            order = generator.permutation(n)
            for start in range(0, n - size + 1, size):
                block = order[start : start + size]
                values.append(square[np.ix_(block, block)].sum() / (size * size))
        mean = float(np.mean(values))
        rows.append(
            dict(
                batch=int(size),
                batches=len(values),
                mean_squared=mean,
                product=mean * size,
                standard_error=(
                    float(np.std(values, ddof=1) / math.sqrt(len(values))) * size
                    if repeats == 1 and len(values) > 1
                    else None
                ),
                independent_batches_per_partition=n // size,
                partitions=repeats,
            )
        )
    return rows


def jackknife(matrix: np.ndarray) -> dict:
    """Leave-one-group-out interval for the noise scale.

    This smooth-ratio approximation is unreliable near zero signal. Duplicated rows in
    an ordinary bootstrap can also make this diagnostic unstable. Neither approach
    certifies confidence coverage here; leave-one-out subsets are strongly dependent.

    The spread of the deleted estimates is not itself the uncertainty, since removing one
    group of many barely moves the answer; the jackknife scales it by the panel size. The
    arithmetic is done on the logarithm because the noise scale is a positive ratio whose
    error is multiplicative, which also keeps the interval from crossing zero.
    """
    square = np.asarray(matrix, dtype=np.float64)
    n = square.shape[0]
    full = estimate_from_gram(square)["noise_scale"]
    values = []
    for index in range(n):
        counts = np.ones(n)
        counts[index] = 0.0
        value = estimate_from_gram(square, counts)["noise_scale"]
        if math.isfinite(value) and value > 0:
            values.append(math.log(value))
    if len(values) < n:
        logger.warning("%d of %d leave-one-out panels gave no resolvable signal", n - len(values), n)
    if len(values) < n or not math.isfinite(full) or full <= 0:
        return dict(usable=len(values), estimate=full, low=math.nan, high=math.nan, log_standard_error=math.nan)
    array = np.array(values)
    error = math.sqrt((len(array) - 1) / len(array) * float(((array - array.mean()) ** 2).sum()))
    # Bias-corrected point estimate on the same logarithmic scale.
    corrected = math.exp(len(array) * math.log(full) - (len(array) - 1) * float(array.mean()))
    return dict(
        usable=len(values),
        estimate=corrected,
        low=math.exp(math.log(full) - 1.96 * error),
        high=math.exp(math.log(full) + 1.96 * error),
        log_standard_error=error,
    )


def lower_bound(accumulation_or_estimate: dict) -> float:
    """The noise scale one gets without correcting the signal for its own noise.

    Historical function name: this is NOT a bound on the population noise scale.
    The sample mean squared norm exceeds the true squared norm only in expectation;
    a realized sample may cancel. Retained for reproducing earlier diagnostic output.
    """
    mean_squared = accumulation_or_estimate["mean_squared"]
    if not mean_squared > 0:
        raise ValueError("The mean gradient has no squared norm to divide by")
    return accumulation_or_estimate["trace_sigma"] / mean_squared


def identifiability(matrix: np.ndarray, *, confidence: float = 1.645) -> dict:
    """Exploratory null-signal diagnostic; reported floors are not confidence bounds.

    The signal is the difference between the observed mean gradient's squared norm and the
    noise floor it would have if the true gradient were zero. That difference has its own
    sampling error, set by how concentrated the covariance is: `tr(Sigma)^2 / tr(Sigma^2)`
    is the effective number of directions the per-group noise occupies, and the fewer
    there are, the more the observed mean fluctuates.

    `confidence` is a Gaussian multiplier, not a guaranteed coverage probability. The
    null approximation omits the nonzero-mean variance term 4 G^T Sigma G / n and uses
    uncentered off-diagonal inner products. Numerator and sketch uncertainty are not
    accounted for. Legacy keys such as `noise_scale_floor` are retained to reproduce
    historical reports, with an explicit flag that confidence coverage is uncalibrated.
    """
    square = np.asarray(matrix, dtype=np.float64)
    n = square.shape[0]
    if n < 3:
        raise ValueError(f"At least three groups are needed to estimate the covariance spread; got {n}")
    estimated = estimate_from_gram(square)
    off_diagonal = square.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    squared_trace = float((off_diagonal**2).sum()) / (n * (n - 1))
    if squared_trace <= 0:
        raise ValueError("The per-group gradients are mutually orthogonal to numerical precision")
    deviation = math.sqrt(2 * squared_trace) / n
    limit = estimated["gradient_squared"] + confidence * deviation
    return dict(
        groups=n,
        effective_dimension=estimated["trace_sigma"] ** 2 / squared_trace,
        gradient_squared=estimated["gradient_squared"],
        null_deviation=deviation,
        sigma=estimated["gradient_squared"] / deviation,
        gradient_squared_limit=limit,
        noise_scale_floor=estimated["trace_sigma"] / limit if limit > 0 else math.inf,
        resolved=estimated["gradient_squared"] > confidence * deviation,
        confidence_calibrated=False,
    )
