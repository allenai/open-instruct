"""Noise-scale estimators: unbiasedness, the panel-size trap, and the efficiency curve."""

import math

import numpy as np
import pytest
from scripts.miles import noise_scale


def synthetic(groups, dimension, signal, noise, seed=17):
    """Per-group gradients with a known true gradient and known covariance trace.

    Every group is the same true gradient plus isotropic noise, so |G|^2 = signal^2 and
    tr(Sigma) = dimension * noise^2, which makes the target noise scale exact.
    """
    generator = np.random.default_rng(seed)
    direction = np.zeros(dimension)
    direction[0] = signal
    return direction + generator.normal(0.0, noise, size=(groups, dimension))


def test_estimate_recovers_a_known_noise_scale():
    dimension, signal, noise = 8, 1.0, 0.5
    target = dimension * noise**2 / signal**2
    values = [
        noise_scale.sketch_estimate(synthetic(4000, dimension, signal, noise, seed=seed))["noise_scale"]
        for seed in range(20)
    ]
    assert target == pytest.approx(float(np.mean(values)), rel=0.05)


def test_naive_mean_square_understates_the_noise_scale():
    """Skipping the bias correction inflates the signal, which is the whole trap."""
    sketches = synthetic(256, 8, 1.0, 0.5)
    result = noise_scale.sketch_estimate(sketches)
    naive = result["trace_sigma"] / result["mean_squared"]
    assert naive < result["noise_scale"]
    assert result["mean_squared"] > result["gradient_squared"]


def test_estimate_is_unchanged_by_scaling_every_gradient():
    """The noise scale is a ratio, so the loss normalization constant cannot move it."""
    sketches = synthetic(512, 8, 1.0, 0.5)
    plain = noise_scale.sketch_estimate(sketches)["noise_scale"]
    scaled = noise_scale.sketch_estimate(sketches * 37.0)["noise_scale"]
    assert plain == pytest.approx(scaled, rel=1e-9)


def spread_signal(groups, dimension, signal, noise, seed=17):
    """Gradients whose signal is spread over every coordinate, as a dense gradient is."""
    generator = np.random.default_rng(seed)
    direction = np.full(dimension, signal / math.sqrt(dimension))
    return direction + generator.normal(0.0, noise, size=(groups, dimension))


def test_random_coordinate_subsampling_preserves_the_ratio():
    """A uniformly random coordinate subset scales signal and noise alike."""
    generator = np.random.default_rng(3)
    sketches = spread_signal(4000, 256, 1.0, 0.5)
    full = noise_scale.sketch_estimate(sketches)
    columns = generator.choice(256, size=64, replace=False)
    part = noise_scale.sketch_estimate(sketches[:, columns], dimension_ratio=64 / 256)
    assert full["noise_scale"] == pytest.approx(part["noise_scale"], rel=0.2)
    assert full["trace_sigma"] == pytest.approx(part["trace_sigma"], rel=0.2)
    assert full["gradient_squared"] == pytest.approx(part["gradient_squared"], rel=0.3)


def test_a_subset_chosen_where_the_signal_lives_biases_the_ratio():
    """The failure mode the driver must avoid: coordinates picked non-randomly.

    Here all the signal sits in one coordinate and the noise is spread evenly, so
    keeping a leading block retains the whole signal and a quarter of the noise and
    understates the noise scale fourfold. Only uniform sampling over the full parameter
    vector makes the sketched ratio meaningful.
    """
    sketches = synthetic(2000, 64, 1.0, 0.5)
    full = noise_scale.sketch_estimate(sketches)["noise_scale"]
    biased = noise_scale.sketch_estimate(sketches[:, :16], dimension_ratio=16 / 64)["noise_scale"]
    assert biased == pytest.approx(full / 4, rel=0.2)


def test_accumulation_rejects_panels_too_small_to_separate_signal_from_noise():
    with pytest.raises(ValueError):
        noise_scale.Accumulation(groups=1, sum_squared_norms=1.0, squared_sum_norm=1.0)
    with pytest.raises(ValueError):
        noise_scale.Accumulation(groups=4, sum_squared_norms=-1.0, squared_sum_norm=1.0)
    with pytest.raises(TypeError):
        noise_scale.Accumulation(4, 1.0, 1.0) + noise_scale.Accumulation(4, 1.0, 1.0)


def test_estimate_matches_the_accumulated_scalars():
    sketches = synthetic(300, 8, 1.0, 0.5)
    accumulation = noise_scale.Accumulation(
        groups=sketches.shape[0],
        sum_squared_norms=float((sketches * sketches).sum()),
        squared_sum_norm=float((sketches.sum(axis=0) ** 2).sum()),
    )
    assert noise_scale.estimate(accumulation) == noise_scale.sketch_estimate(sketches)


def test_a_panel_far_below_the_noise_scale_is_visibly_unconverged():
    """Small panels must show instability rather than a confident wrong number."""
    sketches = synthetic(4000, 256, 1.0, 1.0)
    rows = noise_scale.subsample_curve(sketches, [8, 64, 2000], repeats=32)
    spread = [(row["high"] - row["low"]) / row["median"] for row in rows]
    assert spread[0] > spread[-1]
    assert rows[-1]["median"] == pytest.approx(256.0, rel=0.25)


def test_bootstrap_interval_covers_the_point_estimate():
    sketches = synthetic(1500, 16, 1.0, 0.5)
    point = noise_scale.sketch_estimate(sketches)["noise_scale"]
    interval = noise_scale.bootstrap(sketches, draws=400)
    assert interval["low"] < point < interval["high"]
    assert interval["draws"] > 300


def test_efficiency_curves_cross_at_the_noise_scale():
    rows = noise_scale.efficiency(100.0, [25, 100, 400])
    assert rows[1]["updates_vs_minimum"] == pytest.approx(rows[1]["examples_vs_minimum"])
    assert rows[0]["updates_vs_minimum"] > rows[2]["updates_vs_minimum"]
    assert rows[0]["examples_vs_minimum"] < rows[2]["examples_vs_minimum"]
    with pytest.raises(ValueError):
        noise_scale.efficiency(math.inf, [1])


def test_before_filtering_matches_a_panel_padded_with_discarded_groups():
    """Adding the zero-gradient groups back must give what the formula predicts."""
    kept = spread_signal(4000, 64, 1.0, 0.5)
    fraction = 0.4
    padding = int(round(kept.shape[0] * fraction / (1 - fraction)))
    sampled = np.vstack([kept, np.zeros((padding, kept.shape[1]))])
    retained_scale = noise_scale.sketch_estimate(kept)["noise_scale"]
    direct = noise_scale.sketch_estimate(sampled)["noise_scale"]
    predicted = noise_scale.before_filtering(retained_scale, padding / sampled.shape[0])
    assert predicted == pytest.approx(direct, rel=0.05)
    assert predicted > retained_scale


def test_before_filtering_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        noise_scale.before_filtering(10.0, 1.0)
    with pytest.raises(ValueError):
        noise_scale.before_filtering(10.0, -0.1)
    assert noise_scale.before_filtering(10.0, 0.0) == pytest.approx(10.0)


def test_gram_estimate_matches_the_direct_one():
    """Resampling through inner products must give exactly the direct computation."""
    sketches = spread_signal(300, 32, 1.0, 0.5)
    direct = noise_scale.sketch_estimate(sketches)
    through = noise_scale.estimate_from_gram(noise_scale.gram(sketches))
    for key in ("gradient_squared", "trace_sigma", "noise_scale"):
        assert direct[key] == pytest.approx(through[key], rel=1e-9)


def test_gram_counts_repeat_a_group_as_a_separate_draw():
    sketches = spread_signal(40, 8, 1.0, 0.5)
    matrix = noise_scale.gram(sketches)
    counts = np.ones(40)
    counts[0] = 2.0
    doubled = noise_scale.estimate_from_gram(matrix, counts)
    stacked = noise_scale.sketch_estimate(np.vstack([sketches, sketches[:1]]))
    assert doubled["noise_scale"] == pytest.approx(stacked["noise_scale"], rel=1e-9)
    with pytest.raises(ValueError):
        noise_scale.estimate_from_gram(matrix, np.zeros(40))


def test_batch_scaling_is_flat_below_the_noise_scale_and_climbs_above_it():
    """The product B times the mean squared batch gradient is a line in B."""
    dimension, noise = 64, 1.0
    sketches = spread_signal(4096, dimension, 1.0, noise)
    matrix = noise_scale.gram(sketches)
    rows = noise_scale.batch_scaling(matrix, [1, 8, 64, 512], repeats=4)
    target = dimension * noise**2
    assert rows[0]["product"] == pytest.approx(target, rel=0.15)
    # Intercept is the covariance trace; the climb from it is the signal times the batch.
    assert rows[-1]["product"] - rows[0]["product"] == pytest.approx(511 * 1.0, rel=0.4)
    assert rows[-1]["product"] > rows[0]["product"]


def test_lower_bound_never_exceeds_the_corrected_estimate():
    sketches = spread_signal(2000, 32, 1.0, 0.5)
    estimate = noise_scale.sketch_estimate(sketches)
    assert noise_scale.lower_bound(estimate) < estimate["noise_scale"]


def test_jackknife_is_not_pulled_down_the_way_resampling_with_replacement_is():
    """A duplicated group adds coherently, which is why bootstrapping this ratio fails."""
    sketches = spread_signal(400, 16, 1.0, 2.0)
    matrix = noise_scale.gram(sketches)
    point = noise_scale.estimate_from_gram(matrix)["noise_scale"]
    deleted = noise_scale.jackknife(matrix)
    with_replacement = noise_scale.bootstrap(sketches, draws=200)
    assert deleted["usable"] == matrix.shape[0]
    assert abs(math.log(deleted["estimate"] / point)) < abs(math.log(with_replacement["median"] / point))
    assert deleted["low"] < point < deleted["high"]


def test_identifiability_resolves_a_strong_signal_and_bounds_a_hidden_one():
    """A clear signal is reported as resolved; one buried in noise yields only a floor."""
    strong = noise_scale.gram(spread_signal(2000, 256, 1.0, 0.3))
    resolved = noise_scale.identifiability(strong)
    assert resolved["resolved"]
    assert resolved["sigma"] > 3
    assert resolved["noise_scale_floor"] < noise_scale.estimate_from_gram(strong)["noise_scale"]

    generator = np.random.default_rng(5)
    hidden = noise_scale.gram(generator.normal(0.0, 1.0, size=(600, 256)))
    bounded = noise_scale.identifiability(hidden)
    assert not bounded["resolved"]
    assert abs(bounded["sigma"]) < 3
    assert bounded["noise_scale_floor"] > 100


def test_effective_dimension_counts_the_directions_the_noise_occupies():
    """Noise confined to a few coordinates must report that few, not the full width."""
    generator = np.random.default_rng(7)
    narrow = np.zeros((900, 128))
    narrow[:, :4] = generator.normal(0.0, 1.0, size=(900, 4))
    wide = generator.normal(0.0, 1.0, size=(900, 128))
    assert noise_scale.identifiability(noise_scale.gram(narrow))["effective_dimension"] == pytest.approx(4, rel=0.3)
    assert noise_scale.identifiability(noise_scale.gram(wide))["effective_dimension"] == pytest.approx(128, rel=0.3)


def test_identifiability_needs_enough_groups():
    with pytest.raises(ValueError):
        noise_scale.identifiability(np.eye(2))


def test_repeated_partitions_do_not_claim_independent_sampling_error():
    matrix = np.diag([1.0, 4.0, 9.0, 16.0])
    single = noise_scale.batch_scaling(matrix, [1, 4], repeats=1)
    repeated = noise_scale.batch_scaling(matrix, [1, 4], repeats=8)
    assert single[0]["standard_error"] == pytest.approx(np.std(np.diag(matrix), ddof=1) / 2)
    assert repeated[0]["product"] == single[0]["product"]
    assert repeated[0]["standard_error"] is None
    assert repeated[0]["independent_batches_per_partition"] == 4
    assert single[1]["standard_error"] is None
    with pytest.raises(ValueError, match="positive integer"):
        noise_scale.batch_scaling(matrix, [1], repeats=0)


def test_null_diagnostic_explicitly_disclaims_confidence_coverage():
    matrix = noise_scale.gram(synthetic(40, 8, 0.1, 1.0))
    assert noise_scale.identifiability(matrix)["confidence_calibrated"] is False
