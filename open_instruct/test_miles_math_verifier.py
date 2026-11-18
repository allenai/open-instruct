"""General math grading regressions, including compact saved-rollout examples."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest import mock

import pytest
from math_verify import errors, grader, parser
from scripts.miles import prepare_eopd_math_prompts, prepare_qwen35_math_prompts

from open_instruct import ground_truth_utils
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.datasets import run_data, source_preparation
from open_instruct.miles.rewards import math_verifier, rewards


@pytest.fixture(autouse=True)
def clear_reference_probe_cache():
    math_verifier._requires_legacy_verifier.cache_clear()
    math_verifier._false_equation.cache_clear()
    yield
    math_verifier._requires_legacy_verifier.cache_clear()
    math_verifier._false_equation.cache_clear()


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        ("f(1)=1", r"\boxed{f(1)=0}", 0),
        ("f(1)=1", r"\boxed{3+4=12}", 0),
        ("2+2=4", r"\boxed{2+2=5}", 0),
        ("360 / 72 = 5", r"\boxed{1+2=8}", 0),
        ("P(0) = 7", r"\boxed{P(0)=8}", 0),
        ("i=0", r"\boxed{1=0}", 0),
        ("I = 0", r"\boxed{1=0}", 0),
        ("i=0", r"\boxed{j=0}", 0),
        ("f(1)=1", r"\boxed{f(1)=1}", 1),
        ("f(1)=1", r"\boxed{f(1)=2-1}", 1),
        ("2+2=4", r"\boxed{2+2=4}", 1),
        ("360 / 72 = 5", r"\boxed{360 / 72 = 5}", 1),
        ("360 / 72 = 5", r"\boxed{360/72=5}", 1),
        (r"\gcd(m,n) = 1", r"\boxed{\gcd(m, n) = 1}", 1),
        (r"10\times6-(9-3)\times2=48", r"\boxed{10 \times 6 - (9 - 3) \times 2 = 48}", 1),
        (r"10\times6-(9-3)\times2=48", r"\boxed{10 + 6 \times (9 - 3) + 2 = 48}", 1),
        ("x+2=4", r"\boxed{x=2}", 1),
        ("x+2=4", r"\boxed{x=3}", 0),
        (r"\emptyset", r"\boxed{\emptyset}", 1),
        (r"\emptyset", r"\boxed{\{1\}}", 0),
        (r"\{1\}", r"\boxed{\emptyset}", 0),
    ],
)
def test_false_equation_probe_guards_symbolic_matches(label, prediction, expected):
    assert math_verifier.MathVerifier()([], prediction, label).score == expected


@pytest.mark.parametrize(
    "label,prediction,expected",
    [("360 / 72 = 5", r"\boxed{5}", 1), (r"10\times6-(9-3)\times2=48", r"\boxed{48=48}", 0)],
)
@pytest.mark.xfail(strict=True, reason="Numeric equation references need task-aware value/construction grading")
def test_numeric_equation_reference_limitations(label, prediction, expected):
    # The first is a value question; the second asks for an arithmetic construction.
    # Equation-side equivalence alone cannot enforce these different task contracts.
    assert math_verifier.MathVerifier()([], prediction, label).score == expected


@pytest.mark.parametrize("reference,expected,probe_calls", [("f(1)=1", True, 1), ("0.5", False, 0), ("8", False, 0)])
def test_reference_probe_is_reused_for_different_predictions(reference, expected, probe_calls):
    with mock.patch.object(math_verifier.math_verify, "verify", wraps=math_verifier.math_verify.verify) as verify:
        assert math_verifier._requires_legacy_verifier(reference) is expected
        assert math_verifier._requires_legacy_verifier(reference) is expected
        assert verify.call_count == probe_calls
    assert math_verifier._requires_legacy_verifier.cache_info().maxsize == 4096


@pytest.mark.parametrize("error", [errors.TimeoutException, ValueError])
def test_failed_reference_probe_is_not_cached(error):
    with mock.patch.object(math_verifier.math_verify, "verify", side_effect=error), pytest.raises(error):
        math_verifier._requires_legacy_verifier("f(1)=1")
    assert math_verifier._requires_legacy_verifier.cache_info().currsize == 0
    assert math_verifier._requires_legacy_verifier("f(1)=1") is True


@pytest.mark.parametrize("error", [errors.TimeoutException, ValueError])
@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        ("f(1)=1", r"\boxed{f(1)=2-1}", 1),
        ("f(1)=1", r"\boxed{f(1)=0}", 0),
        ("i=0", r"\boxed{0}", 1),
        ("i=0", r"\boxed{1=0}", 0),
        (r"y=\frac{x^2}{2}+C\sqrt{x}", r"\boxed{y=\frac{1}{2}x^2+C\sqrt{x}}", 1),
        (r"Ae^{100kt}=\frac{N}{100-N}", r"\boxed{\frac{N}{100-N}=Ae^{100kt}}", 1),
    ],
)
def test_probe_failure_preserves_safe_comparisons(error, label, prediction, expected):
    with mock.patch.object(math_verifier, "_requires_legacy_verifier", side_effect=error):
        result = math_verifier.MathVerifier()([], prediction, label)
    assert result.score == expected
    assert result.diagnostics["symbolic_timeouts"] == int(error is errors.TimeoutException)


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        ("i=0", r"\boxed{0}", 1),
        ("i=4", r"\boxed{4}", 1),
        ("i=4", r"\boxed{5}", 0),
        (r"B=\begin{pmatrix}2c&c\\c&2c\end{pmatrix}", r"\boxed{\begin{pmatrix}2c&c\\c&2c\end{pmatrix}}", 1),
        (r"\gcd(m,n)=1", r"\boxed{1=1}", 0),
    ],
)
def test_guard_preserves_assignment_values_without_comparing_parsed_equations(label, prediction, expected):
    assert math_verifier.MathVerifier()([], prediction, label).score == expected


@pytest.mark.parametrize("prediction", ["f(1)=1", "$f(1)=1$", "We need to calculate f(1)=1 and then", r"\boxed{}"])
def test_legacy_fallback_preserves_answer_anchor_requirement(prediction):
    with mock.patch.object(math_verifier, "_legacy_compare") as legacy:
        assert math_verifier.MathVerifier()([], prediction, "f(1)=1").score == 0
    legacy.assert_not_called()


def test_legacy_fallback_is_limited_to_flagged_references():
    with mock.patch.object(math_verifier, "_legacy_compare") as legacy:
        assert math_verifier.MathVerifier()([], r"\boxed{3,-2}", "-2,3").score == 1
        assert math_verifier.MathVerifier()([], r"\boxed{-2}", "-2,3").score == 0
    legacy.assert_not_called()


def test_legacy_fallback_preserves_literal_match_and_timeout_diagnostics():
    with (
        mock.patch.object(
            math_verifier.math_utils,
            "is_equiv",
            side_effect=lambda *args, diagnostics: diagnostics.update(symbolic_timeouts=2) or False,
        ),
        mock.patch.object(math_verifier.math_utils, "hendrycks_is_equiv", return_value=False),
    ):
        result = math_verifier.MathVerifier()([], r"\boxed{f(1)=1}", r"\(f(1)=1\)")
    assert result.score == 1
    assert result.diagnostics == {"symbolic_timeouts": 2}


def test_false_equation_guard_runs_inside_isolated_worker():
    async def run():
        verifier = math_verifier.MathVerifier()
        return await asyncio.gather(
            verifier.async_call([], r"\boxed{f(1)=0}", "f(1)=1"), verifier.async_call([], r"\boxed{f(1)=1}", "f(1)=1")
        )

    results = asyncio.run(run())
    assert [result.score for result in results] == [0, 1]


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        ("4", r"\boxed{5}", 0),
        ("0.5", r"\boxed{\frac{1}{2}}", 1),
        ("4", "Answer: 4", 1),
        ("4", "Final Answer: The final answer is 4. I hope it is correct.", 1),
        ("4", "$4$", 0),
        ("4", "4", 0),
        ("4", "", 0),
        ("", r"\boxed{}", 0),
        ("4", "No answer.", 0),
        ("-2, 3", r"\boxed{-2}", 0),
        ("-2, 3", r"\boxed{-2, 4}", 0),
        ("-2, 3", r"\boxed{3, -2}", 1),
        ("1,000", r"\boxed{1000}", 1),
        ("(1,2,3)", r"\boxed{(3,2,1)}", 0),
        (r"\{1,2,3\}", r"\boxed{\{3,2,1\}}", 1),
        (r"x \in (0,1)", r"\boxed{(0,1)}", 1),
        # C039/81481: a wrapped tuple reference previously rejected a correct answer.
        (r"\((4,2,5)\)", r"\boxed{(4,2,5)}", 1),
        # C040/2600: old comparison kept only the first root, accepting omissions.
        (r"\(-\pi,-\frac{\pi}{6},0,\frac{\pi}{6},\pi\)", r"\boxed{-\pi,0,\pi}", 0),
        (
            r"\(-\pi,-\frac{\pi}{6},0,\frac{\pi}{6},\pi\)",
            r"\boxed{-\pi}, \boxed{-\frac{\pi}{6}}, \boxed{0}, \boxed{\frac{\pi}{6}}, \boxed{\pi}",
            1,
        ),
    ],
)
def test_math_verifier(label, prediction, expected):
    result = math_verifier.MathVerifier()([], prediction, label)
    assert result.score == expected
    assert result.diagnostics == {"symbolic_timeouts": 0}


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        ("52", "We need to compute 52 and then", 0),
        ("1", r"We need to evaluate $P(\bar{X} \leq 1)$ and then", 0),
        ("52", r"We have 52 remaining. \boxed{}", 0),
        (r"\mathbb{Z}", r"A group is $\mathbb{Z}$. Final answer: \boxed{\mathbb{Z} \times \mathbb{Z}}", 0),
        ("52", "We have 52 remaining. " * 4000, 0),
        ("24", "**Answer:** **24**", 1),
        ("24", "The **answer** is **24**.", 1),
        ("24", r"**Answer:** $24$", 1),
        ("24", r"**\boxed{24}**", 1),
        ("52", "The answer is 52", 1),
        ("8", "Answer: 2**3", 1),
        ("23", "Answer: 2**3", 0),
        ("8", "**Answer:** **2**3**", 1),
        ("23", "**Answer:** **2**3**", 0),
        ("12", "Answer: 2**3 + 2**2", 1),
        # This setting requires an answer anchor, not a complete-response detector.
        ("52", "The answer is 52, and then", 1),
    ],
)
def test_prediction_requires_box_or_answer_anchor(label, prediction, expected):
    assert math_verifier.MathVerifier()([], prediction, label).score == expected


@pytest.mark.parametrize(
    "_name,prediction,label",
    [
        # Equivalent answers that are not copies of the reference.
        ("evaluated_vs_inline_paren", r"\boxed{16}", r"\(2^4\)"),
        ("evaluated_vs_display_bracket", r"\boxed{16}", r"\[2^4\]"),
        ("evaluated_vs_dollar", r"\boxed{16}", r"$2^4$"),
        ("evaluated_vs_double_dollar", r"\boxed{16}", r"$$2^4$$"),
        ("evaluated_vs_plain", r"\boxed{16}", "2^4"),
        ("same_expression_unwrapped", r"\boxed{2^4}", r"\(2^4\)"),
        ("dfrac_vs_wrapped_frac", r"\boxed{\dfrac{\pi}{2}}", r"\(\frac{\pi}{2}\)"),
        ("padded_reference", r"\boxed{x^4}", r"\( x^4 \)"),
        ("wrapped_answer_vs_plain", r"\boxed{\(16\)}", "16"),
        ("wrapped_answer_vs_wrapped", r"\boxed{\(16\)}", r"\(2^4\)"),
        ("minerva_answer_vs_wrapped", "Final Answer: The final answer is 16. I hope it is correct.", r"\(2^4\)"),
        # Literal copies remain accepted alongside equivalent expressions.
        ("verbatim_copy", r"\boxed{\(2^4\)}", r"\(2^4\)"),
    ],
)
def test_equivalent_answers_match(_name, prediction, label):
    assert math_verifier.MathVerifier()([], prediction, label).score == 1


@pytest.mark.parametrize(
    "_name,prediction,label",
    [
        ("wrong_value", r"\boxed{15}", r"\(2^4\)"),
        ("wrong_expression", r"\boxed{\frac{\pi}{3}}", r"\(\frac{\pi}{2}\)"),
        ("wrong_wrapped_answer", r"\boxed{\(15\)}", "16"),
        ("empty_prediction", "", r"\(2^4\)"),
        ("empty_box", r"\boxed{}", r"\(2^4\)"),
        # Malformed references and incomplete answers must not gain credit.
        ("wrong_vs_unbalanced_reference", r"\boxed{15}", r"\(2^4"),
        ("empty_wrapped_reference", r"\boxed{16}", r"\(\)"),
        ("one_of_two_spans", r"\boxed{a}", r"\(a\) and \(b\)"),
        ("wrong_vs_escaped_dollar_amount", r"\boxed{6}", r"\$5\$"),
    ],
)
def test_non_equivalent_or_malformed_inputs_do_not_match(_name, prediction, label):
    assert math_verifier.MathVerifier()([], prediction, label).score == 0


@pytest.mark.parametrize(
    "_name,text,expected",
    [
        ("inline_paren", r"\(2^4\)", "2^4"),
        ("inline_paren_padded", r"\( x^4 \)", "x^4"),
        ("display_bracket", r"\[ 98 \]", "98"),
        ("display_bracket_multiline", "\\[\n172890\n\\]", "172890"),
        ("dollar", r"$\pi$", r"\pi"),
        ("double_dollar", r"$$\frac{1}{2}$$", r"\frac{1}{2}"),
        ("surrounding_whitespace", "  \\(\\frac{\\pi}{2}\\) ", r"\frac{\pi}{2}"),
        ("only_one_pair_removed", r"\(\(2^4\)\)", r"\(\(2^4\)\)"),
    ],
)
def test_removes_one_pair_around_whole_answer(_name, text, expected):
    assert math_verifier.strip_outer_math_delimiters(text) == expected


@pytest.mark.parametrize(
    "_name,text",
    [
        ("plain", "2^4"),
        ("two_spans", r"\(a\) and \(b\)"),
        ("two_dollar_spans", r"$a$, $b$"),
        ("prose_with_span", r"The set \( X \) exists."),
        ("unbalanced_open", r"\(2^4"),
        ("unbalanced_close", r"2^4\)"),
        ("mismatched_pair", r"\(2^4\]"),
        ("empty_pair", r"\(\)"),
        ("whitespace_pair", r"$ $"),
        ("escaped_dollar_amount", r"\$5"),
        ("escaped_closing_dollar", r"$5\$"),
        ("boxed_is_not_a_delimiter", r"\boxed{16}"),
        ("left_right_parentheses", r"\left(\frac{1}{2}\right)"),
    ],
)
def test_leaves_other_text_unchanged(_name, text):
    assert math_verifier.strip_outer_math_delimiters(text) == text


@pytest.mark.parametrize("value", [16, None])
def test_delimiter_helper_preserves_non_string_input(value):
    assert math_verifier.strip_outer_math_delimiters(value) == value


@pytest.mark.parametrize(
    "label,prediction,expected", [("1,000, 2,000", r"\boxed{1000, 2000}", 1), ("1,000, 2,000", r"\boxed{0, 1, 2}", 0)]
)
@pytest.mark.xfail(strict=True, reason="Known Math-Verify 0.9.0 limitations; review on dependency upgrades")
def test_upstream_limitations(label, prediction, expected):
    assert math_verifier.MathVerifier()([], prediction, label).score == expected


def test_parse_timeout_is_counted_and_resets():
    verifier = math_verifier.MathVerifier()
    with mock.patch.object(math_verifier.math_verify, "parse", side_effect=errors.TimeoutException):
        result = verifier([], r"\boxed{4}", "4")
    assert result.score == 0
    assert result.diagnostics == {"symbolic_timeouts": 1}
    assert verifier([], r"\boxed{4}", "4").diagnostics == {"symbolic_timeouts": 0}


def test_each_timed_out_candidate_pair_is_counted():
    with (
        mock.patch.object(math_verifier, "_requires_legacy_verifier", return_value=False),
        mock.patch.object(math_verifier.math_verify, "verify", side_effect=errors.TimeoutException),
    ):
        result = math_verifier.MathVerifier()([], r"\boxed{4}", "4")
    assert result.score == 0
    # Each side has a symbolic and a literal candidate.
    assert result.diagnostics == {"symbolic_timeouts": 4}


def test_real_comparison_alarm_is_recorded(monkeypatch, caplog):
    monkeypatch.setattr(math_verifier, "_TIMEOUT_SECONDS", 1)
    with mock.patch.object(grader, "sympy_expr_eq", side_effect=lambda *a: time.sleep(5)):
        result = math_verifier.MathVerifier()([], r"\boxed{4}", "4")
    # Math-Verify's own literal candidate still agrees after the symbolic timeout.
    assert result.score == 1
    assert result.diagnostics == {"symbolic_timeouts": 1}
    assert "Math-Verify parsing or comparison exceeded 1 seconds" in caplog.text


def test_real_parse_alarm_is_recorded(monkeypatch):
    monkeypatch.setattr(math_verifier, "_TIMEOUT_SECONDS", 1)
    with mock.patch.object(parser, "extract_target_from_pred", side_effect=lambda *a, **kw: time.sleep(5)):
        result = math_verifier.MathVerifier()([], r"\boxed{4}", "4")
    assert result.score == 0
    assert result.diagnostics == {"symbolic_timeouts": 1}


def test_async_math_uses_process_isolation_and_reaps_workers():
    async def run():
        verifier = math_verifier.MathVerifier()
        return await asyncio.gather(
            verifier.async_call([], r"\boxed{-2}", "-2, 3"), verifier.async_call([], r"\boxed{3, -2}", "-2, 3")
        )

    children_before = set(rewards._CHILDREN)
    results = asyncio.run(run())
    assert [result.score for result in results] == [0, 1]
    assert all(isinstance(result, ground_truth_utils.VerificationResult) for result in results)
    assert all(result.diagnostics["status"] == "ok" for result in results)
    assert children_before == rewards._CHILDREN


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param("open_instruct.ground_truth_utils.MathVerifier", id="existing-prepared-registry"),
        pytest.param(run_data.FACTORIES["math"], id="builtin-math-preparation"),
        pytest.param(source_preparation.FACTORIES["math"], id="offline-source-preparation"),
        pytest.param(prepare_eopd_math_prompts.VERIFIERS["math"]["factory"], id="eopd-preparation"),
        pytest.param(prepare_qwen35_math_prompts.VERIFIERS["math"]["factory"], id="qwen35-preparation"),
    ],
)
def test_miles_math_routes_use_math_verify_through_worker(tmp_path, factory):
    spec = {"factory": factory, "config": {}}
    registry = tmp_path / "verifiers.json"
    registry.write_text(json.dumps({"math": spec}))
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(registry)))
    samples = [
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response=response,
            prompt="Find both roots",
            metadata={"verifiers": [{"name": "math", "target": "-2, 3"}]},
        )
        for response in (r"\boxed{-2}", r"\boxed{3, -2}")
    ]
    assert asyncio.run(rewards.registered_reward(args, samples)) == [0, 1]
    assert json.loads(registry.read_text()) == {"math": spec}
    # Offline/direct construction and child worker construction share the mapping.
    assert isinstance(rewards._resolve_verifier(spec), math_verifier.MathVerifier)
    assert spec["factory"] == factory


def test_legacy_math_and_explicit_strict_contract_are_unchanged():
    legacy = ground_truth_utils.MathVerifier()
    assert legacy([], r"\boxed{-2}", "-2, 3").score == 1
    assert legacy([], r"\boxed{3, -2}", "-2, 3").score == 0
    # Legacy build_all_verifiers enumerates direct subclasses: importing the MILES
    # implementation must not add a second verifier that overwrites its "math" key.
    assert math_verifier.MathVerifier not in ground_truth_utils.VerifierFunction.__subclasses__()
    assert isinstance(
        rewards._resolve_verifier({"factory": "open_instruct.ground_truth_utils.StrictMathVerifier"}),
        ground_truth_utils.StrictMathVerifier,
    )


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        # C039/72954: preserve both separately delimited solution pairs.
        (r"\(x_1=125, y_1=4\) and \(x_2=625, y_2=3\)", r"\boxed{(125,4)} \text{ and } \boxed{(625,3)}", 1),
        (r"\(x_1=125, y_1=4\) and \(x_2=625, y_2=3\)", r"\boxed{(125,4)}", 0),
        # C040/146620: the old startswith/endswith loop corrupted this reference.
        (r"\(x=2\) or \(x=-2\)", r"\boxed{2} \text{ and } \boxed{-2}", 1),
        (r"\(x=2\) or \(x=-2\)", r"\boxed{2}", 0),
        # C039/94889: complete pair of function equations.
        (r"\(f(x)=x\) or \(f(x)=-x\)", r"\boxed{f(x)=x} \text{ and } \boxed{f(x)=-x}", 1),
        # C040/272435: a fragment of the damaged reference previously gave credit.
        (
            r"\(E[W]=-1\) and \(P(W>0)=1-(\frac{36}{37})^{35}\)",
            r"\boxed{E[W]=-37} \text{ and } \boxed{P(W>0)=\frac{36}{37}}",
            0,
        ),
    ],
)
def test_multi_span_reference_replay(label, prediction, expected):
    assert math_verifier.MathVerifier()([], prediction, label).score == expected


_TRIG_REFERENCE = (
    r"\(\sin \beta = \frac{12}{13}\), \(\tan \beta = \frac{12}{5}\), "
    r"\(\cot \beta = \frac{5}{12}\), \(\cos(2\beta) = \frac{-119}{169}\)"
)


def test_multi_span_trig_equations_with_labels_inside_boxes():
    # C040/94572 draw 5: the whole set now matches instead of only its last value.
    answer = (
        r"\boxed{\sin \beta = \dfrac{12}{13}}, \boxed{\tan \beta = \dfrac{12}{5}}, "
        r"\boxed{\cot \beta = \dfrac{5}{12}}, \boxed{\cos(2\beta) = -\dfrac{119}{169}}"
    )
    assert math_verifier.MathVerifier()([], answer, _TRIG_REFERENCE).score == 1


@pytest.mark.xfail(
    strict=True, reason="Math-Verify cannot associate separate boxed values with equation labels outside them"
)
def test_multi_span_trig_equations_with_labels_outside_boxes():
    # C040/94572 draw 4: a known regression from preserving the full reference.
    answer = (
        "\\[\n\\sin \\beta = \\boxed{\\dfrac{12}{13}}\n\\]\n"
        "\\[\n\\tan \\beta = \\boxed{\\dfrac{12}{5}}\n\\]\n"
        "\\[\n\\cot \\beta = \\boxed{\\dfrac{5}{12}}\n\\]\n"
        "\\[\n\\cos(2\\beta) = \\boxed{-\\dfrac{119}{169}}\n\\]"
    )
    assert math_verifier.MathVerifier()([], answer, _TRIG_REFERENCE).score == 1
