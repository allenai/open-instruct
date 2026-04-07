"""Tests for example verifiers (Manufactoria, Ballsim) under `examples/`."""

import unittest
from unittest.mock import Mock, patch

from examples.ballsim.verifier import BallsimVerifier, BallsimVerifierConfig
from examples.manufactoria.verifier import ManufactoriaVerifier, ManufactoriaVerifierConfig

from open_instruct.ground_truth_registry import _decorated_verifier_classes, get_registered_verifier_config_classes
from open_instruct.ground_truth_utils import import_extra_verifier_modules, parse_extra_verifier_cli_args


class TestBallsimVerifier(unittest.IsolatedAsyncioTestCase):
    async def test_pass_rate_scoring(self):
        verifier = BallsimVerifier(
            BallsimVerifierConfig(
                ballsim_api_url="http://localhost:2345/test_program",
                ballsim_max_execution_time=1.0,
                ballsim_scoring_mode="pass_rate",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"results": [1, 0, 1], "runtimes": [0.1, 0.2, 0.1]}
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(BallsimVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call([], "```python\npass\n```", ["assert True"], None)

        self.assertAlmostEqual(result.score, 2 / 3)

    async def test_all_pass_scoring(self):
        verifier = BallsimVerifier(
            BallsimVerifierConfig(
                ballsim_api_url="http://localhost:2345/test_program",
                ballsim_max_execution_time=1.0,
                ballsim_scoring_mode="all_pass",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"results": [1, 0, 1], "runtimes": [0.1, 0.2, 0.1]}
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(BallsimVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call([], "```python\npass\n```", ["assert True"], None)

        self.assertEqual(result.score, 0.0)


class TestManufactoriaVerifier(unittest.IsolatedAsyncioTestCase):
    async def test_pass_rate_scoring(self):
        verifier = ManufactoriaVerifier(
            ManufactoriaVerifierConfig(
                manufactoria_api_url="http://localhost:1235/test_solution",
                manufactoria_max_execution_time=1.0,
                manufactoria_scoring_mode="pass_rate",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "valid": True,
            "all_passed": False,
            "results": [{"passed": True}, {"passed": False}, {"passed": True}],
        }
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(ManufactoriaVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call(
                [], "```manufactoria\nSTART start:\n    NEXT end\nEND end\n```", [{}], None
            )

        self.assertAlmostEqual(result.score, 2 / 3)

    async def test_all_pass_scoring(self):
        verifier = ManufactoriaVerifier(
            ManufactoriaVerifierConfig(
                manufactoria_api_url="http://localhost:1235/test_solution",
                manufactoria_max_execution_time=1.0,
                manufactoria_scoring_mode="all_pass",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "valid": True,
            "all_passed": False,
            "results": [{"passed": True}, {"passed": False}, {"passed": True}],
        }
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(ManufactoriaVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call(
                [], "```manufactoria\nSTART start:\n    NEXT end\nEND end\n```", [{}], None
            )

        self.assertEqual(result.score, 0.0)


class TestParseExtraVerifierCliArgs(unittest.TestCase):
    def test_parses_flags_and_coerces_float(self):
        ns = parse_extra_verifier_cli_args(
            ["--manufactoria_scoring_mode", "pass_rate", "--manufactoria_max_execution_time", "2.5"]
        )
        self.assertEqual(ns.manufactoria_scoring_mode, "pass_rate")
        self.assertEqual(ns.manufactoria_max_execution_time, 2.5)

    def test_equals_form(self):
        ns = parse_extra_verifier_cli_args(["--manufactoria_api_url=http://localhost:1235/test_solution"])
        self.assertEqual(ns.manufactoria_api_url, "http://localhost:1235/test_solution")


class TestImportExtraVerifierModules(unittest.TestCase):
    def test_import_registers_example_verifiers(self):
        import_extra_verifier_modules(["examples.manufactoria.register", "examples.ballsim.register"])
        m = ManufactoriaVerifier(ManufactoriaVerifierConfig())
        b = BallsimVerifier(BallsimVerifierConfig())
        self.assertEqual(m.name, "manufactoria")
        self.assertEqual(b.name, "ballsim")


class TestRegisterVerifier(unittest.TestCase):
    def test_example_verifiers_opt_in_via_decorator(self):
        names = {c.__name__ for c in _decorated_verifier_classes}
        self.assertIn("ManufactoriaVerifier", names)
        self.assertIn("BallsimVerifier", names)


class TestRegisterVerifierConfig(unittest.TestCase):
    def test_verifier_configs_registered_for_streaming_merge(self):
        cfg_names = {c.__name__ for c in get_registered_verifier_config_classes()}
        self.assertIn("ManufactoriaVerifierConfig", cfg_names)
        self.assertIn("BallsimVerifierConfig", cfg_names)
