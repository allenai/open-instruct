"""Tests for open_instruct.parsing."""

import dataclasses
import enum
import os
import tempfile
import unittest
from typing import Literal

from parameterized import parameterized

from open_instruct import parsing


@dataclasses.dataclass
class _SimpleConfig:
    foo: int = 1
    bar: str = "hi"
    flag: bool = False
    nums: list[int] = dataclasses.field(default_factory=list)
    maybe: int | None = None


class _Mode(enum.StrEnum):
    a = "a"
    b = "b"


@dataclasses.dataclass
class _RichConfig:
    mode: _Mode = _Mode.a
    kl: Literal[0, 1, 2, 3] = 2


@dataclasses.dataclass
class _GroupB:
    other: int = 7


class ParseSingleDataclassTest(unittest.TestCase):
    def test_returns_instance_not_tuple_for_single_dataclass(self) -> None:
        result = parsing.parse(_SimpleConfig, args=["--foo", "5"])
        self.assertIsInstance(result, _SimpleConfig)
        self.assertEqual(result.foo, 5)
        self.assertEqual(result.bar, "hi")

    def test_uses_defaults_when_no_args(self) -> None:
        result = parsing.parse(_SimpleConfig, args=[])
        self.assertEqual(result.foo, 1)
        self.assertEqual(result.flag, False)

    @parameterized.expand([("snake", "--foo"), ("kebab", "--foo")])
    def test_snake_and_kebab_names_both_accepted(self, _name: str, flag: str) -> None:
        result = parsing.parse(_SimpleConfig, args=[flag, "9"])
        self.assertEqual(result.foo, 9)


class ParseMultipleDataclassesTest(unittest.TestCase):
    def test_returns_tuple_in_input_order(self) -> None:
        a, b = parsing.parse(_SimpleConfig, _GroupB, args=["--foo", "3", "--other", "11"])
        self.assertIsInstance(a, _SimpleConfig)
        self.assertIsInstance(b, _GroupB)
        self.assertEqual(a.foo, 3)
        self.assertEqual(b.other, 11)

    def test_duplicate_field_names_across_dataclasses_raise(self) -> None:
        @dataclasses.dataclass
        class _DupA:
            x: int = 1

        @dataclasses.dataclass
        class _DupB:
            x: int = 2

        with self.assertRaises(ValueError):
            parsing.parse(_DupA, _DupB, args=[])


class RichTypeSupportTest(unittest.TestCase):
    def test_literal_and_enum_round_trip(self) -> None:
        result = parsing.parse(_RichConfig, args=["--mode", "b", "--kl", "3"])
        self.assertEqual(result.mode, _Mode.b)
        self.assertEqual(result.kl, 3)

    def test_literal_rejects_invalid_value(self) -> None:
        with self.assertRaises(SystemExit):
            parsing.parse(_RichConfig, args=["--kl", "99"])


class YamlTest(unittest.TestCase):
    def _write_yaml(self, body: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(body)
            path = f.name
        self.addCleanup(os.unlink, path)
        return path

    def test_yaml_loads_values(self) -> None:
        path = self._write_yaml("foo: 7\nbar: hello\nflag: true\nnums: [1, 2, 3]\n")
        result = parsing.parse(_SimpleConfig, args=[path])
        self.assertEqual(result.foo, 7)
        self.assertEqual(result.bar, "hello")
        self.assertTrue(result.flag)
        self.assertEqual(result.nums, [1, 2, 3])

    def test_cli_overrides_yaml(self) -> None:
        path = self._write_yaml("foo: 7\nbar: hello\n")
        result = parsing.parse(_SimpleConfig, args=[path, "--foo", "99"])
        self.assertEqual(result.foo, 99)
        self.assertEqual(result.bar, "hello")

    def test_unknown_yaml_key_raises(self) -> None:
        path = self._write_yaml("nonexistent_field: 1\n")
        with self.assertRaises(ValueError):
            parsing.parse(_SimpleConfig, args=[path])

    def test_unknown_yaml_key_ignored_when_allow_extra_keys(self) -> None:
        path = self._write_yaml("nonexistent_field: 1\nfoo: 4\n")
        result = parsing.parse(_SimpleConfig, args=[path], allow_extra_keys=True)
        self.assertEqual(result.foo, 4)


class DefaultsOverrideTest(unittest.TestCase):
    def test_defaults_override_field(self) -> None:
        result = parsing.parse(_SimpleConfig, args=[], defaults={"foo": 42})
        self.assertEqual(result.foo, 42)

    def test_defaults_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            parsing.parse(_SimpleConfig, args=[], defaults={"nope": 1})

    def test_cli_still_overrides_defaults(self) -> None:
        result = parsing.parse(_SimpleConfig, args=["--foo", "9"], defaults={"foo": 42})
        self.assertEqual(result.foo, 9)


if __name__ == "__main__":
    unittest.main()
