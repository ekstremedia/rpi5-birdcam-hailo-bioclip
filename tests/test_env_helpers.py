"""Tests for _load_env, _env_int, _env_float helpers."""

import os

from bird_monitor import _load_env, _env_int, _env_float


class TestLoadEnv:
    def test_sets_variables(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_VAR=hello\n")
        # Clear first to ensure setdefault works
        os.environ.pop("MY_TEST_VAR", None)
        _load_env(env_file)
        assert os.environ["MY_TEST_VAR"] == "hello"
        os.environ.pop("MY_TEST_VAR", None)

    def test_skips_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# a comment\nMY_TEST_VAR2=world\n")
        os.environ.pop("MY_TEST_VAR2", None)
        _load_env(env_file)
        assert os.environ["MY_TEST_VAR2"] == "world"
        os.environ.pop("MY_TEST_VAR2", None)

    def test_skips_blank_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("\n\nMY_TEST_VAR3=yes\n\n")
        os.environ.pop("MY_TEST_VAR3", None)
        _load_env(env_file)
        assert os.environ["MY_TEST_VAR3"] == "yes"
        os.environ.pop("MY_TEST_VAR3", None)

    def test_does_not_override_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_VAR4=new\n")
        os.environ["MY_TEST_VAR4"] = "existing"
        _load_env(env_file)
        assert os.environ["MY_TEST_VAR4"] == "existing"
        os.environ.pop("MY_TEST_VAR4", None)

    def test_missing_file_ok(self, tmp_path):
        # Should not raise
        _load_env(tmp_path / "nonexistent.env")

    def test_value_with_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_URL=http://host:5555/path?a=b\n")
        os.environ.pop("MY_TEST_URL", None)
        _load_env(env_file)
        assert os.environ["MY_TEST_URL"] == "http://host:5555/path?a=b"
        os.environ.pop("MY_TEST_URL", None)


class TestEnvInt:
    def test_returns_env_value(self):
        os.environ["MY_INT_TEST"] = "42"
        assert _env_int("MY_INT_TEST", 10) == 42
        os.environ.pop("MY_INT_TEST", None)

    def test_returns_default(self):
        os.environ.pop("MY_INT_MISSING", None)
        assert _env_int("MY_INT_MISSING", 99) == 99


class TestEnvFloat:
    def test_returns_env_value(self):
        os.environ["MY_FLOAT_TEST"] = "0.75"
        assert _env_float("MY_FLOAT_TEST", 0.5) == 0.75
        os.environ.pop("MY_FLOAT_TEST", None)

    def test_returns_default(self):
        os.environ.pop("MY_FLOAT_MISSING", None)
        assert _env_float("MY_FLOAT_MISSING", 3.14) == 3.14
