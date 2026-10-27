from __future__ import annotations

from typer.testing import CliRunner

from moray_surgeon import __version__
from moray_surgeon.cli import app

runner = CliRunner()


def test_version_command_prints_version() -> None:
    """`version` must print the package version; unknown commands must fail.

    Per R10 each public function needs a normal-path and an exception-path
    test; this single smoke test deliberately covers both (exit code 0 with
    the version string, then a non-zero exit for an unknown subcommand) so
    the CI baseline stays at exactly one test as required by the acceptance
    gate.
    """
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output

    unknown = runner.invoke(app, ["definitely-not-a-command"])
    assert unknown.exit_code != 0
