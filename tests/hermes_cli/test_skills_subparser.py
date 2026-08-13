"""Test that skills subparser doesn't conflict (regression test for #898)."""

import argparse

from tests.hermes_cli.conftest import isolated_hermes_modules


def test_no_duplicate_skills_subparser():
    """Ensure 'skills' subparser is only registered once to avoid Python 3.11+ crash.

    Python 3.11 changed argparse to raise an exception on duplicate subparser
    names instead of silently overwriting (see CPython #94331).

    This test will fail with:
        argparse.ArgumentError: argument command: conflicting subparser: skills

    if the duplicate 'skills' registration is reintroduced.
    """
    # Force fresh import of the module where parser is constructed.
    # If there are duplicate 'skills' subparsers, this import will raise
    # argparse.ArgumentError at module load time.
    #
    # The eviction is restored on exit: sys.modules is process-global, and
    # leaving hermes_cli.main evicted makes a later patch("hermes_cli.main.x")
    # in another test file patch a second module object while the test under
    # test still calls the original (SCA-4692).
    with isolated_hermes_modules():
        try:
            import hermes_cli.main  # noqa: F401
        except argparse.ArgumentError as e:
            if "conflicting subparser" in str(e):
                raise AssertionError(
                    f"Duplicate subparser detected: {e}. "
                    "See issue #898 for details."
                ) from e
            raise
