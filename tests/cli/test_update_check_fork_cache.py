import json
import subprocess
from pathlib import Path


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    repo = tmp_path / "hermes-agent"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "hermes_cli").mkdir()
    (repo / "hermes_cli" / "banner.py").write_text("# placeholder\n")
    (repo / "README.md").write_text("test\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "fetch", "origin", "--quiet")
    return repo, tmp_path / "home"


def test_update_check_cache_invalidates_when_checkout_git_state_changes(tmp_path, monkeypatch):
    from hermes_cli import banner

    repo, hermes_home = _make_repo(tmp_path)
    hermes_home.mkdir()
    cache_file = hermes_home / ".update_check"
    cache_file.write_text(
        json.dumps(
            {
                "ts": 9_999_999_999,
                "behind": 42,
                "rev": None,
                "ver": banner.VERSION,
                "git": {
                    "head": "stale",
                    "branch": "sync/upstream-rebase",
                    "origin_head": "stale",
                    "origin_url": "https://example.invalid/old.git",
                },
            }
        )
    )

    monkeypatch.setattr(banner, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(banner, "__file__", str(repo / "hermes_cli" / "banner.py"))

    assert banner.check_for_updates() == 0
    refreshed = json.loads(cache_file.read_text())
    assert refreshed["behind"] == 0
    assert refreshed["git"]["branch"] == "main"
    assert refreshed["git"]["head"] == _git(repo, "rev-parse", "HEAD")
