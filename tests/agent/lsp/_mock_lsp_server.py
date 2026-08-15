#!/usr/bin/env python3
"""A minimal in-process LSP server used by tests.

Speaks just enough LSP to drive :class:`agent.lsp.client.LSPClient`
through a full lifecycle: ``initialize``, ``initialized``,
``textDocument/didOpen``, ``textDocument/didChange``, then a
``textDocument/publishDiagnostics`` notification followed by
``shutdown`` + ``exit``.

Behaviour (all behaviours selectable via env var ``MOCK_LSP_SCRIPT``):

- ``"clean"`` — initialize, accept didOpen/didChange, push empty
  diagnostics on every open/change, exit cleanly on shutdown.
- ``"errors"`` — same as ``clean`` but the published diagnostics
  carry one severity-1 entry pointing at line 0:0.
- ``"crash"`` — exit immediately after responding to ``initialize``
  (simulates a crashing server).
- ``"slow"`` — same as ``clean`` but sleeps 1s before responding to
  ``initialize`` (lets us test timeout behaviour).
- ``"stale"`` — pushes one error on ``didOpen``, then goes SILENT on
  ``didChange`` (no push) and rejects the pull endpoint with
  method-not-found.  Models a slow tsserver that hasn't re-checked
  the edited content yet — the ghost-diagnostics scenario.
- ``"slow_push"`` — like ``stale`` on didOpen (one error) but on
  ``didChange`` sleeps ``MOCK_LSP_PUSH_DELAY`` seconds (default 1.0)
  and then pushes EMPTY diagnostics.  Models a server that fixes
  the ghost if you actually wait for it.  Pull endpoint rejects.
- ``"hang_shutdown"`` — serves like ``errors``, but refuses to die:
  never answers ``shutdown``, ignores ``exit``, and ignores SIGTERM.
  Only SIGKILL ends it, so evicting one costs the client's full
  ``shutdown`` request timeout plus ``SHUTDOWN_GRACE``.  Models the
  wedged tsserver that makes an eviction expensive.

The script writes JSON-RPC framed messages to stdout and reads from
stdin.  No third-party dependencies — uses only stdlib so it runs
under whatever Python the test process picks up.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time


def read_message():
    """Read one Content-Length framed JSON-RPC message from stdin."""
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.rstrip(b"\r\n")
        if not line:
            break
        k, _, v = line.decode("ascii").partition(":")
        headers[k.strip().lower()] = v.strip()
    n = int(headers["content-length"])
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))


def write_message(obj):
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _spawn_grandchild():
    """Fork a long-lived worker the way a real language server does.

    ``typescript-language-server`` runs ``tsserver`` as a child, and
    ``pyright-langserver`` forks node workers.  The worker inherits this
    process's group -- we are the group leader, courtesy of
    ``start_new_session=True`` in the client -- and it ignores SIGTERM, so
    it survives anything aimed at this PID alone.  Its PID goes to
    ``MOCK_LSP_CHILD_PIDFILE`` so the test can go looking for the corpse.
    """
    import subprocess

    # The worker publishes its OWN pid, and only after SIG_IGN is armed.
    # Writing it here (right after Popen returns) would race: shutdown could
    # win, the still-default SIGTERM would kill the worker, and the test
    # would pass without ever exercising the SIGKILL escalation it exists
    # to prove.  The rename is atomic, so the reader never sees a partial
    # write and never has to retry a half-written pid.
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,signal,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "pf = os.environ.get('MOCK_LSP_CHILD_PIDFILE')\n"
            "if pf:\n"
            "    tmp = pf + '.tmp'\n"
            "    with open(tmp, 'w', encoding='utf-8') as fh:\n"
            "        fh.write(str(os.getpid()))\n"
            "        fh.flush()\n"
            "        os.fsync(fh.fileno())\n"
            "    os.replace(tmp, pf)\n"
            "while True: time.sleep(0.05)\n",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return worker


def main():
    script = os.environ.get("MOCK_LSP_SCRIPT", "clean")
    if script == "hang_shutdown":
        # Survive the SIGTERM in ``_cleanup_process`` so the eviction has
        # to spend the whole SHUTDOWN_GRACE before SIGKILL lands.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if script == "spawns_child":
        # Hold the handle so the worker is not reaped by GC; it outlives
        # us on purpose.
        _spawn_grandchild()

    while True:
        msg = read_message()
        if msg is None:
            if script == "hang_shutdown":
                # A genuinely wedged server does not tidy itself up when
                # its stdin closes either.  Exiting here would let the
                # eviction finish on the terminate() before SHUTDOWN_GRACE
                # is ever spent, which understates what a hung victim costs.
                while True:
                    time.sleep(1.0)
            return 0

        if "id" in msg and msg.get("method") == "initialize":
            if script == "slow":
                time.sleep(1.0)
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {
                        "capabilities": {
                            "textDocumentSync": 1,  # Full
                            "diagnosticProvider": {"interFileDependencies": False, "workspaceDiagnostics": False},
                        },
                        "serverInfo": {"name": "mock-lsp", "version": "0.1"},
                    },
                }
            )
            if script == "crash":
                return 0
            continue

        if msg.get("method") == "initialized":
            continue

        if msg.get("method") == "workspace/didChangeConfiguration":
            continue

        if msg.get("method") == "workspace/didChangeWatchedFiles":
            continue

        if msg.get("method") in {"textDocument/didOpen", "textDocument/didChange"}:
            params = msg.get("params") or {}
            td = params.get("textDocument") or {}
            uri = td.get("uri", "")
            version = td.get("version", 0)
            is_change = msg.get("method") == "textDocument/didChange"
            error_diag = [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 5},
                    },
                    "severity": 1,
                    "code": "MOCK001",
                    "source": "mock-lsp",
                    "message": "synthetic error from mock-lsp",
                }
            ]
            if script == "stale":
                # Ghost scenario: publish an error for the ORIGINAL
                # content, then never publish again after edits.
                if not is_change:
                    write_message(
                        {
                            "jsonrpc": "2.0",
                            "method": "textDocument/publishDiagnostics",
                            "params": {"uri": uri, "version": version, "diagnostics": error_diag},
                        }
                    )
                continue
            if script == "slow_push":
                diagnostics = error_diag
                if is_change:
                    time.sleep(float(os.environ.get("MOCK_LSP_PUSH_DELAY", "1.0")))
                    diagnostics = []
                write_message(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {"uri": uri, "version": version, "diagnostics": diagnostics},
                    }
                )
                continue
            diagnostics = []
            if script in {"errors", "hang_shutdown"}:
                diagnostics = error_diag
            write_message(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": uri,
                        "version": version,
                        "diagnostics": diagnostics,
                    },
                }
            )
            continue

        if msg.get("method") == "textDocument/diagnostic":
            if script in {"stale", "slow_push"}:
                # These scripts model push-only servers so the ghost
                # can't be papered over by the pull channel.
                write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "error": {"code": -32601, "message": "method not found"},
                    }
                )
                continue
            # Pull endpoint — return empty.
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {"kind": "full", "items": []},
                }
            )
            continue

        if msg.get("method") == "textDocument/didSave":
            continue

        if msg.get("method") == "shutdown":
            if script == "hang_shutdown":
                # No reply, ever.  The client waits out its request
                # timeout, then escalates to signals we also ignore.
                continue
            write_message({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            continue

        if msg.get("method") == "exit":
            if script == "hang_shutdown":
                continue
            return 0

        # Unknown request: respond with method-not-found.
        if "id" in msg:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "error": {"code": -32601, "message": f"method not found: {msg.get('method')}"},
                }
            )


if __name__ == "__main__":
    sys.exit(main())
