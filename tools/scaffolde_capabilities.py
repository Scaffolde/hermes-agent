"""Scaffolde capability registry loading and safe subprocess execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from hermes_constants import get_hermes_home

REGISTRY_RELATIVE_PATH = Path("scaffolde") / "capabilities.json"
MAX_STDOUT_CHARS = 64_000
MAX_STDERR_CHARS = 16_000
DEFAULT_TIMEOUT_SECONDS = 60
_ALLOWED_PARAM_TYPES = {"string", "integer", "boolean"}
_ALLOWED_RISKS = {"read", "write"}
_SECRET_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PRIVATE[_-]?KEY|CREDENTIAL)", re.I)
_SECRET_VALUE_RE = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-|gh[pousr]_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,}|ya29\.)"
)
_TEMPLATE_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class CapabilityRegistryStatus:
    status: str
    capabilities: Dict[str, dict]
    errors: list[dict]
    path: str

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "capabilities": list(self.capabilities.values()),
            "errors": self.errors,
            "path": self.path,
        }


def _err(code: str, message: str, capability_id: str | None = None) -> dict:
    data = {"code": code, "message": message}
    if capability_id:
        data["capability_id"] = capability_id
    return data


def _expand_home(value: Any, hermes_home: Path) -> Any:
    if isinstance(value, str):
        return value.replace("${HERMES_HOME}", str(hermes_home))
    return value


def _inside(base: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _validate_string(value: Any, field: str, errors: list[dict], cid: str | None = None, required: bool = True) -> str:
    if not isinstance(value, str) or (required and not value.strip()):
        errors.append(_err("capability_schema", f"{field} must be a non-empty string", cid))
        return ""
    return value


def _contains_secret_value(env_set: Mapping[str, Any]) -> bool:
    for name, value in env_set.items():
        if not isinstance(value, str):
            return True
        # Secret-like variable names are allowed only for non-secret account/path labels;
        # values that look like embedded credentials are rejected fail-closed.
        if _SECRET_NAME_RE.search(str(name)) and _SECRET_VALUE_RE.search(value):
            return True
        if _SECRET_VALUE_RE.search(value):
            return True
        if "literal-token" in value.lower() or "secret" in value.lower():
            return True
    return False


def _load_managed_files(hermes_home: Path) -> tuple[dict[str, str], list[dict]]:
    path = hermes_home / ".scaffolde" / "deployment.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, [_err("missing_deployment_metadata", "Scaffolde deployment metadata is missing")]
    except Exception:
        return {}, [_err("malformed_deployment_metadata", "Scaffolde deployment metadata is unreadable")]
    managed = raw.get("managed_files") if isinstance(raw, dict) else None
    if not isinstance(managed, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in managed.items()):
        return {}, [_err("malformed_deployment_metadata", "managed_files must be a string hash map")]
    return dict(managed), []


def _managed_relative_path(hermes_home: Path, candidate: Path) -> str | None:
    try:
        return candidate.resolve().relative_to(hermes_home.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _managed_file_matches(hermes_home: Path, relative_path: str, managed_files: Mapping[str, str]) -> bool:
    expected = managed_files.get(relative_path)
    if not expected:
        return False
    try:
        actual = hashlib.sha256((hermes_home / relative_path).read_bytes()).hexdigest()
    except OSError:
        return False
    return actual == expected


def _validate_capability(
    raw: Any,
    hermes_home: Path,
    managed_files: Mapping[str, str],
) -> tuple[str | None, dict | None, list[dict]]:
    errors: list[dict] = []
    if not isinstance(raw, dict):
        return None, None, [_err("capability_schema", "capability must be an object")]
    cid = _validate_string(raw.get("id"), "id", errors, None)
    if raw.get("version") != 1:
        errors.append(_err("capability_version", "capability.version must be 1", cid or None))
    for field in ("tool_name", "authority", "kind", "description"):
        _validate_string(raw.get(field), field, errors, cid or None)
    if raw.get("authority") != "scaffolde":
        errors.append(_err("capability_authority", "capability.authority must be scaffolde", cid or None))
    triggers = raw.get("triggers")
    if not isinstance(triggers, list) or not all(isinstance(t, str) and t.strip() for t in triggers):
        errors.append(_err("capability_schema", "triggers must be a list of strings", cid or None))

    entry = raw.get("entrypoint")
    if not isinstance(entry, dict):
        errors.append(_err("capability_schema", "entrypoint must be an object", cid or None))
    else:
        command = _validate_string(entry.get("command"), "entrypoint.command", errors, cid or None)
        if command and command != "bun":
            errors.append(_err("entrypoint_command", "entrypoint.command must be bun for contract v1", cid or None))
        args = entry.get("args")
        cwd_raw = entry.get("cwd")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            errors.append(_err("capability_schema", "entrypoint.args must be a list of strings", cid or None))
            args = []
        elif len(args) != 1 or not args[0].endswith(".ts") or args[0].startswith("-"):
            errors.append(_err("entrypoint_args", "entrypoint.args must contain one TypeScript path", cid or None))
        else:
            script = Path(_expand_home(args[0], hermes_home))
            if not script.is_absolute():
                script = hermes_home / script
            managed_path = _managed_relative_path(hermes_home, script)
            if managed_path is None or not _managed_file_matches(hermes_home, managed_path, managed_files):
                errors.append(_err("unmanaged_entrypoint", "entrypoint must match its Scaffolde-managed hash", cid or None))
        cwd = Path(_expand_home(cwd_raw, hermes_home)) if isinstance(cwd_raw, str) else None
        if cwd is None or not _inside(hermes_home, cwd):
            errors.append(_err("entrypoint_outside_hermes_home", "entrypoint.cwd must stay under HERMES_HOME", cid or None))
        for part in args:
            expanded = str(_expand_home(part, hermes_home))
            looks_like_entrypoint_path = (
                expanded.startswith("/")
                or expanded.startswith(".")
                or "/" in expanded
                or expanded.endswith((".ts", ".js", ".mjs", ".cjs", ".py", ".sh"))
            )
            if looks_like_entrypoint_path:
                candidate = Path(expanded) if expanded.startswith("/") else (hermes_home / expanded)
                if not _inside(hermes_home, candidate):
                    errors.append(_err("entrypoint_outside_hermes_home", "entrypoint script/path args must stay under HERMES_HOME", cid or None))

    env = raw.get("environment", {})
    if not isinstance(env, dict):
        errors.append(_err("capability_schema", "environment must be an object", cid or None))
    else:
        inherit = env.get("inherit", [])
        env_set = env.get("set", {})
        if not isinstance(inherit, list) or not all(isinstance(n, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in inherit):
            errors.append(_err("capability_schema", "environment.inherit must be env var names", cid or None))
        if not isinstance(env_set, dict) or not all(
            isinstance(k, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
            and isinstance(v, str)
            for k, v in env_set.items()
        ):
            errors.append(_err("capability_schema", "environment.set must be string map", cid or None))
        elif {"HOME", "PATH"} & set(env_set):
            errors.append(_err("environment_override", "environment.set may not override HOME or PATH", cid or None))
        elif _contains_secret_value(env_set):
            errors.append(_err("descriptor_secret_value", "descriptor embeds a secret-like value", cid or None))

    operations = raw.get("operations")
    if not isinstance(operations, dict) or not operations:
        errors.append(_err("capability_schema", "operations must be a non-empty object", cid or None))
    else:
        for opname, op in operations.items():
            if not isinstance(opname, str) or not opname:
                errors.append(_err("capability_schema", "operation names must be non-empty strings", cid or None))
                continue
            if not isinstance(op, dict):
                errors.append(_err("capability_schema", f"operation {opname} must be object", cid or None))
                continue
            if op.get("risk") not in _ALLOWED_RISKS:
                errors.append(_err("capability_schema", f"operation {opname} has invalid risk", cid or None))
            argv = op.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
                errors.append(_err("capability_schema", f"operation {opname}.argv must be non-empty string list", cid or None))
                argv = []
            params = op.get("parameters", {})
            if not isinstance(params, dict):
                errors.append(_err("capability_schema", f"operation {opname}.parameters must be object", cid or None))
                continue
            for pname, spec in params.items():
                if not isinstance(pname, str) or not isinstance(spec, dict):
                    errors.append(_err("capability_schema", f"operation {opname} parameter schema invalid", cid or None))
                    continue
                if spec.get("type") not in _ALLOWED_PARAM_TYPES:
                    errors.append(_err("capability_schema", f"parameter {pname} has invalid type", cid or None))
                if "required" not in spec or not isinstance(spec.get("required"), bool):
                    errors.append(_err("capability_schema", f"parameter {pname} required must be boolean", cid or None))
                if spec.get("type") == "integer":
                    for bound in ("minimum", "maximum"):
                        if bound in spec and (not isinstance(spec[bound], int) or isinstance(spec[bound], bool)):
                            errors.append(_err("capability_schema", f"parameter {pname}.{bound} must be integer", cid or None))
                    if (
                        isinstance(spec.get("minimum"), int)
                        and isinstance(spec.get("maximum"), int)
                        and spec["minimum"] > spec["maximum"]
                    ):
                        errors.append(_err("capability_schema", f"parameter {pname} minimum exceeds maximum", cid or None))
                if "default" in spec:
                    default = spec["default"]
                    expected = spec.get("type")
                    default_valid = (
                        (expected == "string" and isinstance(default, str))
                        or (expected == "integer" and isinstance(default, int) and not isinstance(default, bool))
                        or (expected == "boolean" and isinstance(default, bool))
                    )
                    if not default_valid:
                        errors.append(_err("capability_schema", f"parameter {pname}.default has wrong type", cid or None))
            parameter_names = set(params)
            referenced_parameters: set[str] = set()
            for fragment in argv:
                referenced_parameters.update(_TEMPLATE_RE.findall(fragment))
                without_templates = _TEMPLATE_RE.sub("", fragment)
                if "{" in without_templates or "}" in without_templates:
                    errors.append(_err("capability_schema", f"operation {opname}.argv contains malformed templates", cid or None))
            undeclared = referenced_parameters - parameter_names
            unreferenced = parameter_names - referenced_parameters
            if undeclared:
                errors.append(_err("capability_schema", f"operation {opname}.argv references undeclared parameters: {', '.join(sorted(undeclared))}", cid or None))
            if unreferenced:
                errors.append(_err("capability_schema", f"operation {opname} declares unused parameters: {', '.join(sorted(unreferenced))}", cid or None))
    if errors:
        return cid or None, None, errors
    normalized = dict(raw)
    normalized["entrypoint"] = {
        "command": _expand_home(raw["entrypoint"]["command"], hermes_home),
        "args": [_expand_home(a, hermes_home) for a in raw["entrypoint"].get("args", [])],
        "cwd": _expand_home(raw["entrypoint"]["cwd"], hermes_home),
    }
    env = raw.get("environment", {})
    normalized["environment"] = {
        "inherit": list(env.get("inherit", [])),
        "set": {k: _expand_home(v, hermes_home) for k, v in env.get("set", {}).items()},
    }
    return cid, normalized, []


def load_capability_registry() -> CapabilityRegistryStatus:
    hermes_home = get_hermes_home().resolve()
    path = hermes_home / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return CapabilityRegistryStatus("absent", {}, [], str(path))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CapabilityRegistryStatus("malformed", {}, [_err("registry_json", f"invalid JSON: {exc}")], str(path))
    if not isinstance(raw, dict) or raw.get("version") != 1 or raw.get("producer") != "scaffolde" or not isinstance(raw.get("capabilities"), list):
        return CapabilityRegistryStatus("malformed", {}, [_err("registry_schema", "registry must be version 1 producer scaffolde with capabilities list")], str(path))
    capabilities: dict[str, dict] = {}
    managed_files, metadata_errors = _load_managed_files(hermes_home)
    errors: list[dict] = list(metadata_errors)
    if not _managed_file_matches(hermes_home, "scaffolde/capabilities.json", managed_files):
        errors.append(_err("unmanaged_registry", "capability registry does not match its Scaffolde-managed hash"))
    for cap in raw["capabilities"]:
        cid, normalized, cap_errors = _validate_capability(cap, hermes_home, managed_files)
        errors.extend(cap_errors)
        if normalized and cid:
            if cid in capabilities:
                errors.append(_err("duplicate_capability", f"duplicate capability id: {cid}", cid))
            else:
                capabilities[cid] = normalized
    if errors:
        return CapabilityRegistryStatus("degraded", capabilities, errors, str(path))
    return CapabilityRegistryStatus("valid", capabilities, [], str(path))


def registry_has_valid_capabilities() -> bool:
    return load_capability_registry().status == "valid"


def _validate_arguments(op: dict, args: Mapping[str, Any]) -> tuple[dict | None, str | None]:
    if not isinstance(args, Mapping):
        return None, "arguments must be an object"
    specs = op.get("parameters", {}) or {}
    extra = set(args) - set(specs)
    if extra:
        return None, f"unexpected argument(s): {', '.join(sorted(extra))}"
    validated: dict[str, Any] = {}
    for name, spec in specs.items():
        if name not in args:
            if spec.get("required"):
                return None, f"missing required argument: {name}"
            if "default" in spec:
                value = spec["default"]
            else:
                continue
        else:
            value = args[name]
        typ = spec["type"]
        if typ == "string":
            if not isinstance(value, str):
                return None, f"argument {name} must be string"
        elif typ == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return None, f"argument {name} must be integer"
            if "minimum" in spec and value < spec["minimum"]:
                return None, f"argument {name} below minimum"
            if "maximum" in spec and value > spec["maximum"]:
                return None, f"argument {name} above maximum"
        elif typ == "boolean":
            if not isinstance(value, bool):
                return None, f"argument {name} must be boolean"
        validated[name] = value
    return validated, None


def _render_template(template: str, values: Mapping[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"unknown template parameter {name}")
        return str(values[name])
    return _TEMPLATE_RE.sub(repl, template)


def _build_env(desc: dict) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in desc.get("environment", {}).get("inherit", []):
        if name in os.environ:
            env[name] = os.environ[name]
    for name, value in desc.get("environment", {}).get("set", {}).items():
        env[name] = str(value)
    return env


def _redact(text: str) -> str:
    text = _SECRET_VALUE_RE.sub("[REDACTED]", text or "")
    text = re.sub(r"(?i)(TOKEN|SECRET|PASSWORD|PRIVATE[_-]?KEY|CREDENTIAL)(=|:)[^\s]+", r"\1\2[REDACTED]", text)
    return text


def _approval_details(
    capability_id: str,
    operation: str,
    values: Mapping[str, Any],
    hermes_home: Path,
) -> tuple[str, str, Path | None]:
    canonical = json.dumps(dict(values), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    preview: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, str) and len(value) > 160:
            preview[name] = f"{value[:160]}… ({len(value)} chars)"
        else:
            preview[name] = value
    preview_path: Path | None = None
    if len(canonical) <= 2_000:
        detail = f"Arguments: {json.dumps(preview, ensure_ascii=False, sort_keys=True)}"
    else:
        preview_dir = hermes_home / "state" / "scaffolde-approval-previews"
        preview_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        preview_dir.chmod(0o700)
        fd, name = tempfile.mkstemp(prefix=f"{digest}-", suffix=".json", dir=preview_dir)
        preview_path = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical)
            handle.write("\n")
        preview_path.chmod(0o600)
        full_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        detail = (
            f"Exact payload SHA256={full_digest}, bytes={len(canonical.encode('utf-8'))}, "
            f"review_file={json.dumps(str(preview_path))}. Approve only after reviewing that exact 0600 preview file."
        )
    reason = f"Scaffolde capability {capability_id}.{operation} is a write operation and requires approval. {detail}"
    return reason, f"scaffolde:{capability_id}:{operation}:{digest}", preview_path


def invoke_capability(capability_id: str, operation: str, arguments: Mapping[str, Any], *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    registry = load_capability_registry()
    if registry.status != "valid":
        return {"status": "capability_degraded", "errors": registry.errors, "path": registry.path}
    desc = registry.capabilities.get(capability_id)
    if not desc:
        return {"status": "error", "error_type": "unknown_capability", "message": f"Unknown capability: {capability_id}"}
    op = desc.get("operations", {}).get(operation)
    if not op:
        return {"status": "error", "error_type": "unknown_operation", "message": f"Unknown operation: {operation}"}
    values, arg_error = _validate_arguments(op, arguments or {})
    if arg_error:
        return {"status": "error", "error_type": "invalid_arguments", "message": arg_error}
    if op.get("risk") == "write":
        from tools.approval import request_tool_approval
        approval_reason, approval_rule_key, preview_path = _approval_details(
            capability_id,
            operation,
            values or {},
            Path(get_hermes_home()),
        )
        try:
            approval = request_tool_approval(
                "scaffolde_capability",
                approval_reason,
                rule_key=approval_rule_key,
            )
        finally:
            if preview_path is not None:
                preview_path.unlink(missing_ok=True)
        if not approval.get("approved"):
            return {"status": "error", "error_type": "approval_required", "message": approval.get("message") or "approval denied"}
    bun_path = shutil.which("bun")
    if not bun_path:
        return {"status": "capability_degraded", "error_type": "missing_runtime", "message": "bun executable is unavailable"}
    try:
        argv = [bun_path, *map(str, desc["entrypoint"].get("args", [])), *[_render_template(a, values or {}) for a in op.get("argv", [])]]
    except ValueError as exc:
        return {"status": "error", "error_type": "template_error", "message": str(exc)}
    try:
        proc = subprocess.run(
            argv,
            cwd=desc["entrypoint"]["cwd"],
            env=_build_env(desc),
            shell=False,
            text=True,
            capture_output=True,
            timeout=max(1, min(int(timeout), 300)),
        )
    except subprocess.TimeoutExpired:
        return {"status": "capability_degraded", "error_type": "timeout", "message": f"Capability timed out after {timeout}s"}
    except Exception as exc:
        return {"status": "capability_degraded", "error_type": "execution_failed", "message": _redact(str(exc))}
    result = {
        "status": "ok" if proc.returncode == 0 else "capability_degraded",
        "returncode": proc.returncode,
        "stdout": _redact((proc.stdout or "")[:MAX_STDOUT_CHARS]),
        "stderr": _redact((proc.stderr or "")[:MAX_STDERR_CHARS]),
        "capability_id": capability_id,
        "operation": operation,
        "risk": op.get("risk"),
    }
    if proc.returncode != 0:
        result["error_type"] = "nonzero_exit"
    return result
