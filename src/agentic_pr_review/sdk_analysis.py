"""Static SDK app analysis helpers for connector PR review."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from typing import Any


RESERVED_ASSET_FIELD_NAMES = {"app_version", "directory", "ingest", "main_module", "appname"}
MAKE_REQUEST_ALLOWED_FIELDS = {
    "http_method",
    "endpoint",
    "headers",
    "query_parameters",
    "body",
    "timeout",
    "verify_ssl",
}


def build_sdk_review_inventory(review_input: dict[str, Any], full_files: dict[str, str] | None = None) -> dict[str, Any]:
    """Return compact SDK migration context for deterministic checks and prompts."""

    if full_files is None:
        full_files = active_full_files_for_sdk(review_input)
    base_files = review_input.get("base_files") or {}
    changed = changed_file_map(review_input)
    base_jsons = load_root_manifest_jsons(base_files)
    removed_jsons = load_removed_root_manifest_jsons(review_input)
    for path, manifest in removed_jsons.items():
        base_jsons.setdefault(path, manifest)
    legacy = legacy_manifest_inventory(base_jsons)
    pyproject = parse_pyproject(full_files.get("pyproject.toml", ""))
    main_module = pyproject_main_module(pyproject)
    app_file, app_instance = main_module_to_file_and_instance(main_module)
    python = python_sdk_inventory(full_files)
    app_instances = python["app_instances"]
    main_app_present = any(
        item["file"] == app_file and item["name"] == app_instance for item in app_instances
    )
    removed_root_json_files = [
        path
        for path, item in changed.items()
        if item.get("status") == "removed" and "/" not in path and path.endswith(".json")
    ]
    sdk_markers = {
        "has_pyproject_tool": bool(pyproject.get("tool", {}).get("soar", {}).get("app")),
        "has_soar_sdk_imports": any("soar_sdk" in text for text in full_files.values()),
        "has_app_instance": bool(app_instances),
    }
    is_sdk_app = any(sdk_markers.values())
    is_sdk_migration = bool(is_sdk_app and (removed_root_json_files or base_jsons))

    return {
        "is_sdk_app": is_sdk_app,
        "is_sdk_migration": is_sdk_migration,
        "removed_root_json_files": removed_root_json_files,
        "main_module": main_module,
        "main_module_file": app_file,
        "main_module_app_name": app_instance,
        "main_module_file_present": app_file in full_files,
        "main_app_instance_present": main_app_present,
        "sdk_markers": sdk_markers,
        "python": python,
        "legacy": legacy,
    }


def active_full_files_for_sdk(review_input: dict[str, Any]) -> dict[str, str]:
    full_files = review_input.get("full_files") or {}
    removed_paths = {
        str(item.get("filename"))
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("filename") and item.get("status") == "removed"
    }
    return {path: text for path, text in full_files.items() if path not in removed_paths}


def changed_file_map(review_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("filename")): item
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("filename")
    }


def load_root_manifest_jsons(files: dict[str, str]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for path, text in files.items():
        if "/" in path or not path.endswith(".json"):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("actions"), list):
            manifests[path] = data
    return manifests


def load_removed_root_manifest_jsons(review_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load removed root app JSON from collected deleted-file evidence.

    SDK migration PRs often delete the legacy root manifest. GitHub's contents
    API can fail to fetch that file from the base ref in forks, but the PR file
    API may still provide deleted raw content. Use it only as legacy comparison
    input, never as current SDK metadata.
    """

    full_files = review_input.get("full_files") or {}
    changed = changed_file_map(review_input)
    removed_paths = {
        path
        for path, item in changed.items()
        if item.get("status") == "removed" and "/" not in path and path.endswith(".json")
    }
    candidates = {path: text for path, text in full_files.items() if path in removed_paths}
    return load_root_manifest_jsons(candidates)


def parse_pyproject(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def pyproject_main_module(pyproject: dict[str, Any]) -> str:
    value = (
        pyproject.get("tool", {})
        .get("soar", {})
        .get("app", {})
        .get("main_module")
    )
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "src.app:app"


def main_module_to_file_and_instance(main_module: str) -> tuple[str, str]:
    module_part, _, instance = main_module.partition(":")
    instance = instance or "app"
    if module_part.endswith(".py"):
        path = module_part
    else:
        path = module_part.replace(".", "/") + ".py"
    return path, instance


def legacy_manifest_inventory(app_jsons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    action_identifiers: set[str] = set()
    config_fields: set[str] = set()
    custom_view_actions: set[str] = set()
    summary_actions: set[str] = set()
    action_output_paths: dict[str, list[str]] = {}
    action_summary_fields: dict[str, list[str]] = {}
    has_rest_handler = False
    has_webhooks = False
    manifest_paths = sorted(app_jsons)

    for app_json in app_jsons.values():
        has_rest_handler = has_rest_handler or isinstance(app_json.get("rest_handler"), str)
        has_webhooks = has_webhooks or bool(app_json.get("webhooks"))

        config = app_json.get("configuration") or app_json.get("asset_params") or {}
        config_items = config.items() if isinstance(config, dict) else enumerate(config if isinstance(config, list) else [])
        for key, item in config_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("data_type") or "").startswith("ph"):
                continue
            name = str(item.get("name") or item.get("key") or item.get("identifier") or key or "").strip()
            if name:
                config_fields.add(name)

        for action in app_json.get("actions", []):
            if not isinstance(action, dict):
                continue
            identifier = str(action.get("identifier") or "").strip()
            if identifier:
                action_identifiers.add(identifier)
            render = action.get("render") if isinstance(action.get("render"), dict) else {}
            if render.get("type") == "custom":
                custom_view_actions.add(identifier or str(action.get("action") or "").strip())
            output = action.get("output") or []
            paths = [
                str(item.get("data_path") or "")
                for item in output
                if isinstance(item, dict) and item.get("data_path")
            ]
            if identifier and paths:
                action_output_paths[identifier] = sorted(set(paths))
                summary_fields = sorted(
                    {
                        match.group("field")
                        for path in paths
                        for match in [re.search(r"action_result\.summary\.(?P<field>[A-Za-z0-9_]+)", path)]
                        if match
                    }
                )
                if summary_fields:
                    action_summary_fields[identifier] = summary_fields
            if any(
                isinstance(item, dict) and "action_result.summary." in str(item.get("data_path") or "")
                for item in output
            ):
                summary_actions.add(identifier or str(action.get("action") or "").strip())

    return {
        "manifest_paths": manifest_paths,
        "action_identifiers": sorted(action_identifiers),
        "config_fields": sorted(config_fields),
        "custom_view_actions": sorted(name for name in custom_view_actions if name),
        "summary_actions": sorted(name for name in summary_actions if name),
        "action_output_paths": action_output_paths,
        "action_summary_fields": action_summary_fields,
        "has_rest_handler": has_rest_handler,
        "has_webhooks": has_webhooks,
    }


def python_sdk_inventory(full_files: dict[str, str]) -> dict[str, Any]:
    app_instances: list[dict[str, Any]] = []
    asset_fields: dict[str, dict[str, Any]] = {}
    action_registrations: list[dict[str, Any]] = []
    test_connectivity: list[dict[str, Any]] = []
    view_handlers: list[dict[str, Any]] = []
    webhooks: list[dict[str, Any]] = []
    param_fields: list[dict[str, Any]] = []
    unsupported_param_fields: list[dict[str, Any]] = []
    invalid_make_request_param_fields: list[dict[str, Any]] = []
    function_defs: list[dict[str, Any]] = []
    enable_webhooks_count = 0
    cli_invocation_present = False

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        visitor = SDKVisitor(path)
        visitor.visit(tree)
        app_instances.extend(visitor.app_instances)
        asset_fields.update(visitor.asset_fields)
        action_registrations.extend(visitor.action_registrations)
        test_connectivity.extend(visitor.test_connectivity)
        view_handlers.extend(visitor.view_handlers)
        webhooks.extend(visitor.webhooks)
        param_fields.extend(visitor.param_fields)
        unsupported_param_fields.extend(visitor.unsupported_param_fields)
        invalid_make_request_param_fields.extend(visitor.invalid_make_request_param_fields)
        function_defs.extend(visitor.function_defs)
        enable_webhooks_count += visitor.enable_webhooks_count
        cli_invocation_present = cli_invocation_present or visitor.cli_invocation_present

    enrich_action_registrations(action_registrations, function_defs)
    enrich_test_connectivity(test_connectivity, function_defs)

    return {
        "app_instances": app_instances,
        "asset_fields": sorted(asset_fields.values(), key=lambda item: (item["file"], item["line"], item["name"])),
        "asset_field_names": sorted(asset_fields),
        "action_registrations": action_registrations,
        "registered_action_identifiers": sorted({item["identifier"] for item in action_registrations if item.get("identifier")}),
        "test_connectivity": test_connectivity,
        "test_connectivity_count": len(test_connectivity),
        "view_handlers": view_handlers,
        "view_handler_count": len(view_handlers),
        "webhooks": webhooks,
        "webhook_count": len(webhooks),
        "enable_webhooks_count": enable_webhooks_count,
        "param_fields": param_fields,
        "unsupported_param_fields": unsupported_param_fields,
        "invalid_make_request_param_fields": invalid_make_request_param_fields,
        "function_defs": function_defs,
        "cli_invocation_present": cli_invocation_present,
    }


class SDKVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.app_instances: list[dict[str, Any]] = []
        self.asset_fields: dict[str, dict[str, Any]] = {}
        self.action_registrations: list[dict[str, Any]] = []
        self.test_connectivity: list[dict[str, Any]] = []
        self.view_handlers: list[dict[str, Any]] = []
        self.webhooks: list[dict[str, Any]] = []
        self.param_fields: list[dict[str, Any]] = []
        self.unsupported_param_fields: list[dict[str, Any]] = []
        self.invalid_make_request_param_fields: list[dict[str, Any]] = []
        self.function_defs: list[dict[str, Any]] = []
        self.enable_webhooks_count = 0
        self.cli_invocation_present = False
        self._class_stack: list[str] = []
        self._class_bases: list[set[str]] = []

    def visit_Assign(self, node: ast.Assign) -> Any:  # noqa: ANN401
        if is_call_named(node.value, "App"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.app_instances.append(app_instance_info(self.path, target.id, node.value, node.lineno))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:  # noqa: ANN401
        if is_call_named(node.value, "App") and isinstance(node.target, ast.Name):
            self.app_instances.append(app_instance_info(self.path, node.target.id, node.value, node.lineno))
        if self._class_stack and isinstance(node.target, ast.Name):
            current_bases = self._class_bases[-1]
            if "BaseAsset" in current_bases:
                field = asset_field_info(self.path, node)
                if field:
                    self.asset_fields[field["manifest_name"]] = field
            if current_bases & {"Params", "MakeRequestParams", "OnPollParams", "OnESPollParams"}:
                param_field = param_field_info(self.path, self._class_stack[-1], node)
                self.param_fields.append(param_field)
                if annotation_has_list(node.annotation):
                    self.unsupported_param_fields.append(param_field)
            if "MakeRequestParams" in current_bases and node.target.id not in MAKE_REQUEST_ALLOWED_FIELDS:
                self.invalid_make_request_param_fields.append(
                    param_field_info(self.path, self._class_stack[-1], node)
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # noqa: ANN401
        base_names = {base_name(base) for base in node.bases}
        self._class_stack.append(node.name)
        self._class_bases.append(base_names)
        self.generic_visit(node)
        self._class_stack.pop()
        self._class_bases.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: ANN401
        self._visit_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # noqa: ANN401
        self._visit_function(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: ANN401
        name = call_name(node.func)
        if name.endswith(".register_action"):
            self.action_registrations.append(register_action_info(self.path, node))
        elif name.endswith(".action") and node.args:
            self.action_registrations.append(call_style_action_info(self.path, node, "action"))
        elif name.endswith(".make_request") and node.args:
            self.action_registrations.append(call_style_action_info(self.path, node, "make_request", identifier="make_request"))
        elif name.endswith(".test_connectivity") and node.args:
            self.test_connectivity.append(call_style_test_connectivity_info(self.path, node))
            self.action_registrations.append(call_style_action_info(self.path, node, "test_connectivity", identifier="test_connectivity"))
        elif name.endswith(".on_poll") and node.args:
            self.action_registrations.append(call_style_action_info(self.path, node, "on_poll", identifier="on_poll"))
        elif name.endswith(".on_es_poll") and node.args:
            self.action_registrations.append(call_style_action_info(self.path, node, "on_es_poll", identifier="on_es_poll"))
        elif name.endswith(".enable_webhooks"):
            self.enable_webhooks_count += 1
        elif name.endswith(".cli"):
            self.cli_invocation_present = True
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        first_arg = node.args.args[0] if node.args.args else None
        self.function_defs.append(
            {
                "file": self.path,
                "line": node.lineno,
                "function": node.name,
                "args": [arg.arg for arg in node.args.args],
                "arg_annotations": arg_annotations(node),
                "return_annotation": safe_unparse(node.returns) if node.returns else None,
                "first_param": first_arg.arg if first_arg else None,
                "first_param_annotation": safe_unparse(first_arg.annotation) if first_arg and first_arg.annotation else None,
                "returns_value": any(isinstance(child, ast.Return) and child.value is not None for child in ast.walk(node)),
            }
        )
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            func = call.func if call else decorator
            name = call_name(func)
            if name.endswith(".action"):
                self.action_registrations.append(decorated_action_info(self.path, node, call, "action"))
            elif name.endswith(".make_request"):
                self.action_registrations.append(decorated_action_info(self.path, node, call, "make_request", identifier="make_request"))
            elif name.endswith(".on_poll"):
                self.action_registrations.append(decorated_action_info(self.path, node, call, "on_poll", identifier="on_poll"))
            elif name.endswith(".on_es_poll"):
                self.action_registrations.append(decorated_action_info(self.path, node, call, "on_es_poll", identifier="on_es_poll"))
            elif name.endswith(".test_connectivity"):
                self.test_connectivity.append(test_connectivity_info(self.path, node))
                self.action_registrations.append(decorated_action_info(self.path, node, call, "test_connectivity", identifier="test_connectivity"))
            elif name.endswith(".view_handler"):
                self.view_handlers.append(
                    {
                        "file": self.path,
                        "line": node.lineno,
                        "function": node.name,
                        "template": literal_keyword(call, "template") if call else None,
                    }
                )
            elif name.endswith(".webhook"):
                self.webhooks.append(
                    {
                        "file": self.path,
                        "line": node.lineno,
                        "function": node.name,
                        "pattern": literal_arg(call, 0) if call else None,
                    }
                )


def enrich_action_registrations(
    action_registrations: list[dict[str, Any]],
    function_defs: list[dict[str, Any]],
) -> None:
    functions_by_name: dict[str, list[dict[str, Any]]] = {}
    for function in function_defs:
        functions_by_name.setdefault(str(function.get("function")), []).append(function)

    for registration in action_registrations:
        function_name = registration.get("function")
        if not function_name:
            continue
        matches = functions_by_name.get(str(function_name), [])
        if len(matches) != 1:
            continue
        target = matches[0]
        registration.setdefault("implementation_file", target.get("file"))
        registration.setdefault("implementation_line", target.get("line"))
        if not registration.get("return_annotation"):
            registration["return_annotation"] = target.get("return_annotation")
        if not registration.get("first_param"):
            registration["first_param"] = target.get("first_param")
        if not registration.get("first_param_annotation"):
            registration["first_param_annotation"] = target.get("first_param_annotation")
        if not registration.get("args"):
            registration["args"] = target.get("args") or []
        if not registration.get("arg_annotations"):
            registration["arg_annotations"] = target.get("arg_annotations") or {}
        registration["signature_visible"] = True


def enrich_test_connectivity(
    test_connectivity: list[dict[str, Any]],
    function_defs: list[dict[str, Any]],
) -> None:
    functions_by_name: dict[str, list[dict[str, Any]]] = {}
    for function in function_defs:
        functions_by_name.setdefault(str(function.get("function")), []).append(function)

    for registration in test_connectivity:
        function_name = registration.get("function")
        if not function_name:
            continue
        matches = functions_by_name.get(str(function_name), [])
        if len(matches) != 1:
            continue
        target = matches[0]
        registration.setdefault("implementation_file", target.get("file"))
        registration.setdefault("implementation_line", target.get("line"))
        registration["return_annotation"] = target.get("return_annotation")
        registration["args"] = target.get("args") or []
        registration["arg_annotations"] = target.get("arg_annotations") or {}
        registration["returns_value"] = target.get("returns_value", registration.get("returns_value", False))


def app_instance_info(path: str, name: str, call: ast.Call, line: int) -> dict[str, Any]:
    kwargs = {kw.arg: safe_unparse(kw.value) for kw in call.keywords if kw.arg}
    return {
        "file": path,
        "line": line,
        "name": name,
        "asset_cls": strip_quotes(kwargs.get("asset_cls")),
        "appid": strip_quotes(kwargs.get("appid")),
        "app_type": strip_quotes(kwargs.get("app_type")),
        "product_name": strip_quotes(kwargs.get("product_name")),
        "min_phantom_version": strip_quotes(kwargs.get("min_phantom_version")),
        "kwargs": kwargs,
    }


def asset_field_info(path: str, node: ast.AnnAssign) -> dict[str, Any] | None:
    if not isinstance(node.target, ast.Name):
        return None
    name = node.target.id
    manifest_name = name
    sensitive = False
    is_file = False
    if isinstance(node.value, ast.Call) and base_name(node.value.func) == "AssetField":
        alias = literal_keyword(node.value, "alias")
        if alias:
            manifest_name = alias
        sensitive = literal_keyword(node.value, "sensitive") is True
        is_file = literal_keyword(node.value, "is_file") is True
    return {
        "file": path,
        "line": node.lineno,
        "name": name,
        "manifest_name": manifest_name,
        "annotation": safe_unparse(node.annotation),
        "sensitive": sensitive,
        "is_file": is_file,
        "reserved": name.startswith("_reserved_") or name in RESERVED_ASSET_FIELD_NAMES,
    }


def param_field_info(path: str, class_name: str, node: ast.AnnAssign) -> dict[str, Any]:
    sensitive = False
    if isinstance(node.value, ast.Call) and base_name(node.value.func) == "Param":
        sensitive = literal_keyword(node.value, "sensitive") is True
    return {
        "file": path,
        "line": node.lineno,
        "class_name": class_name,
        "field": node.target.id if isinstance(node.target, ast.Name) else "",
        "annotation": safe_unparse(node.annotation),
        "sensitive": sensitive,
    }


def decorated_action_info(
    path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    decorator_call: ast.Call | None,
    registration_type: str,
    *,
    identifier: str | None = None,
) -> dict[str, Any]:
    identifier = identifier or str(literal_keyword(decorator_call, "identifier") or node.name)
    read_only = literal_keyword(decorator_call, "read_only") if decorator_call else None
    action_type = literal_keyword(decorator_call, "action_type") if decorator_call else None
    summary_type = literal_keyword(decorator_call, "summary_type") if decorator_call else None
    first_arg = node.args.args[0] if node.args.args else None
    return {
        "file": path,
        "line": node.lineno,
        "identifier": identifier,
        "function": node.name,
        "registration_type": registration_type,
        "signature_visible": True,
        "read_only": read_only,
        "action_type": action_type,
        "summary_type": summary_type,
        "params_class": literal_keyword(decorator_call, "params_class") if decorator_call else None,
        "output_class": literal_keyword(decorator_call, "output_class") if decorator_call else None,
        "view_handler": literal_keyword(decorator_call, "view_handler") if decorator_call else None,
        "args": [arg.arg for arg in node.args.args],
        "arg_annotations": arg_annotations(node),
        "return_annotation": safe_unparse(node.returns) if node.returns else None,
        "first_param": first_arg.arg if first_arg else None,
        "first_param_annotation": safe_unparse(first_arg.annotation) if first_arg and first_arg.annotation else None,
    }


def register_action_info(path: str, node: ast.Call) -> dict[str, Any]:
    action_ref = literal_keyword(node, "action") or literal_arg(node, 0)
    function = function_name_from_ref(action_ref)
    identifier = literal_keyword(node, "identifier") or function
    return {
        "file": path,
        "line": node.lineno,
        "identifier": identifier,
        "function": function,
        "registration_type": "register_action",
        "signature_visible": False,
        "action_ref": action_ref,
        "read_only": literal_keyword(node, "read_only"),
        "action_type": literal_keyword(node, "action_type"),
        "summary_type": literal_keyword(node, "summary_type"),
        "params_class": literal_keyword(node, "params_class"),
        "output_class": literal_keyword(node, "output_class"),
        "args": [],
        "arg_annotations": {},
        "return_annotation": None,
        "first_param": None,
        "first_param_annotation": None,
        "view_handler": literal_keyword(node, "view_handler"),
    }


def call_style_action_info(
    path: str,
    node: ast.Call,
    registration_type: str,
    *,
    identifier: str | None = None,
) -> dict[str, Any]:
    decorator_call = node.func if isinstance(node.func, ast.Call) else None
    function_ref = literal_arg(node, 0)
    function = function_name_from_ref(function_ref)
    identifier = identifier or literal_keyword(decorator_call, "identifier") or function
    return {
        "file": path,
        "line": node.lineno,
        "identifier": identifier,
        "function": function,
        "registration_type": registration_type,
        "signature_visible": False,
        "action_ref": function_ref,
        "read_only": literal_keyword(decorator_call, "read_only"),
        "action_type": literal_keyword(decorator_call, "action_type"),
        "summary_type": literal_keyword(decorator_call, "summary_type"),
        "params_class": literal_keyword(decorator_call, "params_class"),
        "output_class": literal_keyword(decorator_call, "output_class"),
        "args": [],
        "arg_annotations": {},
        "return_annotation": None,
        "first_param": None,
        "first_param_annotation": None,
        "view_handler": literal_keyword(decorator_call, "view_handler"),
    }


def call_style_test_connectivity_info(path: str, node: ast.Call) -> dict[str, Any]:
    function_ref = literal_arg(node, 0)
    function = function_name_from_ref(function_ref)
    return {
        "file": path,
        "line": node.lineno,
        "function": function or str(function_ref or "test_connectivity"),
        "return_annotation": None,
        "args": [],
        "arg_annotations": {},
        "returns_value": False,
        "registration_type": "call_style",
    }


def test_connectivity_info(path: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "file": path,
        "line": node.lineno,
        "function": node.name,
        "return_annotation": safe_unparse(node.returns) if node.returns else None,
        "args": [arg.arg for arg in node.args.args],
        "arg_annotations": {
            arg.arg: safe_unparse(arg.annotation)
            for arg in node.args.args
            if arg.annotation is not None
        },
        "returns_value": any(isinstance(child, ast.Return) and child.value is not None for child in ast.walk(node)),
    }


def arg_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    return {
        arg.arg: safe_unparse(arg.annotation) or ""
        for arg in node.args.args
        if arg.annotation is not None
    }


def function_name_from_ref(action_ref: Any) -> str | None:  # noqa: ANN401
    if not isinstance(action_ref, str) or not action_ref:
        return None
    return action_ref.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def is_call_named(node: ast.AST | None, name: str) -> bool:
    return isinstance(node, ast.Call) and base_name(node.func) == name


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return ""


def base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return base_name(node.value)
    if isinstance(node, ast.Call):
        return base_name(node.func)
    return ""


def literal_arg(call: ast.Call | None, index: int) -> Any:  # noqa: ANN401
    if call is None or len(call.args) <= index:
        return None
    return literal_value(call.args[index])


def literal_keyword(call: ast.Call | None, name: str) -> Any:  # noqa: ANN401
    if call is None:
        return None
    for keyword in call.keywords:
        if keyword.arg == name:
            return literal_value(keyword.value)
    return None


def literal_value(node: ast.AST) -> Any:  # noqa: ANN401
    try:
        return ast.literal_eval(node)
    except Exception:
        return safe_unparse(node)


def safe_unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def strip_quotes(value: Any) -> Any:  # noqa: ANN401
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"['\"](.*)['\"]", value)
    return match.group(1) if match else value


def annotation_has_list(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    text = safe_unparse(annotation) or ""
    return bool(re.search(r"\b(list|List|Sequence|Iterable)\s*\[", text))


def annotation_is_string_like_text(annotation: str | None) -> bool:
    if not annotation:
        return False
    compact = re.sub(r"\s+", "", annotation)
    return compact in {
        "str",
        "builtins.str",
        "str|None",
        "None|str",
        "Optional[str]",
        "typing.Optional[str]",
    }
