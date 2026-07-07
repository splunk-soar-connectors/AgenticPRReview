"""Build a public-safe historical training examples file.

The raw mining outputs under ``runs/`` can contain repo names, users, URLs,
comments, CI logs, and local paths. This script keeps the retrieval signal the
bot uses at runtime while replacing that raw evidence with synthetic,
allowlisted examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from agentic_pr_review.historical_context import CATEGORY_KEYWORDS


CompiledReplacements = tuple[re.Pattern[str] | None, dict[str, str]]

DEFAULT_INPUT_GLOBS = (
    "runs/training-mining-full/*/training_examples.jsonl",
    "runs/training-mining-pipeline-validation/*/training_examples.jsonl",
    "runs/training-mining-known/*/training_examples.jsonl",
    "runs/training-mining/*/training_examples.jsonl",
)
DEFAULT_OUTPUT = Path("runs/training-mining-known/public-sanitized/training_examples.jsonl")
DEFAULT_REPLACEMENT_FILE = Path("runs/private_sanitizer_replacements.txt")

AUTO_REPLACEMENT_STOPWORDS = {
    "action",
    "actions",
    "api",
    "app",
    "apps",
    "auth",
    "check",
    "cloud",
    "connector",
    "connectors",
    "docs",
    "event",
    "events",
    "file",
    "files",
    "github",
    "json",
    "metadata",
    "oauth",
    "poll",
    "polling",
    "readme",
    "rest",
    "sdk",
    "security",
    "soar",
    "test",
    "tests",
    "token",
    "user",
    "vendor",
}

SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"https?://\S+", re.IGNORECASE), "[url]"),
    (re.compile(r"\bgithub\.com/[^\s)>\"]+", re.IGNORECASE), "[url]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[email]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[ip]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "[token]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[token]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[token]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[token]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "[private-key]"),
    (re.compile(r"\b/(?:home|users|tmp|var|private|ci-runner)/[^\s'\")]+", re.IGNORECASE), "[path]"),
    (re.compile(r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*){2,}\b", re.IGNORECASE), "[host]"),
)

PUBLIC_BLOCKLIST: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY", re.IGNORECASE)),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("URL", re.compile(r"https?://|github\.com/", re.IGNORECASE)),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("IPv4 address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("local path", re.compile(r"\b/(?:home|users|tmp|var|private|ci-runner)/", re.IGNORECASE)),
)

CATEGORY_DEFAULT_PATHS = {
    "api_auth_correctness": "connector.py",
    "api_contract": "connector.py",
    "app_mapping": "metadata/app_mapping.json",
    "dependency_packaging": "requirements.txt",
    "docs_pr_accuracy": "manual_readme_content.md",
    "error_handling": "connector.py",
    "missing_tests": "tests/test_connector.py",
    "output_schema_mismatch": "app.json",
    "pagination": "connector.py",
    "polling_checkpoint": "connector.py",
    "precommit": "connector.py",
    "soar_metadata": "app.json",
    "static_sanity_tests": "app.json",
    "unsafe_logging": "connector.py",
    "validation": "connector.py",
}

CATEGORY_PATCH_EXCERPTS = {
    "api_auth_correctness": "+response = requests.post(token_url, json=payload, timeout=timeout)",
    "api_contract": "+response = requests.post(api_url, json=payload, timeout=timeout)",
    "app_mapping": "+  \"example_app_id\": \"example_package\"",
    "dependency_packaging": "+example-dependency==1.2.3",
    "docs_pr_accuracy": "+Document the changed default and regenerated output.",
    "error_handling": "+result = client.call_api(payload)",
    "missing_tests": "+def test_changed_action_handles_error_path():",
    "output_schema_mismatch": "+action_result.add_data(response_json)",
    "pagination": "+response = requests.get(url, params=params, timeout=timeout)",
    "polling_checkpoint": "+self.save_state({\"checkpoint\": latest_timestamp})",
    "precommit": "+# sanitized hook failure evidence",
    "soar_metadata": "+  \"read_only\": true",
    "static_sanity_tests": "+  \"min_platform_version\": \"x.y.z\"",
    "unsafe_logging": "+self.debug_print(response_json)",
    "validation": "+value = params.get(\"field\")",
}

CI_RELEVANT_CATEGORIES = {
    "precommit",
    "static_sanity_tests",
    "missing_tests",
    "dependency_packaging",
    "docs_pr_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input training_examples.jsonl path. Defaults to all private mining outputs under runs/.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Public-safe JSONL output path.")
    parser.add_argument(
        "--max-per-pattern",
        type=int,
        default=0,
        help="Optional cap per pattern_id. 0 keeps every sanitized source example.",
    )
    parser.add_argument(
        "--sensitive-term",
        action="append",
        default=[],
        metavar="RAW=REPLACEMENT",
        help="Exact text replacement to apply before public validation. Repeat as needed.",
    )
    parser.add_argument(
        "--replacement-file",
        default=str(DEFAULT_REPLACEMENT_FILE) if DEFAULT_REPLACEMENT_FILE.is_file() else None,
        help=(
            "Optional ignored local file of RAW=REPLACEMENT lines. Use this for "
            "organization-specific terms that should not be committed. Defaults to "
            "runs/private_sanitizer_replacements.txt when that ignored file exists."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    input_paths = discover_input_paths(args.input, output=output)
    if not input_paths:
        raise SystemExit("No training_examples.jsonl files found to sanitize.")

    raw_examples = load_examples(input_paths)
    replacements = build_replacements(raw_examples, args.sensitive_term, args.replacement_file)
    sanitized = sanitize_examples(raw_examples, max_per_pattern=args.max_per_pattern, replacements=replacements)
    rendered = render_jsonl(sanitized)
    findings = find_public_blockers(rendered)
    if findings:
        joined = "\n".join(f"- {kind}: {sample}" for kind, sample in findings[:20])
        raise SystemExit(f"Sanitized output still contains blocked public content:\n{joined}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")

    categories = Counter(str(example.get("category") or "general") for example in sanitized)
    print(f"Wrote {len(sanitized)} sanitized examples to {output}")
    print("Categories: " + ", ".join(f"{name}={count}" for name, count in sorted(categories.items())))
    return 0


def discover_input_paths(inputs: list[str], *, output: Path) -> list[Path]:
    if inputs:
        paths = [Path(value) for value in inputs]
    else:
        paths = []
        for pattern in DEFAULT_INPUT_GLOBS:
            paths.extend(Path.cwd().glob(pattern))

    output_resolved = output.resolve()
    unique = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved == output_resolved or resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)


def load_examples(paths: list[Path]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if isinstance(value, dict):
                    examples.append(value)
    return examples


def load_sensitive_replacements(items: list[str], replacement_file: str | None) -> dict[str, str]:
    replacements: dict[str, str] = {}
    lines: list[str] = []
    if replacement_file:
        path = Path(replacement_file)
        if not path.is_file():
            raise SystemExit(f"Replacement file does not exist: {replacement_file}")
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    lines.extend(items)
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"Invalid replacement entry, expected RAW=REPLACEMENT: {raw_line}")
        source, replacement = line.split("=", 1)
        source = source.strip()
        replacement = replacement.strip()
        if source:
            replacements[source] = replacement
    return replacements


def build_replacements(
    raw_examples: list[dict[str, Any]],
    sensitive_terms: list[str],
    replacement_file: str | None,
) -> dict[str, str]:
    return merge_replacements(
        build_auto_replacements(raw_examples),
        load_sensitive_replacements(sensitive_terms, replacement_file),
    )


def merge_replacements(auto_replacements: dict[str, str], explicit_replacements: dict[str, str]) -> dict[str, str]:
    replacements = dict(auto_replacements)
    for source, replacement in explicit_replacements.items():
        for existing in list(replacements):
            if existing.lower() == source.lower():
                replacements.pop(existing)
        replacements[source] = replacement
    return replacements


def build_auto_replacements(raw_examples: list[dict[str, Any]]) -> dict[str, str]:
    """Derive safe replacements from metadata that should never survive public output."""

    replacements: dict[str, str] = {}
    for item in raw_examples:
        repo = str(item.get("repo") or "")
        if repo:
            add_replacement_variants(replacements, repo, "example-org/example-repo")
            if "/" in repo:
                owner, name = repo.split("/", 1)
                add_replacement_variants(replacements, owner, "example-org")
                add_replacement_variants(replacements, name, "example-repo")

        for key in ("user", "author", "comment_user", "reviewer", "sender"):
            value = item.get(key)
            if isinstance(value, str):
                add_replacement_variants(replacements, value, "reviewer")
            elif isinstance(value, dict):
                for nested_key in ("login", "name", "email"):
                    nested = value.get(nested_key)
                    if isinstance(nested, str):
                        add_replacement_variants(replacements, nested, "reviewer")

        for raw_path in example_source_paths(item):
            for term in path_private_terms(raw_path):
                add_replacement_variants(replacements, term, "example")

    return replacements


def add_replacement_variants(replacements: dict[str, str], value: str, replacement: str) -> None:
    text = str(value or "").strip()
    if not is_private_replacement_term(text):
        return
    variants = {
        text,
        text.lower(),
        text.replace("_", "-"),
        text.replace("-", "_"),
        text.replace("-", " "),
        text.replace("_", " "),
    }
    for token in re.split(r"[/_.\-\s]+", text):
        if is_private_replacement_term(token):
            variants.add(token)
    for variant in variants:
        cleaned = " ".join(str(variant).split())
        if is_private_replacement_term(cleaned):
            replacements.setdefault(cleaned, replacement)


def is_private_replacement_term(value: str) -> bool:
    text = str(value or "").strip().strip("/._-")
    if len(text) < 4:
        return False
    lowered = text.lower()
    if lowered in AUTO_REPLACEMENT_STOPWORDS:
        return False
    if re.fullmatch(r"\d+", lowered):
        return False
    if re.fullmatch(r"repo-\d+|pr-\d+", lowered):
        return False
    return bool(re.search(r"[a-zA-Z]", text))


def example_source_paths(item: dict[str, Any]) -> list[str]:
    paths = [str(item.get("path") or "")]
    paths.extend(str(value or "") for value in item.get("changed_files") or [])
    for evidence in item.get("file_evidence") or []:
        if isinstance(evidence, dict):
            paths.append(str(evidence.get("filename") or ""))
    return [path for path in paths if path]


def path_private_terms(raw_path: str) -> set[str]:
    path = PurePosixPath(str(raw_path or "").replace("\\", "/").lower())
    terms: set[str] = set()
    for part in path.parts:
        stem = PurePosixPath(part).stem
        if stem.endswith("_connector"):
            terms.add(stem.removesuffix("_connector"))
        if stem.endswith("-connector"):
            terms.add(stem.removesuffix("-connector"))
        if stem.endswith("_app"):
            terms.add(stem.removesuffix("_app"))
        if stem.endswith("-app"):
            terms.add(stem.removesuffix("-app"))
    return terms


def sanitize_examples(
    raw_examples: list[dict[str, Any]],
    *,
    max_per_pattern: int = 0,
    replacements: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    effective_replacements = merge_replacements(
        build_auto_replacements(raw_examples),
        replacements or {},
    )
    compiled_replacements = compile_replacements(effective_replacements)
    repo_aliases = alias_map(
        (str(item.get("repo") or "") for item in raw_examples if item.get("repo")),
        prefix="repo",
    )
    pr_aliases = alias_map(
        (
            f"{item.get('repo') or ''}#{item.get('pr_number') or ''}"
            for item in raw_examples
            if item.get("repo") or item.get("pr_number")
        ),
        prefix="pr",
    )

    output: list[dict[str, Any]] = []
    pattern_counts: Counter[str] = Counter()
    for item in raw_examples:
        pattern_id = sanitize_identifier(item.get("pattern_id"))
        category = sanitize_identifier(item.get("category"))
        if not pattern_id or not category:
            continue
        if max_per_pattern and pattern_counts[pattern_id] >= max_per_pattern:
            continue
        pattern_counts[pattern_id] += 1
        output.append(
            sanitize_example(
                item,
                repo_aliases=repo_aliases,
                pr_aliases=pr_aliases,
                replacements=compiled_replacements,
            )
        )

    output.sort(
        key=lambda item: (
            str(item.get("pattern_id") or ""),
            str(item.get("category") or ""),
            str(item.get("repo") or ""),
            int(item.get("pr_number") or 0),
            str(item.get("path") or ""),
        )
    )
    return output


def sanitize_example(
    item: dict[str, Any],
    *,
    repo_aliases: dict[str, str],
    pr_aliases: dict[str, str],
    replacements: CompiledReplacements,
) -> dict[str, Any]:
    pattern_id = sanitize_identifier(item.get("pattern_id"))
    category = sanitize_identifier(item.get("category"))
    title = sanitize_text(str(item.get("pattern_title") or pattern_id.replace("_", " ")).strip(), replacements)
    hint = sanitize_text(
        str(item.get("implementation_hint") or "Review current evidence for the same issue.").strip(),
        replacements,
    )
    changed_files = sanitized_changed_files(item, category=category)
    path = sanitize_path(item.get("path"), category=category) or changed_files[0]
    repo_key = str(item.get("repo") or "")
    pr_key = f"{repo_key}#{item.get('pr_number') or ''}"

    sanitized: dict[str, Any] = {
        "pattern_id": pattern_id,
        "category": category,
        "pattern_title": title,
        "implementation_hint": hint,
        "repo": repo_aliases.get(repo_key, "repo-0000"),
        "pr_number": int(pr_aliases.get(pr_key, "0").removeprefix("pr-") or 0),
        "pr_title": f"{title} training case",
        "pr_state": "closed",
        "comment_kind": sanitize_comment_kind(item.get("comment_kind")),
        "path": path,
        "line": None,
        "user": "human-reviewer",
        "body": synthetic_body(title=title, hint=hint, category=category, path=path, replacements=replacements),
        "diff_hunk": CATEGORY_PATCH_EXCERPTS.get(category, "+# sanitized training evidence"),
        "changed_files": changed_files,
        "file_evidence": [
            {
                "filename": filename,
                "status": "modified",
                "patch_excerpt": CATEGORY_PATCH_EXCERPTS.get(category, "+# sanitized training evidence"),
            }
            for filename in changed_files[:2]
        ],
        "check_evidence": sanitized_check_evidence(item, category=category, title=title, hint=hint),
    }
    return sanitized


def alias_map(values: Any, *, prefix: str) -> dict[str, str]:
    unique = sorted({str(value) for value in values if str(value)})
    return {value: f"{prefix}-{index:04d}" for index, value in enumerate(unique, start=1)}


def sanitized_changed_files(item: dict[str, Any], *, category: str) -> list[str]:
    candidates: list[str] = []
    candidates.append(str(item.get("path") or ""))
    candidates.extend(str(value or "") for value in item.get("changed_files") or [])
    for evidence in item.get("file_evidence") or []:
        if isinstance(evidence, dict):
            candidates.append(str(evidence.get("filename") or ""))

    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = sanitize_path(candidate, category=category)
        if not path or path in seen:
            continue
        seen.add(path)
        output.append(path)
    if not output:
        output.append(CATEGORY_DEFAULT_PATHS.get(category, "connector.py"))
    return output[:8]


def sanitize_path(raw_path: Any, *, category: str) -> str | None:
    if raw_path is None:
        return None
    raw = str(raw_path).strip().replace("\\", "/")
    if not raw:
        return None
    path = PurePosixPath(raw.lower())
    basename = path.name
    suffix = path.suffix

    if basename in {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "uv.lock", "poetry.lock"}:
        return basename
    if basename in {"readme.md", "readme.html", "manual_readme_content.md"}:
        return basename
    if "release_notes" in raw.lower() or "unreleased" in basename:
        return "release_notes/unreleased.md"
    if ".github" in path.parts or "workflow" in raw.lower():
        return ".github/workflows/ci.yml"
    if "appid_to_name" in raw.lower() or "appid_to_package_name" in raw.lower() or "mapping" in raw.lower():
        return "metadata/app_mapping.json"
    if suffix == ".py" and ("test" in raw.lower() or basename.startswith("test_")):
        return "tests/test_connector.py"
    if suffix == ".py":
        return "connector.py"
    if suffix == ".json":
        if "test" in raw.lower():
            return "tests/action_test.json"
        if category in {"soar_metadata", "output_schema_mismatch", "static_sanity_tests"} or "app" in basename:
            return "app.json"
        return "config/example.json"
    if suffix in {".yaml", ".yml"}:
        return ".github/workflows/ci.yml"
    if suffix == ".md":
        return "docs/example.md"
    if suffix:
        return f"example{suffix}"
    return CATEGORY_DEFAULT_PATHS.get(category, "connector.py")


def sanitized_check_evidence(item: dict[str, Any], *, category: str, title: str, hint: str) -> list[dict[str, Any]]:
    checks = []
    for evidence in (item.get("check_evidence") or [])[:3]:
        if not isinstance(evidence, dict):
            continue
        checks.append(
            {
                "kind": sanitize_identifier(evidence.get("kind") or "check"),
                "name": safe_check_name(str(evidence.get("name") or evidence.get("summary") or "")),
                "conclusion": sanitize_identifier(evidence.get("conclusion") or "failure"),
                "summary": f"Sanitized CI signal: {title}. {hint}",
            }
        )
    if not checks and category in CI_RELEVANT_CATEGORIES:
        checks.append(
            {
                "kind": "check",
                "name": safe_check_name(category),
                "conclusion": "failure",
                "summary": f"Sanitized CI signal: {title}. {hint}",
            }
        )
    return checks


def safe_check_name(value: str) -> str:
    lowered = value.lower()
    for token in (
        "pre-commit",
        "ruff",
        "semgrep",
        "detect-secrets",
        "build-docs",
        "check-json",
        "check-yaml",
        "mdformat",
        "static-sanity",
        "unit-tests",
        "integration-tests",
    ):
        if token in lowered:
            return token
    return "ci-check"


def synthetic_body(*, title: str, hint: str, category: str, path: str, replacements: CompiledReplacements) -> str:
    keywords = ", ".join(sanitize_text(keyword, replacements) for keyword in CATEGORY_KEYWORDS.get(category, ())[:8])
    keyword_sentence = f" Relevant signals include {keywords}." if keywords else ""
    return (
        f"{title}. {hint}{keyword_sentence} "
        f"Historical path kind: {path}. Use this only as reviewer memory; current PR evidence must prove the issue."
    )


def compile_replacements(replacements: dict[str, str]) -> CompiledReplacements:
    sources = sorted((source for source in replacements if source), key=len, reverse=True)
    if not sources:
        return None, {}
    pattern = re.compile("|".join(re.escape(source) for source in sources), flags=re.IGNORECASE)
    lookup = {source.lower(): replacements[source] for source in sources}
    return pattern, lookup


def sanitize_text(value: str, replacements: dict[str, str] | CompiledReplacements) -> str:
    text = " ".join(str(value or "").split())
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    if isinstance(replacements, tuple):
        replacement_pattern, lookup = replacements
    else:
        replacement_pattern, lookup = compile_replacements(replacements)
    if replacement_pattern is not None:
        text = replacement_pattern.sub(lambda match: lookup.get(match.group(0).lower(), match.group(0)), text)
    return text.strip()


def sanitize_identifier(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = text.strip("_.-")
    return text or "unknown"


def sanitize_comment_kind(value: Any) -> str:
    text = sanitize_identifier(value)
    if text in {"issue_comment", "review_comment", "review", "pull_request_review"}:
        return text
    return "review_comment"


def render_jsonl(examples: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples)


def find_public_blockers(text: str) -> list[tuple[str, str]]:
    findings = []
    for kind, pattern in PUBLIC_BLOCKLIST:
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            findings.append((kind, text[start:end].replace("\n", "\\n")))
    return findings


if __name__ == "__main__":
    raise SystemExit(main())
