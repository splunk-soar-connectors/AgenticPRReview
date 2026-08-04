"""Minimal GitHub REST client using the Python standard library."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from .github_auth import TokenProvider
from .secret_redactor import redact_text


class GitHubError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        token_provider: TokenProvider | None = None,
        api_url: str = "https://api.github.com",
        auth_mode: str = "unknown",
    ) -> None:
        self.token = token
        self.token_provider = token_provider
        self.api_url = api_url.rstrip("/")
        self.auth_mode = auth_mode

    def get(self, path: str, *, params: dict[str, Any] | None = None, accept: str | None = None) -> Any:
        if params:
            path = f"{path}?{urlencode(params)}"
        url = path if path.startswith("http") else f"{self.api_url}{path}"
        headers = self._headers(accept=accept)
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            body = redact_text(exc.read().decode("utf-8", errors="replace"))
            raise GitHubError(f"GitHub API error {exc.code} for {redact_text(url)}: {body}", status_code=exc.code) from exc

    def get_raw(self, path: str, *, params: dict[str, Any] | None = None, accept: str | None = None) -> bytes:
        if params:
            path = f"{path}?{urlencode(params)}"
        url = path if path.startswith("http") else f"{self.api_url}{path}"
        headers = self._headers(accept=accept)
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            body = redact_text(exc.read().decode("utf-8", errors="replace"))
            raise GitHubError(f"GitHub API error {exc.code} for {redact_text(url)}: {body}", status_code=exc.code) from exc

    def post(self, path: str, payload: dict[str, Any], *, accept: str | None = None) -> Any:
        if not self._auth_token():
            raise GitHubError("GitHub token is required for write operations.")
        url = path if path.startswith("http") else f"{self.api_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = self._headers(accept=accept)
        headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            body = redact_text(exc.read().decode("utf-8", errors="replace"))
            raise GitHubError(f"GitHub API error {exc.code} for {redact_text(url)}: {body}", status_code=exc.code) from exc

    def _headers(self, *, accept: str | None = None) -> dict[str, str]:
        headers = {
            "Accept": accept or "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "agentic-pr-review-prototype",
        }
        token = self._auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _auth_token(self) -> str | None:
        if self.token_provider:
            return self.token_provider.get_token()
        return self.token

    def get_text_url(self, url: str) -> str | None:
        headers = {
            "Accept": "text/plain",
            "User-Agent": "agentic-pr-review-prototype",
        }
        token = self._auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
        except HTTPError as exc:
            body = redact_text(exc.read().decode("utf-8", errors="replace"))
            raise GitHubError(f"GitHub raw file error {exc.code} for {redact_text(url)}: {body}", status_code=exc.code) from exc
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def paginate(self, path: str, *, params: dict[str, Any] | None = None, limit_pages: int = 10) -> list[Any]:
        output: list[Any] = []
        page = 1
        while page <= limit_pages:
            page_params = dict(params or {})
            page_params.update({"per_page": 100, "page": page})
            chunk = self.get(path, params=page_params)
            if not chunk:
                break
            if not isinstance(chunk, list):
                raise GitHubError(f"Expected list from paginated endpoint {path}")
            output.extend(chunk)
            if len(chunk) < 100:
                break
            page += 1
        return output

    def get_pr(self, repo: str, number: int) -> dict[str, Any]:
        return self.get(f"/repos/{repo}/pulls/{number}")

    def list_pr_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/pulls/{number}/files", limit_pages=20)

    def list_issue_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/issues/{number}/comments", limit_pages=10)

    def list_review_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/pulls/{number}/comments", limit_pages=20)

    def list_reviews(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/pulls/{number}/reviews", limit_pages=10)

    def list_root_contents(self, repo: str, ref: str) -> list[dict[str, Any]]:
        data = self.get(f"/repos/{repo}/contents", params={"ref": ref})
        return data if isinstance(data, list) else []

    def fetch_text_file(self, repo: str, path: str, ref: str, *, max_bytes: int = 450_000) -> str | None:
        encoded_path = quote(path, safe="/")
        data = self.get(f"/repos/{repo}/contents/{encoded_path}", params={"ref": ref})
        if not isinstance(data, dict):
            return None
        if data.get("type") != "file":
            return None
        if int(data.get("size") or 0) > max_bytes:
            return None
        content = data.get("content")
        encoding = data.get("encoding")
        if not content or encoding != "base64":
            return None
        try:
            return base64.b64decode(content).decode("utf-8")
        except UnicodeDecodeError:
            return None

    def fetch_text_file_deep(self, repo: str, path: str, ref: str, *, max_bytes: int = 5_000_000) -> str | None:
        """Fetch a text file for deep review, falling back to the raw media type."""

        encoded_path = quote(path, safe="/")
        data = self.get(f"/repos/{repo}/contents/{encoded_path}", params={"ref": ref})
        if not isinstance(data, dict) or data.get("type") != "file":
            return None
        if int(data.get("size") or 0) > max_bytes:
            return None

        content = data.get("content")
        encoding = data.get("encoding")
        if content and encoding == "base64":
            try:
                return base64.b64decode(content).decode("utf-8")
            except UnicodeDecodeError:
                return None

        try:
            raw = self.get_raw(
                f"/repos/{repo}/contents/{encoded_path}",
                params={"ref": ref},
                accept="application/vnd.github.raw",
            )
            return raw.decode("utf-8")
        except (GitHubError, UnicodeDecodeError):
            download_url = data.get("download_url")
            if not download_url:
                return None
            return self.get_text_url(str(download_url))

    def get_check_runs(self, repo: str, ref: str) -> dict[str, Any]:
        return self.get(
            f"/repos/{repo}/commits/{ref}/check-runs",
            params={"per_page": 100},
            accept="application/vnd.github+json",
        )

    def get_combined_status(self, repo: str, ref: str) -> dict[str, Any]:
        return self.get(f"/repos/{repo}/commits/{ref}/status")

    def get_actions_job_logs(self, repo: str, job_id: str | int) -> bytes:
        return self.get_redirected_raw_without_auth(f"/repos/{repo}/actions/jobs/{job_id}/logs")

    def list_workflow_run_jobs(
        self,
        repo: str,
        run_id: str | int,
        *,
        attempt: str | int | None = None,
        limit_pages: int = 10,
    ) -> list[dict[str, Any]]:
        if attempt:
            path = f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
        else:
            path = f"/repos/{repo}/actions/runs/{run_id}/jobs"
        jobs: list[dict[str, Any]] = []
        page = 1
        while page <= limit_pages:
            data = self.get(path, params={"per_page": 100, "page": page})
            if not isinstance(data, dict):
                raise GitHubError(f"Expected object from workflow jobs endpoint {path}")
            chunk = data.get("jobs") or []
            if not isinstance(chunk, list):
                raise GitHubError(f"Expected jobs list from workflow jobs endpoint {path}")
            jobs.extend(job for job in chunk if isinstance(job, dict))
            if len(chunk) < 100:
                break
            page += 1
        return jobs

    def get_repository_zipball(self, repo: str, ref: str) -> bytes:
        return self.get_redirected_raw_without_auth(f"/repos/{repo}/zipball/{quote(ref, safe='')}")

    def get_installation_repositories(self) -> dict[str, Any]:
        return self.get("/installation/repositories")

    def create_pull_request_line_comment(
        self,
        repo: str,
        number: int,
        *,
        body: str,
        commit_id: str,
        path: str,
        line: int,
        side: str = "RIGHT",
    ) -> dict[str, Any]:
        return self.post(
            f"/repos/{repo}/pulls/{number}/comments",
            {
                "body": body,
                "commit_id": commit_id,
                "path": path,
                "line": line,
                "side": side,
            },
        )

    def create_issue_comment(self, repo: str, number: int, *, body: str) -> dict[str, Any]:
        return self.post(f"/repos/{repo}/issues/{number}/comments", {"body": body})

    def add_issue_labels(self, repo: str, number: int, labels: list[str]) -> dict[str, Any]:
        return self.post(f"/repos/{repo}/issues/{number}/labels", {"labels": labels})

    def get_redirected_raw_without_auth(self, path: str) -> bytes:
        """Fetch a GitHub API endpoint that redirects to a signed asset URL.

        Actions log endpoints return a 302 to a short-lived storage URL. Sending
        the GitHub Authorization header to that signed URL can make the storage
        service reject the request, so we intentionally follow the redirect with
        only generic headers.
        """

        url = path if path.startswith("http") else f"{self.api_url}{path}"
        opener = build_opener(NoRedirectHandler)
        request = Request(url, headers=self._headers())
        try:
            with opener.open(request, timeout=45) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                body = redact_text(exc.read().decode("utf-8", errors="replace"))
                raise GitHubError(f"GitHub API error {exc.code} for {redact_text(url)}: {body}", status_code=exc.code) from exc
            location = exc.headers.get("Location")
            if not location:
                raise GitHubError(f"GitHub API redirect for {redact_text(url)} did not include a Location header") from exc

        redirected = Request(
            location,
            headers={
                "Accept": "text/plain, application/zip, */*",
                "User-Agent": "agentic-pr-review-prototype",
            },
        )
        try:
            with urlopen(redirected, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            body = redact_text(exc.read().decode("utf-8", errors="replace"))
            raise GitHubError(f"GitHub redirected asset error {exc.code} for {redact_text(url)}: {body}", status_code=exc.code) from exc


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None
