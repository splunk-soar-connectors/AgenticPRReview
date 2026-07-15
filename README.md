# Agentic PR Review

Agentic PR Review is a local and GitHub Actions-ready reviewer for SOAR
connector pull requests. It collects PR context, runs deterministic connector
checks, optionally asks the configured model gateway for deeper review, and
writes review artifacts. GitHub comment publishing is opt-in.

The bot supports the CIRCUIT-compatible chat-completions gateway with OAuth
client-credentials auth.

## Setup

Use the repo virtual environment or create one:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

If editable install is not available yet, run through `PYTHONPATH`:

```bash
PYTHONPATH=src .venv/bin/python -m agentic_pr_review.cli --help
```

## Configuration

GitHub App auth is required for normal review runs. The App must be installed
on the repositories it reviews and must be able to read pull requests, read
contents and checks, write pull request review comments, write issue comments,
and add labels.

```bash
export AI_REVIEW_APP_ID="<github app id>"
export AI_REVIEW_INSTALLATION_ID="<github app installation id>"
export AI_REVIEW_PRIVATE_KEY="$(cat /path/to/github-app.private-key.pem)"
```

For model access:

```bash
export MODEL_PROVIDER="circuit"
export CIRCUIT_BASE_URL="<full chat completions endpoint>"
export CIRCUIT_MODEL="<model name>"
export CIRCUIT_APP_KEY="<app key>"
export CIRCUIT_CLIENT_ID="<oauth client id>"
export CIRCUIT_CLIENT_SECRET="<oauth client secret>"
export CIRCUIT_TOKEN_URL="<oauth token endpoint>"
```

The provider exchanges `CIRCUIT_CLIENT_ID` and
`CIRCUIT_CLIENT_SECRET` at `CIRCUIT_TOKEN_URL`, caches the returned token in
memory, refreshes it before expiry, and retries once after a 401 or 403 model
response. The generic `GATEWAY_*` names are also supported for local testing,
but the reusable workflow keeps the existing `CIRCUIT_*` interface.
Runtime credential values are redacted from model prompts, errors, saved
artifacts, and published comments. The CIRCUIT app key is sent only as required
gateway request metadata in the `user` JSON string, not inside the chat prompt.

The CLI does not auto-load credential files. For local development only, you
can point `AGENTIC_PR_REVIEW_ENV_FILE` at an ignored env file.

## Run A Review

Dry run with no model call:

```bash
PYTHONPATH=src .venv/bin/python -m agentic_pr_review.cli review example-org/example-connector 1 --skip-model
```

Full local review:

```bash
PYTHONPATH=src .venv/bin/python -m agentic_pr_review.cli review example-org/example-connector 1
```

Publish one GitHub comment per publishable finding:

```bash
PYTHONPATH=src .venv/bin/python -m agentic_pr_review.cli review example-org/example-connector 1 --publish-comments
```

Review artifacts are written under this repo's `runs/` directory by default:

```text
runs/<owner>-<repo>-<pr>/
review_input.json
review_output.json
comment.md
planned_comments.json
planned_comments.md
```

`runs/` is ignored because raw review artifacts can contain PR comments, CI log
excerpts, usernames, repository names, URLs, and other private context.

## Deep Review

Deep collection is enabled by default. The bot fetches base and head contents
for changed text files, generates local diffs, splits large diffs into chunks,
reviews each chunk, and then runs a synthesis pass.

Useful controls:

```bash
PYTHONPATH=src .venv/bin/python -m agentic_pr_review.cli review example-org/example-connector 1 \
  --deep-chunk-chars 60000 \
  --deep-max-file-bytes 5000000 \
  --deep-max-file-chars 250000
```

Use `--shallow` for a legacy-style run without deep base/head collection.

## Historical Context

The bot can retrieve similar historical review examples and inject them into
the model prompt as reviewer memory. Historical examples are never evidence by
themselves; the current PR must independently prove any finding.

By default, the resolver prefers the public sanitized file:

```text
runs/training-mining-known/public-sanitized/training_examples.jsonl
```

That file is allowed by `.gitignore` and is safe to commit. Raw mining outputs
under `runs/` remain ignored and should stay private.

To regenerate the sanitized file from private mining outputs:

```bash
PYTHONPATH=src .venv/bin/python scripts/sanitize_training_examples.py
```

The sanitizer keeps every usable raw example by default, aliases repos and PRs,
derives automatic replacements from raw repo/user/path metadata, strips
secret-shaped values, and replaces raw comments/evidence with synthetic
retrieval signals.

If your raw examples contain organization-specific terms that cannot be derived
from metadata, keep replacements in an ignored file under `runs/`:

```bash
cat > runs/private_sanitizer_replacements.txt <<'EOF'
raw-private-term=public replacement
EOF

PYTHONPATH=src .venv/bin/python scripts/sanitize_training_examples.py \
  --replacement-file runs/private_sanitizer_replacements.txt
```

Each replacement line uses `RAW=REPLACEMENT`. When
`runs/private_sanitizer_replacements.txt` exists, the sanitizer loads it by
default. The file is ignored by Git.

Disable historical retrieval for an A/B test:

```bash
PYTHONPATH=src .venv/bin/python -m agentic_pr_review.cli review example-org/example-connector 1 \
  --disable-historical-context
```

## SDK Manifest Validation

Generated SDK manifest validation is available as an explicit trusted-local
step:

```bash
PYTHONPATH=src .venv/bin/python -m agentic_pr_review.cli review example-org/example-connector 1 \
  --enable-sdk-manifest
```

This fetches the PR head archive through the GitHub App/API and runs
`soarapps manifests create` in a sanitized subprocess. It is disabled by
default because manifest creation imports PR code. Set `SOAR_SDK_SOURCE_ROOT`
or `SOARAPPS_BIN` if your local SDK tooling is not already on `PATH`.

## Reusable Workflow

Target repositories can call the reusable workflow in this repository:

```yaml
jobs:
  review:
    uses: <owner>/AgenticPRReview/.github/workflows/pr-review.yml@main
    with:
      repo: ${{ github.repository }}
      pr_number: ${{ github.event.pull_request.number || inputs.pr_number }}
      circuit_base_url: ${{ vars.CIRCUIT_BASE_URL }}
      circuit_model: ${{ vars.CIRCUIT_MODEL }}
      circuit_token_url: ${{ vars.CIRCUIT_TOKEN_URL }}
    secrets:
      AI_REVIEW_APP_ID: ${{ secrets.AI_REVIEW_APP_ID }}
      AI_REVIEW_INSTALLATION_ID: ${{ secrets.AI_REVIEW_INSTALLATION_ID }}
      AI_REVIEW_PRIVATE_KEY: ${{ secrets.AI_REVIEW_PRIVATE_KEY }}
      CIRCUIT_CLIENT_ID: ${{ secrets.CIRCUIT_CLIENT_ID }}
      CIRCUIT_CLIENT_SECRET: ${{ secrets.CIRCUIT_CLIENT_SECRET }}
      CIRCUIT_APP_KEY: ${{ secrets.CIRCUIT_APP_KEY }}
```

The workflow intentionally does not pass `--enable-sdk-manifest`, so it does
not execute untrusted PR code from pull request workflows.

For public repositories, keep org-level secrets scoped to selected repositories
and run this through `pull_request_target` only when the workflow never checks
out or executes the contributor branch. The caller passes explicit secrets
rather than `secrets: inherit`; the reusable workflow validates that the caller
and target repository belong to `splunk-soar-connectors`, grants `GITHUB_TOKEN`
permissions at the job level, pins third-party actions by commit SHA, uses a
30-minute timeout, and masks the just-in-time gateway token in GitHub Actions
logs. Keep SDK manifest generation and any PR-code execution disabled in public
PR workflows.

Secret safety checks built into the bot:

- Runtime secret values from GitHub App and CIRCUIT env vars are redacted before
  model prompts, local `runs/` artifacts, stderr failure messages, and GitHub
  comments are written.
- Generated CIRCUIT access tokens are masked with GitHub Actions `add-mask`.
- Common token literals found in PR diffs or CI logs, such as GitHub tokens,
  bearer/basic auth headers, OpenAI-style keys, AWS access keys, JWTs, and
  private-key blocks, are redacted before publication or artifact upload.
- The reusable workflow does not enable SDK manifest generation because that
  path can import PR code.

## What It Reviews

The reviewer is tuned for connector PR issues such as:

- OAuth and API auth correctness, including token flow, timeout, retry, and
  test-connectivity behavior.
- Polling checkpoint and deduplication safety.
- `add_data()` / app JSON / SDK output mismatches.
- SDK migration compatibility, including action params, output contracts,
  summaries, custom views, webhooks, generated manifest drift, and legacy
  manifest parity.
- Unsafe logging of tokens, headers, payloads, responses, tenant data, or PII.
- Metadata drift, including mutating actions marked `read_only`.
- Pagination and truncation risks.
- Missing validation and missing tests for risky behavior changes.
- CI failure synthesis from check metadata and failed-job log excerpts.

## Public Repo Notes

- Keep raw `runs/` artifacts ignored.
- Commit only the sanitized historical examples file under
  `runs/training-mining-known/public-sanitized/`.
- Do not commit local env files, GitHub App private keys, gateway credentials,
  or generated package metadata.
- If this repository already existed as a private remote with sensitive files
  in old commits, rewrite or replace that remote history before making it
  public.
