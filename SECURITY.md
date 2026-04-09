# Security Policy

## Secret Handling

All credentials are loaded exclusively from environment variables.
**Never hard-code API keys, tokens, or secrets in source code or commit them to git.**

### Required secrets

| Variable | Service | Where to set |
|----------|---------|--------------|
| `ALPACA_API_KEY` | Alpaca trading | `.env` locally, GitHub Actions Secrets in CI |
| `ALPACA_SECRET_KEY` | Alpaca trading | `.env` locally, GitHub Actions Secrets in CI |
| `POLYGON_API_KEY` | Polygon market data | `.env` locally, GitHub Actions Secrets in CI |
| `FIREBASE_SERVICE_ACCOUNT` | Firebase RTDB | GitHub Actions Secrets only |
| `KIMI_API_KEY` | Kimi LLM | `.env` locally, GitHub Actions Secrets in CI |

### Safe fallback behaviour

When a secret is missing:
- The system logs a `WARNING` and disables the affected integration.
- It does **not** crash or print the secret value.
- `run_engine_live.py` validates all required secrets at startup and exits cleanly with a descriptive error if any are absent.

---

## API Key Safety

1. **Never log full API keys.** Only the first 10 characters may appear in DEBUG-level logs (e.g. `ABCDEFGHIJ...`).
2. **Rotate keys immediately** if they appear in git history. Use `git filter-repo` or BFG Repo Cleaner.
3. **Paper trading by default.** `ALPACA_PAPER=true` and `PAPER_TRADING=true` are the defaults in `config.py`. Live trading requires explicit opt-in.

---

## Logging Safety

- All application logging uses Python's `logging` module (not `print`).
- Log level is controlled by the `LOG_LEVEL` environment variable (default: `INFO`).
- `DEBUG` level may include API endpoint URLs and key previews — do not enable DEBUG in production.
- Log files are written to `LOG_DIR` (default: `logs/`) which is `.gitignore`d.

---

## Dependency Pinning

- All dependencies are pinned in `requirements.txt`.
- Update dependencies deliberately, not automatically.
- Run `pip-audit` or `safety check` before updating any dependency.
- Do not add new dependencies without reviewing their transitive dependency tree.

---

## CI/CD Secret Handling

- GitHub Actions workflows skip gracefully when secrets are not provisioned (using `if: env.SECRET != ''` guards).
- The `nightly-retrain.yml` workflow has a `check_secrets` pre-flight step that sets `skip=true` and exits cleanly when required secrets are absent.
- Secrets are never echoed in workflow step output.

---

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please open a private security advisory via GitHub's **Security** tab rather than a public issue.
