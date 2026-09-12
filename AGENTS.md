# Project instructions

- Preserve unrelated edits. Do not commit secrets, private keys, result files, or credentials.
- Search algorithm changes follow README.md: develop in `tron_vanity_experimental` before promoting to `tron_vanity`. Deployment-only changes do not require copying either package.
- Deployment tests must not create paid cloud resources. Real GPU acceptance requires explicit authority and must be reported separately from mocked tests.
- Cloud cleanup must use explicit, verified resource names. Never delete project-wide resources or wallet results without user authorization.

## Agent skills

### Issue tracker

Work is tracked in GitHub Issues for `xu666xu888-ai/tron`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context documentation. See `docs/agents/domain.md`.
