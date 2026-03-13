# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 12.x    | Yes                |
| < 12.0  | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in Fracttalix, please report it
responsibly:

1. **Do not** open a public GitHub issue.
2. Email the maintainer at the address listed in `pyproject.toml`, or use
   [GitHub's private vulnerability reporting](https://github.com/thomasbrennan/Fracttalix/security/advisories/new).
3. Include a description of the vulnerability, steps to reproduce, and any
   relevant environment details.

You can expect an initial response within 7 days. Critical issues will be
patched and released as soon as possible.

## Scope

Fracttalix has zero required runtime dependencies and is pure Python (stdlib
only in core). The primary attack surface is:

- Malicious input to the streaming detector API
- The optional REST server (`fracttalix.extras.server`)
- Dependencies in optional extras (`fast`, `full`)

The optional REST server is intended for local/development use and should not
be exposed to untrusted networks without appropriate access controls.

## Repository Access Control Policy

### Principles

1. **Least privilege** — Contributors receive the minimum access necessary for
   their role. No standing write access is granted without justification.
2. **Mandatory review** — All changes to `main` must go through a pull request
   and receive approval from a designated code owner before merging.
3. **Record immutability** — Fundamental records (AI layers, papers, journal
   entries, legal documents, citations) are treated as append-only archives.
   Modifications or deletions require explicit owner approval and
   justification.
4. **Pipeline integrity** — CI/CD workflows, GitHub Actions, and governance
   files (CODEOWNERS, SECURITY.md) are protected paths. Changes are
   automatically flagged and require owner review.

### Protected Asset Tiers

| Tier | Assets | Protection |
|------|--------|------------|
| **1 — Fundamental Records** | `ai-layers/`, `paper/`, `journal/`, `legal/`, `CITATION.cff`, `LICENSE`, `REPRODUCIBILITY.md` | Owner-only approval, deletion blocked by CI |
| **2 — Core Library** | `fracttalix/`, `benchmark/`, `tests/`, `pyproject.toml`, `CHANGELOG.md` | Owner approval, full test suite must pass |
| **3 — Governance & CI** | `.github/`, `SECURITY.md`, `CODEOWNERS`, `scripts/` | Owner approval, workflow safety checks |

### Enforcement Mechanisms

- **CODEOWNERS**: Requires designated reviewer approval for all protected paths
- **Security Gate workflow**: Automated CI checks on every PR that classify
  changes by tier, scan for leaked secrets, block deletion of critical files,
  and detect dangerous workflow patterns
- **Integrity Audit workflow**: Weekly scheduled audit that verifies critical
  files exist, validates AI layer schemas, checks corpus consistency, and
  confirms CODEOWNERS syntax
- **Dependabot**: Automated dependency updates for GitHub Actions and pip
  packages to address known vulnerabilities

### Recommended GitHub Settings

The following settings should be enabled in the repository's GitHub settings
(Settings > Branches > Branch protection rules for `main`):

- [x] Require pull request reviews before merging (1 approval minimum)
- [x] Dismiss stale pull request approvals when new commits are pushed
- [x] Require review from code owners
- [x] Require status checks to pass before merging
  - `Tests` (all matrix variants)
  - `Lint`
  - `Security Gate / Classify changed files by security tier`
  - `Security Gate / Scan for leaked secrets`
  - `Security Gate / Guard against deletion of critical files`
  - `Security Gate / Verify workflow file safety`
- [x] Require branches to be up to date before merging
- [x] Require signed commits (when contributors have GPG/SSH signing set up)
- [x] Do not allow bypassing the above settings (including administrators)
- [x] Restrict who can push to matching branches (owner only)
- [x] Do not allow force pushes
- [x] Do not allow deletions

### Contributor Access Levels

| Role | Permissions | When to use |
|------|------------|-------------|
| **Owner** (`@thomasbrennan`) | Full admin access, sole merge authority for all tiers | Repository governance |
| **Collaborator** | Write access to feature branches only | Active development contributors |
| **External contributor** | Fork + PR (no direct push) | Open-source contributions |
| **Read-only** | Read access | Observers, auditors |

### Incident Response

If unauthorized changes are detected:

1. Immediately revert the unauthorized commits
2. Rotate any potentially exposed credentials
3. Review the audit log (Settings > Audit log) for the actor's actions
4. Revoke the actor's access
5. Document the incident and notify affected parties
