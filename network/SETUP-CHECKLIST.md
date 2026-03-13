# DDN Setup Checklist

What's done (automated) vs. what needs you (manual).

## Done (no action needed)

- [x] Network manifest with 8-node topology
- [x] CLI management tool (`distributed_docs.py`)
- [x] Health monitor with auto-healing (`health_monitor.py`)
- [x] GitHub Actions workflow for automated sync
- [x] FRM-grounded theory document
- [x] Initial documentation snapshot (156 files, SHA-256 verified)
- [x] pyproject.toml entry point (`fracttalix-ddn`)
- [x] REPRODUCIBILITY.md updated with DDN section
- [x] .gitignore for runtime state and snapshot binaries
- [x] 405/405 tests still passing

## Needs You (~30 minutes total)

### 1. Create Mirror Repositories (10 min)

Create **empty** repositories named `Fracttalix` on each provider:

- [ ] **GitLab**: https://gitlab.com/projects/new → `Fracttalix`
- [ ] **Codeberg**: https://codeberg.org/repo/create → `Fracttalix`
- [ ] **Bitbucket**: https://bitbucket.org/repo/create → `Fracttalix`

Do NOT initialize with README — they must be empty to receive the push.

### 2. Add GitHub Secrets (10 min)

Go to: https://github.com/thomasbrennan/Fracttalix/settings/secrets/actions

Add these repository secrets:

| Secret | Value | How to get it |
|--------|-------|---------------|
| `GITLAB_MIRROR_URL` | `https://gitlab.com/thomasbrennan/Fracttalix.git` | Just the URL |
| `GITLAB_TOKEN` | Personal access token | GitLab → Settings → Access Tokens → `write_repository` scope |
| `CODEBERG_MIRROR_URL` | `https://codeberg.org/thomasbrennan/Fracttalix.git` | Just the URL |
| `CODEBERG_TOKEN` | Personal access token | Codeberg → Settings → Applications → Generate Token |
| `BITBUCKET_MIRROR_URL` | `https://bitbucket.org/thomasbrennan/Fracttalix.git` | Just the URL |
| `BITBUCKET_TOKEN` | App password | Bitbucket → Settings → App Passwords → `repository:write` |

### 3. Run Bootstrap (2 min)

From your local machine:

```bash
git pull origin main
python -m network.distributed_docs bootstrap
```

This configures local remotes and does the initial push to all mirrors.

### 4. Verify (1 min)

```bash
python -m network.distributed_docs status
```

Should show 4+ nodes reachable and quorum MET.

## Optional (can do later)

### IPFS Pinning

- [ ] Create account at https://www.pinata.cloud/ (free tier: 1 GB)
- [ ] Add `PINATA_JWT` to GitHub secrets
- [ ] Snapshots will auto-pin on every release

### Software Heritage

- [ ] Visit https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/thomasbrennan/Fracttalix
- [ ] Click "Save" to trigger initial archival (or the CI will do it on next release)

### Internet Archive

- [ ] Install `ia` CLI: `pip install internetarchive`
- [ ] Run the deposit command from the snapshot output
- [ ] Or let the CI handle it on release

## Once Active

The network is fully automated after setup:

- **Every push to main**: Mirrors sync to GitLab, Codeberg, Bitbucket
- **Every release**: Snapshot + IPFS pin + Software Heritage + Wayback Machine
- **Every 6 hours**: Health check verifies all nodes are reachable and in sync
- **If a node diverges**: Auto-heal pushes current state with retry
