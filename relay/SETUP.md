# Relay Setup — Autonomous Claude ↔ Grok Communication

## One-time setup (5 minutes)

### 1. Get your xAI API key

1. Go to https://console.x.ai/
2. Sign in with your Grok account
3. Create an API key
4. Copy it

### 2. Add the key to GitHub Secrets

1. Go to github.com/thomasbrennan/Fracttalix → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `XAI_API_KEY`
4. Value: paste your key
5. Click "Add secret"

### 3. Done

That's it — just the secret. The relay is now autonomous:

- **Claude** sends a message to Grok → commits to `relay/queue/` → pushes to main
- **GitHub Actions** detects the new message → triggers `grok-relay-agent.yml`
- **Grok agent** calls xAI API → gets Grok's review → commits response to `relay/queue/`
- **Claude** picks up Grok's response at next session start (or you can trigger manually)

## How it works

```
Claude writes message  →  git push  →  GitHub Actions triggers
                                              ↓
                                        grok_agent.py runs
                                              ↓
                                        Calls xAI API (Grok)
                                              ↓
                                        Grok reviews claim
                                              ↓
                                        Response committed to relay/queue/
                                              ↓
Claude reads response  ←  git pull  ←  Push to main
```

No human in the loop. You can watch the action runs at:
github.com/thomasbrennan/Fracttalix/actions

## Testing locally

```bash
export XAI_API_KEY="your-key-here"

# Dry run (no API call)
python -m relay.grok_agent --dry-run

# Process pending messages (calls API, no push)
python -m relay.grok_agent --no-push

# Full autonomous run
python -m relay.grok_agent
```

## Cost

The xAI API charges per token. A typical claim review request + response is ~2000 tokens.
At current rates, each review costs roughly $0.01-0.05. A full batch of 4 claims ≈ $0.10.

## Scheduling

The workflow checks for pending messages:
- Every time something is pushed to `relay/queue/` on main
- Every 4 hours (scheduled)
- Manually via workflow dispatch
