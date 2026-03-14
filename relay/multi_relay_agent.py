#!/usr/bin/env python3
"""Multi-AI Relay Agent — unified message processing across AI providers.

Extends the Grok relay pattern to support multiple AI systems:
Gemini (Google), ChatGPT (OpenAI), Mistral, DeepSeek, Qwen (Alibaba),
Yi (01.AI), ERNIE (Baidu), and Llama (via API).

Each provider uses the same DRS-MP v2 message format and the same
queue/archive lifecycle. Provider-specific details (API URLs, auth
headers, model names) are configured in PROVIDERS below.

Usage:
    python -m relay.multi_relay_agent --provider gemini
    python -m relay.multi_relay_agent --provider chatgpt
    python -m relay.multi_relay_agent --provider mistral
    python -m relay.multi_relay_agent --provider deepseek
    python -m relay.multi_relay_agent --provider all  # Process all providers

Requires:
    Provider-specific API key environment variables (see PROVIDERS dict).
"""
from __future__ import annotations

import json
import os
import random
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELAY_DIR = REPO_ROOT / "relay"
QUEUE_DIR = RELAY_DIR / "queue"

PROVIDERS = {
    "grok": {
        "api_url": "https://api.x.ai/v1/chat/completions",
        "env_key": "XAI_API_KEY",
        "model": "grok-4-latest",
        "model_fast": "grok-4-fast",
        "agent_id": "grok",
        "provider_name": "xAI",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",  # OpenAI-compatible chat completions
    },
    "gemini": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "env_key": "GOOGLE_AI_API_KEY",
        "model": "gemini-2.5-pro",
        "model_fast": "gemini-2.5-flash",
        "agent_id": "gemini",
        "provider_name": "Google",
        "auth_header": "x-goog-api-key: {key}",
        "request_format": "gemini",
    },
    "chatgpt": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o",
        "model_fast": "gpt-4o-mini",
        "agent_id": "chatgpt",
        "provider_name": "OpenAI",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",
    },
    "mistral": {
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "env_key": "MISTRAL_API_KEY",
        "model": "mistral-large-latest",
        "model_fast": "mistral-small-latest",
        "agent_id": "mistral",
        "provider_name": "Mistral AI",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",
    },
    "deepseek": {
        "api_url": "https://api.deepseek.com/chat/completions",
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "model_fast": "deepseek-chat",
        "agent_id": "deepseek",
        "provider_name": "DeepSeek",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",
    },
    # --- Overseas AI Systems ---
    "qwen": {
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "env_key": "DASHSCOPE_API_KEY",
        "model": "qwen-max",
        "model_fast": "qwen-turbo",
        "agent_id": "qwen",
        "provider_name": "Alibaba Cloud (Qwen)",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",
    },
    "yi": {
        "api_url": "https://api.lingyiwanwu.com/v1/chat/completions",
        "env_key": "YI_API_KEY",
        "model": "yi-large",
        "model_fast": "yi-medium",
        "agent_id": "yi",
        "provider_name": "01.AI (Yi)",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",
    },
    "ernie": {
        "api_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-8k",
        "env_key": "BAIDU_API_KEY",
        "model": "ernie-4.0-8k",
        "model_fast": "ernie-speed-128k",
        "agent_id": "ernie",
        "provider_name": "Baidu (ERNIE)",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",
    },
    "llama": {
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "env_key": "TOGETHER_API_KEY",
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "model_fast": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "agent_id": "llama",
        "provider_name": "Meta (Llama via Together)",
        "auth_header": "Authorization: Bearer {key}",
        "request_format": "openai",
    },
}


def generate_message_id() -> str:
    now = datetime.now(timezone.utc)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"MSG-{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def get_api_key(provider: dict) -> str:
    key = os.environ.get(provider["env_key"], "")
    if not key:
        print(f"ERROR: {provider['env_key']} not set.", file=sys.stderr)
        sys.exit(1)
    return key


def get_pending_messages(target_agent: str) -> list[tuple[Path, dict]]:
    """Get pending messages addressed to a specific agent or 'all'."""
    pending = []
    for path in sorted(QUEUE_DIR.glob("MSG-*.json")):
        with open(path) as f:
            msg = json.load(f)
        if msg.get("to") in (target_agent, "all") and msg.get("status") == "pending":
            pending.append((path, msg))
    return pending


def build_system_prompt(provider: dict) -> str:
    """Build the system prompt for any provider."""
    return f"""You are {provider['agent_id']}, acting as an independent reviewer for the Fracttalix research project.

PROJECT CONTEXT:
Fracttalix is a unified scientific corpus on the Fractal Rhythm Model (FRM) with 175+ machine-verifiable claims across 8 Meta-Kaizen papers, 14+ science papers, and DRS architecture papers. The corpus uses the Dual Reader Standard (DRS) where every paper has both a human-readable prose version and a machine-readable AI layer (JSON) with typed claims and deterministic falsification predicates.

YOUR ROLE:
You are an independent adversarial reviewer. Your objective is to FALSIFY claims, not confirm them. You succeed when you find a genuine defect.

RESPONSE FORMAT (DRS-MP v2):
You MUST respond with a JSON object. If reviewing claims, include:

"verdicts": [
  {{
    "claim_id": "the claim ID",
    "verdict": "confirmed" | "disputed" | "inconclusive" | "needs-revision",
    "confidence": 0.0 to 1.0,
    "reasoning": "your detailed analysis",
    "predicate_assessment": {{
      "c6_vacuity": "pass" | "fail" | "uncertain",
      "deterministic": "pass" | "fail" | "uncertain",
      "variables_bound": "pass" | "fail" | "uncertain",
      "third_party_executable": "pass" | "fail" | "uncertain"
    }},
    "sources_checked": ["list of sources"]
  }}
]

If raising objections, include:

"objections": [
  {{
    "objection_id": "OBJ-XX-NN",
    "targets_claim": "claim ID",
    "objection_type": "logical-gap" | "counterexample" | "unstated-assumption" | "vacuity" | "circularity" | "scope-overreach" | "empirical-gap" | "prior-art-overlap" | "definition-weakness",
    "statement": "the specific objection",
    "proposed_test": "how to test this objection",
    "severity": "critical" | "major" | "minor"
  }}
]

For non-review messages, respond with at minimum "response" and "status" fields.

Output ONLY the JSON. No markdown fences, no preamble."""


def build_openai_payload(system_prompt: str, user_message: str, model: str) -> str:
    """Build payload for OpenAI-compatible APIs (Grok, ChatGPT, Mistral, DeepSeek)."""
    return json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "stream": False,
    })


def build_gemini_payload(system_prompt: str, user_message: str) -> str:
    """Build payload for Gemini API."""
    return json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.3},
    })


def build_user_message(msg: dict) -> str:
    """Build user message from a relay message, preferring structured content."""
    parts = [
        f"Subject: {msg['subject']}",
        f"From: {msg['from']}",
        f"Type: {msg['type']}",
        f"Priority: {msg.get('priority', 'normal')}",
        "",
    ]

    # If message has structured claims (DRS-MP v2), include them
    if msg.get("claims"):
        parts.append("## STRUCTURED CLAIMS (DRS-MP v2 — machine authoritative)")
        parts.append(json.dumps(msg["claims"], indent=2))
        parts.append("")

    # Always include prose body as context
    parts.append("## PROSE CONTEXT (Channel 1)")
    parts.append(msg["body"])

    # Include attack vectors if present
    if msg.get("metadata", {}).get("attack_vectors"):
        parts.append("\n## SUGGESTED ATTACK VECTORS")
        for av in msg["metadata"]["attack_vectors"]:
            parts.append(f"- {av}")

    return "\n".join(parts)


def call_api(provider: dict, system_prompt: str, user_message: str, api_key: str, model: str | None = None) -> tuple[str, dict]:
    """Call any provider's API via curl. Returns (response_text, usage_dict)."""
    import time

    selected_model = model or provider["model"]
    fmt = provider["request_format"]

    if fmt == "openai":
        payload = build_openai_payload(system_prompt, user_message, selected_model)
        url = provider["api_url"]
    elif fmt == "gemini":
        payload = build_gemini_payload(system_prompt, user_message)
        url = provider["api_url"].format(model=selected_model)
    else:
        raise ValueError(f"Unknown request format: {fmt}")

    auth = provider["auth_header"].format(key=api_key)

    max_retries = 3
    for attempt in range(max_retries + 1):
        cmd = [
            "curl", "-s", "-w", "\n%{http_code}",
            url,
            "-H", "Content-Type: application/json",
            "-H", auth,
            "-d", payload,
            "--max-time", "120",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
        output = result.stdout.strip()
        lines = output.rsplit("\n", 1)
        body = lines[0] if len(lines) > 1 else output
        http_code = int(lines[-1]) if len(lines) > 1 and lines[-1].isdigit() else 0

        if http_code == 200:
            data = json.loads(body)

            # Extract response text based on format
            if fmt == "openai":
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
            elif fmt == "gemini":
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                usage = data.get("usageMetadata", {})
            else:
                text = json.dumps(data)
                usage = {}

            return text, usage

        print(f"API error {http_code} ({provider['provider_name']}): {body[:200]}", file=sys.stderr)

        if attempt < max_retries and http_code in (429, 500, 502, 503):
            wait = 2 ** (attempt + 1)
            print(f"  Retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            continue

        raise RuntimeError(f"{provider['provider_name']} API failed with HTTP {http_code}")


def parse_response(raw: str, original_msg: dict, provider: dict) -> dict:
    """Parse a provider response into a DRS-MP v2 relay message."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        review = json.loads(text)
    except json.JSONDecodeError:
        review = {
            "response": text,
            "status": "parse-error",
            "note": f"{provider['provider_name']}'s response was not valid JSON; raw text preserved.",
        }

    # Build DRS-MP v2 response
    response_msg = {
        "msg_id": generate_message_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from": provider["agent_id"],
        "to": original_msg["from"],
        "type": f"{original_msg['type']}-response" if not original_msg["type"].endswith("-response") else original_msg["type"],
        "priority": original_msg.get("priority", "normal"),
        "subject": f"Re: {original_msg['subject']}",
        "body": review.get("response", json.dumps(review, indent=2)),
        "protocol_version": "2.0.0",
        "references": {
            "parent_message": original_msg.get("msg_id", original_msg.get("id", "unknown")),
        },
        "status": "pending",
    }

    # Carry structured fields from DRS-MP v2 responses
    if "verdicts" in review:
        response_msg["verdicts"] = review["verdicts"]
    if "objections" in review:
        response_msg["objections"] = review["objections"]

    # Store raw review for debugging
    response_msg["_raw_review"] = review
    response_msg["_provider"] = provider["provider_name"]
    response_msg["_model_used"] = provider["model"]

    # Carry over claim IDs
    if original_msg.get("references", {}).get("claim_ids"):
        response_msg["references"]["claim_ids"] = original_msg["references"]["claim_ids"]
    if original_msg.get("claims"):
        response_msg["references"]["claim_ids"] = [c["claim_id"] for c in original_msg["claims"]]

    return response_msg


def save_response(msg: dict) -> Path:
    path = QUEUE_DIR / f"{msg['msg_id']}.json"
    with open(path, "w") as f:
        json.dump(msg, f, indent=2)
    return path


def mark_resolved(path: Path) -> None:
    with open(path) as f:
        msg = json.load(f)
    msg["status"] = "resolved"
    with open(path, "w") as f:
        json.dump(msg, f, indent=2)


def process_message(path: Path, msg: dict, provider: dict, api_key: str, system_prompt: str, *, dry_run: bool = False) -> dict | None:
    msg_id = msg.get("msg_id", msg.get("id", "unknown"))
    print(f"\nProcessing: {msg_id}")
    print(f"  Provider: {provider['provider_name']}")
    print(f"  Type: {msg['type']}")
    print(f"  Subject: {msg['subject']}")

    user_message = build_user_message(msg)

    if dry_run:
        print(f"  [DRY RUN] Would send to {provider['provider_name']} API")
        print(f"  Model: {provider['model']}")
        print(f"  Message length: {len(user_message)} chars")
        return None

    print(f"  Sending to {provider['provider_name']} ({provider['model']})...")
    raw_response, usage = call_api(provider, system_prompt, user_message, api_key)
    print(f"  Response received ({len(raw_response)} chars)")

    response_msg = parse_response(raw_response, msg, provider)
    if usage:
        response_msg["_usage"] = usage
    save_response(response_msg)
    mark_resolved(path)

    print(f"  Response saved: {response_msg['msg_id']}")
    if response_msg.get("verdicts"):
        for v in response_msg["verdicts"]:
            print(f"    {v['claim_id']}: {v['verdict']} ({v.get('confidence', '?')})")

    return response_msg


def git_commit_and_push(files: list[str], message: str) -> bool:
    try:
        for f in files:
            subprocess.run(["git", "add", f], check=True, cwd=REPO_ROOT)
        subprocess.run(["git", "commit", "-m", message], check=True, cwd=REPO_ROOT)
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}", file=sys.stderr)
        return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-AI Relay Agent")
    parser.add_argument("--provider", required=True, choices=list(PROVIDERS.keys()) + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    providers_to_run = list(PROVIDERS.values()) if args.provider == "all" else [PROVIDERS[args.provider]]

    total_processed = 0
    all_changed_files = []

    for provider in providers_to_run:
        api_key = "" if args.dry_run else get_api_key(provider)
        system_prompt = build_system_prompt(provider)
        pending = get_pending_messages(provider["agent_id"])

        if not pending:
            print(f"No pending messages for {provider['provider_name']}.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {len(pending)} message(s) for {provider['provider_name']}")
        print(f"{'='*60}")

        for i, (path, msg) in enumerate(pending):
            if i >= args.batch_size:
                print(f"Batch limit reached ({args.batch_size}).")
                break
            result = process_message(path, msg, provider, api_key, system_prompt, dry_run=args.dry_run)
            if result:
                all_changed_files.append(str(path))
                all_changed_files.append(str(QUEUE_DIR / f"{result['msg_id']}.json"))
                total_processed += 1

    if not args.dry_run and all_changed_files and not args.no_push:
        commit_msg = f"relay(multi-agent): processed {total_processed} message(s) across providers"
        git_commit_and_push(all_changed_files, commit_msg)

    print(f"\nDone. Processed {total_processed} message(s) total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
