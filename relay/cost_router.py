"""Cost-aware model routing for the Grok relay pipeline.

Routes messages to grok-4-latest (high quality) or grok-4-fast (budget)
based on message type, priority, and budget tracking.

Pricing (as of 2026-03):
    grok-4-latest:  $3.00/1M input,  $15.00/1M output
    grok-4-fast:    $0.20/1M input,   $0.50/1M output
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

# Model configurations
MODELS = {
    "quality": {
        "id": "grok-4-latest",
        "input_cost_per_m": 3.00,
        "output_cost_per_m": 15.00,
        "description": "Flagship model — adversarial reviews, complex derivations",
    },
    "fast": {
        "id": "grok-4-fast",
        "input_cost_per_m": 0.20,
        "output_cost_per_m": 0.50,
        "description": "Budget model — routine checks, formatting, sanity checks",
    },
}

# Routing rules: message type/priority → model tier
ROUTING_RULES = {
    # High-value tasks → quality model
    "claim-review": "quality",
    "cross-reference": "quality",
    # Routine tasks → fast model
    "qc-request": "fast",
    "status-query": "fast",
    "general": "fast",
}

# Priority overrides: critical/high always get quality model
PRIORITY_OVERRIDE = {"critical", "high"}

BUDGET_FILE = Path(__file__).resolve().parent / "budget-tracker.json"


def select_model(msg_type: str, priority: str = "normal", force_tier: str | None = None) -> dict:
    """Select the appropriate model based on message type and priority.

    Returns the model config dict with 'id', costs, etc.
    """
    if force_tier and force_tier in MODELS:
        return MODELS[force_tier]

    if priority in PRIORITY_OVERRIDE:
        return MODELS["quality"]

    tier = ROUTING_RULES.get(msg_type, "fast")
    return MODELS[tier]


def estimate_cost(input_tokens: int, output_tokens: int, model_tier: str = "quality") -> float:
    """Estimate cost in USD for a given token count."""
    model = MODELS[model_tier]
    input_cost = (input_tokens / 1_000_000) * model["input_cost_per_m"]
    output_cost = (output_tokens / 1_000_000) * model["output_cost_per_m"]
    return input_cost + output_cost


def estimate_message_cost(system_chars: int, user_chars: int, model_tier: str = "quality") -> float:
    """Rough cost estimate from character counts (4 chars ≈ 1 token)."""
    input_tokens = (system_chars + user_chars) // 4
    output_tokens = 800  # average response
    return estimate_cost(input_tokens, output_tokens, model_tier)


def load_budget() -> dict:
    """Load budget tracking data."""
    if BUDGET_FILE.exists():
        with open(BUDGET_FILE) as f:
            return json.load(f)
    return {
        "total_budget_usd": 25.00,
        "spent_usd": 0.0,
        "transactions": [],
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename to prevent corruption."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def record_transaction(msg_id: str, model_id: str, input_tokens: int, output_tokens: int, cost_usd: float) -> dict:
    """Record a completed API transaction for budget tracking."""
    budget = load_budget()
    budget["spent_usd"] = round(budget["spent_usd"] + cost_usd, 6)
    budget["transactions"].append({
        "msg_id": msg_id,
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
    })
    # Validate consistency: spent should equal sum of transactions
    tx_total = round(sum(t["cost_usd"] for t in budget["transactions"]), 6)
    if abs(budget["spent_usd"] - tx_total) > 0.001:
        budget["spent_usd"] = tx_total  # self-heal from drift
    _atomic_write_json(BUDGET_FILE, budget)
    return budget


def get_remaining_budget() -> float:
    """Get remaining budget in USD."""
    budget = load_budget()
    return budget["total_budget_usd"] - budget["spent_usd"]


def check_budget(estimated_cost: float) -> bool:
    """Check if we have enough budget for an estimated cost."""
    return get_remaining_budget() >= estimated_cost
