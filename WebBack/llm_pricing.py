"""Approximate list prices (USD per 1M tokens) for cost telemetry. Update when OpenAI changes pricing."""

from __future__ import annotations

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the estimated cost, or None when the model is not in the table (unknown price)."""
    base = model.split(":")[0]
    for name, (price_in, price_out) in PRICES_PER_MILLION.items():
        if base == name or base.startswith(name + "-20"):
            return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)
    return None
