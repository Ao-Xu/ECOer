"""Counterfactual generation wrapper for the reusable ECOer API."""

from src.ecoe_optimizer import generate_counterfactual_ecoe


def generate_counterfactual(x, model, Gamma, **kwargs):
    """Generate one ECOer counterfactual for ``x``."""
    return generate_counterfactual_ecoe(x, model, Gamma, **kwargs)
