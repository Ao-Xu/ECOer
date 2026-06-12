"""ELM reconstruction helpers for the reusable ECOer API."""

from src.r2snn import build_elm_reconstruction, reconstruct_input


def build_reconstruction(model, X_data, **kwargs):
    """Build the linear ELM reconstruction map Gamma."""
    return build_elm_reconstruction(model, X_data, **kwargs)


def reconstruct(e, Gamma):
    """Map feature-space points back to the input space."""
    return reconstruct_input(e, Gamma)
