"""A compact estimator-style interface for ECOer."""

from .optimizer import generate_counterfactual
from .r2snn import fit_surrogate
from .reconstruction import build_reconstruction


class ECOerExplainer:
    """Fit a surrogate and generate ECOer counterfactuals."""

    def __init__(self, **surrogate_kwargs):
        self.surrogate_kwargs = surrogate_kwargs
        self.model = None
        self.Gamma = None
        self.predictor = None

    def fit(self, X_train, predictor):
        """Fit the R2SNN surrogate and ELM reconstruction map."""
        self.predictor = predictor
        self.model = fit_surrogate(predictor, X_train, **self.surrogate_kwargs)
        self.Gamma = build_reconstruction(self.model, X_train)
        return self

    def explain(self, x, **kwargs):
        """Return the counterfactual result dictionary for one input."""
        if self.model is None or self.Gamma is None:
            raise RuntimeError("Call fit(...) before explain(...).")
        return generate_counterfactual(
            x,
            self.model,
            self.Gamma,
            clf=self.predictor,
            **kwargs,
        )
