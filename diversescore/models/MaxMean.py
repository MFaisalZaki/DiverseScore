
from .base import Model

class MaxMean(Model):
    """MaxMean metric class.

    This class implements the MaxMean metric from [1].

    References
    ----------
    .. [1] F. Sandoya, A. Mart´ınez-Gavara, R. Aceves, A. Duarte,
           and R. Mart´ı, “Diversity and equity models,” in Handbook of Heuristics, 2018.
    """

    def __init__(self, metrics, normalize=False):
        """Initialize the metric object."""
        super(MaxMean, self).__init__(metrics=metrics, normalize=normalize, name="MaxMean")
        
    def __call__(self, planset):
        """Compute the metric for planset."""
        return round(sum(super().__call__(planset))/len(planset), 5)
