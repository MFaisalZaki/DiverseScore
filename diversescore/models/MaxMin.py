
from .base import Model

class MaxMin(Model):
    """MaxMin metric class.

    This class implements the MaxMin metric from [1].

    References
    ----------
    .. [1] F. Sandoya, A. Mart´ınez-Gavara, R. Aceves, A. Duarte,
           and R. Mart´ı, “Diversity and equity models,” in Handbook of Heuristics, 2018.
    """

    def __init__(self, metric, normalize=False):
        """Initialize the metric object."""
        super(MaxMin, self).__init__(metric=metric, normalize=normalize, name="MaxMin")
        
    def __call__(self, planset):
        """Compute the metric for planset."""
        return round(min(super().__call__(planset)), 5)
