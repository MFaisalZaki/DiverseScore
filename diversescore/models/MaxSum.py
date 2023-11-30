
from .base import Model

class MaxSum(Model):
    """MaxSum metric class.

    This class implements the MaxSum metric from [1].

    References
    ----------
    .. [1] F. Sandoya, A. Mart´ınez-Gavara, R. Aceves, A. Duarte,
           and R. Mart´ı, “Diversity and equity models,” in Handbook of Heuristics, 2018.
    """

    def __init__(self, metric, normalize=False):
        """Initialize the metric object."""
        super(MaxSum, self).__init__(metric=metric, normalize=normalize, name="MaxSum")
        
    def __call__(self, planset):
        """Compute the metric for planset."""
        _mscore = []
        countpairs = 0
        for _i, ip in enumerate(planset):
            for jp in planset[_i+1:]:
                _mscore.append(round(1.0 - self.metric(ip, jp),  5))
                countpairs += 1
        return round(sum(_mscore)/countpairs, 5) if self.normalize else round(sum(_mscore), 5)
