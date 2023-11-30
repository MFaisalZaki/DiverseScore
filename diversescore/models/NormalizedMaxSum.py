
from .base import Model

class NormalizedMaxSum(Model):
    """NormalizedMaxSum metric class.

    This class implements the MaxSum metric from [1] and returns the normalized value.

    References
    ----------
    .. [1] F. Sandoya, A. Mart´ınez-Gavara, R. Aceves, A. Duarte,
           and R. Mart´ı, “Diversity and equity models,” in Handbook of Heuristics, 2018.
    """

    def __init__(self, task, metric):
        """Initialize the metric object."""
        super(NormalizedMaxSum, self).__init__(task=task, metric=metric, name="NormalizedMaxSum")
        
    def __call__(self, planset):
        """Compute the metric for planset."""
        _mscore = []
        countpairs = 0
        for _i, ip in enumerate(planset):
            for jp in planset[_i+1:]:
                countpairs += 1
                _mscore.append(round(1.0 - self.metric(ip, jp),  5))
        return round(sum(_mscore)/countpairs, 5)
