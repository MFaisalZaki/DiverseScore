
from .base import Model

class MaxMean(Model):
    """MaxMean metric class.

    This class implements the MaxMean metric from [1].

    References
    ----------
    .. [1] F. Sandoya, A. Mart´ınez-Gavara, R. Aceves, A. Duarte,
           and R. Mart´ı, “Diversity and equity models,” in Handbook of Heuristics, 2018.
    """

    def __init__(self, task, metric):
        """Initialize the metric object."""
        super(MaxMean, self).__init__(task=task, metric=metric, name="MaxMean")
        
    def __call__(self, planset):
        """Compute the metric for planset."""
        _mscore = []
        for _i, ip in enumerate(planset):
            for jp in planset[_i+1:]:
                _mscore.append(round(1.0 - self.metric(ip, jp),  5))
        return round(sum(_mscore)/len(planset), 5)
