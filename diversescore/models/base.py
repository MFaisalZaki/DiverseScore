from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.shortcuts import *

class Model:
    """Base class for all models."""

    def __init__(self, metrics:list, normalize:bool, name:str) -> None:
        """Initialize model by generating the grounded actions for the planning task."""
        self.name         = name
        self.metrics      = metrics
        self.normalize    = normalize
        self.pairwise_cnt = 0

    def __call__(self, planset) -> list:
        """Compute the diversity model for planset."""
        metricscores = {}
        for metric in self.metrics:
            if not metric in metricscores:
                metricscores[str(metric)] = []
            metric(planset)
            for _i, ip in enumerate(planset):
                for jp in planset[_i+1:]:
                    metricscores[str(metric)].append(round(1.0 - metric(ip, jp),  5))
                    self.pairwise_cnt += 1
        ret_scores = [0 for _ in range(0, self.pairwise_cnt+1)]
        for metric, scores in metricscores.items():
            for _i, score in enumerate(scores):
                ret_scores[_i] += score
        for _i, score in enumerate(ret_scores):
            ret_scores[_i] = round(score/len(metricscores), 5)
        return ret_scores
        
    