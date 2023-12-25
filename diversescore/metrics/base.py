


class Metric:
    def __init__(self, name) -> None:
        self.name = name
        
    def __call__(self, x:tuple, y:tuple):
        """Compute the metric for the plan x and y. Note that x and y are 2-tuples such that
         the first element is the plan and the second element is the plan's states. """
        raise NotImplementedError("Metric must be implemented.")

    def __str__(self) -> str:
        return self.name