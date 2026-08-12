from mmengine.evaluator import BaseMetric

from robonav.registry import METRICS

__all__ = ["DummyAccuracyMetric"]


@METRICS.register_module()
class DummyAccuracyMetric(BaseMetric):
    default_prefix = ""

    def process(self, data_batch, data_samples):
        self.results.append(0)

    def compute_metrics(self, results):
        return {"accuracy": 0.0}
