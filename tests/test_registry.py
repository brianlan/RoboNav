from prefusion.registry import METRICS, MODELS, MODEL_FEEDERS, TENSOR_SMITHS

import robonav  # noqa: F401


def test_robonav_registrations():
    expected = (
        (MODELS, "AquaNet", "robonav.aqua.model.aqua"),
        (MODELS, "FrameBatchMerger", "robonav.common.model.data_preprocessor"),
        (MODEL_FEEDERS, "AquaModelFeeder", "robonav.aqua.model_feeder.aqua_model_feeder"),
        (TENSOR_SMITHS, "CameraImageTensor", "robonav.aqua.tensor_smith.camera_tensor_smith"),
        (METRICS, "DummyAccuracyMetric", "robonav.common.metric"),
    )

    for registry, name, module in expected:
        assert registry.get(f"robonav.{name}").__module__ == module

    metric = METRICS.build({"type": "robonav.DummyAccuracyMetric"})
    metric.process(None, None)
    assert metric.evaluate(1) == {"accuracy": 0.0}


if __name__ == "__main__":
    test_robonav_registrations()
