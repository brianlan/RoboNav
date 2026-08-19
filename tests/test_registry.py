from prefusion.registry import (
    LOOPS,
    METRICS,
    MODELS,
    MODEL_FEEDERS,
    TENSOR_SMITHS,
    TRANSFORMABLES,
    TRANSFORMABLE_LOADERS,
)

import robonav  # noqa: F401


def test_robonav_registrations():
    expected = (
        (LOOPS, "StreamingSequenceBPTTTrainLoop", "robonav.aqua.runner.loop"),
        (MODELS, "AquaNet", "robonav.aqua.model.aqua"),
        (MODELS, "AquaLoss", "robonav.aqua.loss.aqua_loss"),
        (MODELS, "MultiScaleDepthLoss", "robonav.aqua.loss.multiscale_depth_loss"),
        (MODELS, "FrameBatchMerger", "robonav.common.model.data_preprocessor"),
        (MODEL_FEEDERS, "AquaModelFeeder", "robonav.aqua.model_feeder.aqua_model_feeder"),
        (TENSOR_SMITHS, "CameraDepthTensor", "robonav.aqua.tensor_smith.camera_tensor_smith"),
        (TENSOR_SMITHS, "GoalTensorSmith", "robonav.aqua.tensor_smith.goal_tensor_smith"),
        (
            TENSOR_SMITHS,
            "FutureTrajectoryTensorSmith",
            "robonav.aqua.tensor_smith.future_trajectory_tensor_smith",
        ),
        (TRANSFORMABLES, "Goal", "robonav.aqua.transformable.goal"),
        (
            TRANSFORMABLES,
            "FutureTrajectory",
            "robonav.aqua.transformable.future_trajectory",
        ),
        (TRANSFORMABLES, "NavigationMap2D", "robonav.aqua.transformable.navigation_map_2d"),
        (TRANSFORMABLE_LOADERS, "GoalLoader", "robonav.aqua.transformable_loader.goal_loader"),
        (
            TRANSFORMABLE_LOADERS,
            "FutureTrajectoryLoader",
            "robonav.aqua.transformable_loader.future_trajectory_loader",
        ),
        (TRANSFORMABLE_LOADERS, "NavigationMap2DLoader", "robonav.aqua.transformable_loader.navigation_map_2d_loader"),
        (TENSOR_SMITHS, "NavigationMap2DTensorSmith", "robonav.aqua.tensor_smith.navigation_map_2d_tensor_smith"),
        (METRICS, "DummyAccuracyMetric", "robonav.common.metric"),
    )

    for registry, name, module in expected:
        assert registry.get(f"robonav.{name}").__module__ == module

    metric = METRICS.build({"type": "robonav.DummyAccuracyMetric"})
    metric.process(None, None)
    assert metric.evaluate(1) == {"accuracy": 0.0}


if __name__ == "__main__":
    test_robonav_registrations()
