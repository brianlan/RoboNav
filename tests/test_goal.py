import numpy as np
import torch

from robonav.aqua.tensor_smith.goal_tensor_smith import GoalTensorSmith
from robonav.aqua.transformable.goal import Goal
from robonav.aqua.transformable_loader.goal_loader import GoalLoader


def _goal():
    return Goal(
        "goal",
        rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        translation=[1, 2, 3],
        linear_velocity=[4, 5, 6],
        angular_velocity=[7, 8, 9],
    )


def test_goal_flip_3d():
    goal = _goal().flip_3d(np.diag([1, -1, 1]))

    np.testing.assert_allclose(goal.rotation, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(goal.translation.ravel(), [1, -2, 3])
    np.testing.assert_allclose(goal.linear_velocity.ravel(), [4, -5, 6])
    np.testing.assert_allclose(goal.angular_velocity.ravel(), [-7, 8, -9])


def test_goal_rotate_3d():
    rmat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    goal = _goal().rotate_3d(rmat)

    np.testing.assert_allclose(goal.rotation, [[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    np.testing.assert_allclose(goal.translation.ravel(), [1, -3, 2])
    np.testing.assert_allclose(goal.linear_velocity.ravel(), [4, -6, 5])
    np.testing.assert_allclose(goal.angular_velocity.ravel(), [7, -9, 8])


def test_goal_translate_3d():
    goal = _goal().translate_3d([0.5, 1, 1.5])

    np.testing.assert_allclose(goal.rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(goal.translation.ravel(), [0.5, 1, 1.5])
    np.testing.assert_allclose(goal.linear_velocity.ravel(), [4, 5, 6])
    np.testing.assert_allclose(goal.angular_velocity.ravel(), [7, 8, 9])


def _frame_data():
    return {
        "goal": {
            "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [1, 2, 3],
            "linear_velocity": [4, 5, 6],
            "angular_velocity": [7, 8, 9],
        }
    }


def test_goal_loader_field_mapping():
    smith = GoalTensorSmith()
    goal = GoalLoader(data_root=None).load(
        "goal", None, _frame_data(), None, tensor_smith=smith
    )

    assert goal.name == "goal"
    np.testing.assert_allclose(goal.rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(goal.translation.ravel(), [1, 2, 3])
    np.testing.assert_allclose(goal.linear_velocity.ravel(), [4, 5, 6])
    np.testing.assert_allclose(goal.angular_velocity.ravel(), [7, 8, 9])
    assert goal.tensor_smith is smith

    goal.to_tensor()  # tensor_smith propagates through to_tensor
    assert goal.tensor["translation"].shape == (3, 1)


def test_goal_tensor_smith_tensors():
    out = GoalTensorSmith()(_goal())

    assert set(out) == {"rotation", "translation", "linear_velocity", "angular_velocity"}
    for v in out.values():
        assert v.dtype == torch.float32
    assert out["rotation"].shape == (3, 3)
    assert out["translation"].shape == (3, 1)
    assert out["linear_velocity"].shape == (3, 1)
    assert out["angular_velocity"].shape == (3, 1)
    np.testing.assert_allclose(out["translation"].numpy(), [[1], [2], [3]])


def test_goal_tensor_smith_without_velocities():
    frame_data = {"goal": {"rotation": np.eye(3).tolist(), "translation": [1, 2, 3]}}
    goal = GoalLoader(data_root=None).load("goal", None, frame_data, None)

    out = GoalTensorSmith()(goal)

    assert set(out) == {"rotation", "translation"}
    for v in out.values():
        assert v.dtype == torch.float32
