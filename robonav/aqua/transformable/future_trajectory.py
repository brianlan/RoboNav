import numpy as np
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.transform import SpatialTransformable

from robonav.registry import TRANSFORMABLES


__all__ = ["FutureTrajectory"]


@TRANSFORMABLES.register_module()
class FutureTrajectory(SpatialTransformable):
    def __init__(
        self,
        name: str,
        rotation: np.ndarray,
        translation: np.ndarray,
        tensor_smith: TensorSmith = None,
        *,
        linear_velocity: np.ndarray = None,
        angular_velocity: np.ndarray = None,
    ):
        """Ordered future ego states in the current frame body coordinate system.

        Each of the N steps follows the Goal semantics, stored as batched arrays:
        vector fields are row vectors of shape (N, 3), not (N, 3, 1) columns.

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        rotation : np.ndarray
            per-step rotation matrices, of shape (N, 3, 3)
        translation : np.ndarray
            per-step translation vectors, of shape (N, 3)
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        linear_velocity : np.ndarray, optional
            per-step vehicle-frame linear velocities in m/s, of shape (N, 3), by default None
        angular_velocity : np.ndarray, optional
            per-step vehicle-frame angular velocities in rad/s, of shape (N, 3), by default None
        """
        super().__init__(name)
        self.rotation = np.asarray(rotation)
        self.translation = np.asarray(translation)
        self.linear_velocity = None if linear_velocity is None else np.asarray(linear_velocity)
        self.angular_velocity = None if angular_velocity is None else np.asarray(angular_velocity)

        self._validate()

        self.tensor_smith = tensor_smith

    def _validate(self):
        fields = {"rotation": (self.rotation, (3, 3)), "translation": (self.translation, (3,))}
        for name in ("linear_velocity", "angular_velocity"):
            if getattr(self, name) is not None:
                fields[name] = (getattr(self, name), (3,))
        for name, (values, trailing) in fields.items():
            if values.shape[1:] != trailing:
                raise ValueError(
                    f"future_trajectory field {name} must have shape (N, {', '.join(map(str, trailing))}), "
                    f"but got {values.shape}"
                )
        num_steps = len(self.rotation)
        if any(len(values) != num_steps for values, _ in fields.values()):
            raise ValueError(
                "future_trajectory fields have inconsistent numbers of steps"
            )

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, "up down flip is unnecessary."

        # in the mirror world, assume that a object is left-right symmetrical
        # however, y-axis of object coordinate is left-right
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        self.rotation = flip_mat @ self.rotation @ flip_mat_self.T
        self.translation = self.translation @ flip_mat.T  # polar vector

        if self.linear_velocity is not None:
            self.linear_velocity = self.linear_velocity @ flip_mat.T  # polar vector
        if self.angular_velocity is not None:
            self.angular_velocity = np.linalg.det(flip_mat) * (
                self.angular_velocity @ flip_mat.T
            )  # axial vector
        return self

    def rotate_3d(self, rmat, **kwargs):
        self.rotation = rmat @ self.rotation
        self.translation = self.translation @ rmat.T
        if self.linear_velocity is not None:
            self.linear_velocity = self.linear_velocity @ rmat.T
        if self.angular_velocity is not None:
            self.angular_velocity = self.angular_velocity @ rmat.T
        return self

    def translate_3d(self, translation, **kwargs):
        translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
        self.translation = self.translation - translation
        return self
