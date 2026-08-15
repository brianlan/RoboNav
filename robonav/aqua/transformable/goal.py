import numpy as np

from prefusion.dataset.transform import SpatialTransformable, _normalize_3d_vector
from prefusion.dataset.tensor_smith import TensorSmith

from robonav.registry import TRANSFORMABLES


__all__ = ["Goal"]


@TRANSFORMABLES.register_module()
class Goal(SpatialTransformable):
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
        """The goal in 3D space.

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        rotation : np.ndarray
            rotation matrix, of shape (3, 3)
        translation : np.ndarray
            translation vector of exactly three real finite elements, given as a flat (3,),
            row (1, 3), or column (3, 1) vector; normalized and exposed as a (3, 1) float32 column vector
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        linear_velocity : np.ndarray, optional
            instantaneous vehicle-frame linear velocity in m/s, of exactly three real finite elements,
            X forward, Y left, Z up in the right-handed FLU frame, by default None
        angular_velocity : np.ndarray, optional
            instantaneous vehicle-frame angular velocity in rad/s, of exactly three real finite elements,
            direction following the right-hand rule, by default None
        """
        super().__init__(name)
        self.rotation = rotation
        self.translation = _normalize_3d_vector(translation)
        self.linear_velocity = _normalize_3d_vector(linear_velocity)
        self.angular_velocity = _normalize_3d_vector(angular_velocity)
        self.tensor_smith = tensor_smith

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, "up down flip is unnecessary."

        # in the mirror world, assume that a object is left-right symmetrical
        # however, y-axis of object coordinate is left-right
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        self.rotation = flip_mat @ self.rotation @ flip_mat_self.T
        self.translation = flip_mat @ self.translation

        if self.linear_velocity is not None:
            self.linear_velocity = flip_mat @ self.linear_velocity  # polar vector
        if self.angular_velocity is not None:
            self.angular_velocity = np.linalg.det(flip_mat) * (
                flip_mat @ self.angular_velocity
            )  # axial vector

        return self

    def rotate_3d(self, rmat, **kwargs):
        self.rotation = rmat @ self.rotation
        self.translation = rmat @ self.translation
        if self.linear_velocity is not None:
            self.linear_velocity = rmat @ self.linear_velocity
        if self.angular_velocity is not None:
            self.angular_velocity = rmat @ self.angular_velocity
        return self

    def translate_3d(self, translation, **kwargs):
        translation = np.asarray(translation, dtype=np.float32).reshape(3, 1)
        self.translation = self.translation - translation
        return self
