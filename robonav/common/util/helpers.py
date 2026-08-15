from functools import partial
from pathlib import Path

import numpy as np

__all__ = ["ensured_path", "parent_ensured_path", "rt2mat"]


def ensured_path(input, ensure_parent=False):
    """Often used in the scenario that the path we want to write things to is ensured to be exist."""
    p = Path(input)
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


parent_ensured_path = partial(ensured_path, ensure_parent=True)


def points3d_to_homo(points3d: np.ndarray) -> np.ndarray:
    return np.concatenate((points3d, np.ones(len(points3d))[:, None]), axis=1)


def rt2mat(
    rotation: np.ndarray, translation: np.ndarray, as_homo: bool = False
) -> np.ndarray:
    """A helper function that convert rotation matrix (3x3) and translation (3,) to transformation matrix representation.

    Parameters
    ----------
    rotation : np.ndarray
        rotation matrix of shape (3, 3)
    translation : np.ndarray
        translation of shape (3,)
    as_homo: bool
        if true, the matrix will be saved as homogeneous (4x4), otherwise, it will be saved as 3x4

    Returns
    -------
    np.ndarray
        of shape (3, 4) if as_homo == False, otherwise, (4, 4)
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation.flatten()
    if as_homo:
        return T
    return T[:3, :]
