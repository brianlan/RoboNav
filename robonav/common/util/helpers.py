from functools import partial
from pathlib import Path


def ensured_path(input, ensure_parent=False):
    """Often used in the scenario that the path we want to write things to is ensured to be exist."""
    p = Path(input)
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


parent_ensured_path = partial(ensured_path, ensure_parent=True)
