import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from easydict import EasyDict as edict
from loguru import logger
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from robonav.util.helpers import ensured_path, parent_ensured_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-scene-root", type=Path, required=True)
    parser.add_argument("--output-scene-root", type=ensured_path, required=True)
    parser.add_argument("--info-pkl-save-path", type=parent_ensured_path, required=True)
    parser.add_argument("--scene-ids", nargs="*")
    parser.add_argument("--clone-heavy-data", default=False, action="store_true")
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})


# Only parse arguments when running as main script
args = None


def main():
    global args
    args = parse_arguments()
 

if __name__ == "__main__":
    main()
