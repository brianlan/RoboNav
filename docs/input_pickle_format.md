# Prefusion Info Pickle Format Specification

**Version**: 1.0.0  
**Last Updated**: 2026-02-03

---

## Table of Contents

1. [Overview](#overview)
2. [Data Hierarchy](#data-hierarchy)
3. [Frame ID Specification](#frame-id-specification)
4. [Data Root and Scene Folder Structure](#data-root-and-scene-folder-structure)
5. [Info Pickle Schema](#info-pickle-schema)
6. [Frame Pickle Schema](#frame-pickle-schema)
7. [Data Representation Specifications](#data-representation-specifications)
8. [Reference Conventions](#reference-conventions)
9. [Notes and Deprecations](#notes-and-deprecations)

---

## Overview

This document specifies the data format for the prefusion dataset framework. The format is designed to support multi-modal perception tasks including camera images, LiDAR point clouds, 3D bounding boxes, and map polylines.

The prefusion format uses a two-level pickle structure:
- **Info Pickle**: Contains metadata and frame indexing for all scenes
- **Frame Pickles**: Contain actual sensor data and annotations for individual timestamps

---

## Data Hierarchy

```
data_root/
├── info.pkl                    # Info pickle (top-level index)
├── scene_001/                  # Scene folder
│   ├── frame_info_pkl/         # Frame pickle directory
│   │   ├── 1234567890.pkl     # Individual frame data
│   │   ├── 1234567891.pkl
│   │   └── ...
│   ├── samples/                # Camera images (or image_undistortion/)
│   │   ├── CAM_FRONT/
│   │   ├── CAM_BACK/
│   │   └── ...
│   ├── lidar_points/           # LiDAR point clouds
│   │   └── lidar_top/
│   │       ├── 1234567890.pcd
│   │       └── ...
│   └── self_mask/              # Camera ego masks (recommended)
│       ├── CAM_FRONT.png
│       └── ...
├── scene_002/
└── ...
```

---

## Frame ID Specification

- **Format**: String representing timestamp in milliseconds
- **Examples**: `"1729059039200.000"`, `"1531883530449"`
- **Note**: Both formats with and without decimal points are valid
- **Type**: Must be string (not float or int)

---

## Data Root and Scene Folder Structure

### Required Directory Structure

Each scene folder must contain:

```
scene_name/
├── frame_info_pkl/          # Required: Frame pickle files
├── samples/                 # Required: Camera images
│   └── {cam_id}/           # One subdirectory per camera
├── lidar_points/            # Required if using LiDAR
│   └── {lidar_id}/         # One subdirectory per LiDAR sensor
└── self_mask/               # Required: Camera ego masks
    └── {cam_id}.png        # One mask file per camera
```

**Note**: The exact subdirectory names (`samples/`, `lidar_points/`, `self_mask/`) can vary between datasets, but the structure must be consistent within a dataset.

---

## Info Pickle Schema

The info pickle is a dictionary mapping scene IDs to scene information.

```python
{
    "scene_001": {
        "scene_info": {...},        # Optional: Scene-level metadata
        "meta_info": {...},         # Deprecated: Will be removed
        "frame_info": {
            "1729059039200.000": "scene_001/frame_info_pkl/1729059039200.000.pkl",
            "1729059039700.000": "scene_001/frame_info_pkl/1729059039700.000.pkl",
            ...
        }
    },
    "scene_002": {...},
    ...
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scene_info` | dict | No | Scene-level metadata (see below) |
| `meta_info` | dict | **Deprecated** | Not used, will be removed |
| `frame_info` | dict | **Yes** | Mapping from frame_id to frame pickle path |

---

## Frame Pickle Schema

The frame pickle contains all data for a single timestamp.

### Required Fields

```python
{
    # Required: Camera images and calibrations
    "camera_image": {
        "CAM_FRONT": {
            "path": "scene_001/samples/CAM_FRONT/1234567890.jpg",
            "calibration": {
                "camera_type": "PerspectiveCamera",  # or "FisheyeCamera"
                "intrinsic": [cx, cy, fx, fy],       # 4-element list
                "extrinsic": [R, t]                  # R: (3,3), t: (3,)
            }
        },
        "CAM_BACK": {...},
        ...
    },
    
    # Required: Ego vehicle pose in world coordinates
    "ego_pose": {
        "rotation": [[...], [...], [...]],    # (3,3) rotation matrix
        "translation": [x, y, z]               # (3,) translation vector
    },
    
    # Required: Scene-level information
    "scene_info": {
        "camera_mask": {
            "CAM_FRONT": "scene_001/self_mask/CAM_FRONT.png",
            "CAM_BACK": "scene_001/self_mask/CAM_BACK.png",
            ...
        }
    }
}
```

### Optional Fields

```python
{
    # Optional: 3D bounding boxes
    "3d_boxes": [
        {
            "class": "car",                    # Object class
            "attr": {...},                      # Object attributes
            "size": [length, width, height],    # 3D dimensions
            "rotation": [[...], [...], [...]],  # (3,3) rotation matrix
            "translation": [x, y, z],           # (3,) position
            "velocity": [vx, vy, vz],           # (3,) velocity
            "track_id": "obj_001",              # Unique tracking ID
            "visibility": 1.0                   # Optional: visibility score
        },
        ...
    ],
    
    # Optional: 3D map polylines
    "3d_polylines": [
        {
            "class": "divider",                # Polyline class
            "attr": {...},                      # Attributes
            "points": [[x,y,z], ...],           # (N,3) array of points
            "track_id": "line_001"              # Optional: tracking ID
        },
        ...
    ],
    
    # Optional: LiDAR point clouds
    "lidar_points": {
        "lidar_top": {                         # Sensor ID (dataset-dependent)
            "path": "scene_001/lidar_points/lidar_top/1234567890.pcd"
        },
        ...
    },
    
    # Optional: Additional metadata
    "sample_token": "...",                     # Dataset-specific token
    "dataset_name": "nuscenes",                # Dataset identifier
    "timestamp_window": [...],                 # Temporal window info
    "can_bus": {...}                           # CAN bus data
}
```

### Field Descriptions

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `camera_image` | dict | Mapping from camera_id to image data and calibration |
| `camera_image.{cam_id}.path` | str | Relative path to image file |
| `camera_image.{cam_id}.calibration` | dict | Camera calibration parameters |
| `ego_pose` | dict | Ego vehicle pose in world coordinates |
| `ego_pose.rotation` | (3,3) array | Rotation matrix (ego → world) |
| `ego_pose.translation` | (3,) array | Translation vector (ego → world) |
| `scene_info` | dict | Scene-level configuration |
| `scene_info.camera_mask` | dict | Mapping from camera_id to ego mask path |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `3d_boxes` | list | List of 3D bounding box annotations |
| `3d_polylines` | list | List of 3D map polyline annotations |
| `lidar_points` | dict | Mapping from lidar_id to point cloud paths |
| `sample_token` | str | Dataset-specific unique identifier |
| `dataset_name` | str | Name of the source dataset |

---

## Data Representation Specifications

### 1. Ego Vehicle Coordinate System

The ego vehicle coordinate system is **right-handed** with:

- **Origin**: Center of the ego vehicle
- **X-axis**: Points forward (longitudinal direction)
- **Y-axis**: Points left (lateral direction)
- **Z-axis**: Points up (vertical direction)

```
    Z (up)
    |
    |____ Y (left)
   /
  X (forward)
```

**Important**: All dynamic objects, map polylines, and sensor extrinsics are expressed relative to this ego frame.

### 2. Sensor Extrinsic Convention

The sensor extrinsic transformation follows the **sensor → ego** convention:

```
p_ego = R @ p_sensor + t
```

Where:
- `R`: (3,3) rotation matrix from sensor to ego
- `t`: (3,) translation vector from sensor to ego
- `p_sensor`: Point in sensor coordinates
- `p_ego`: Point in ego coordinates

### 3. Camera Intrinsic Format

Camera intrinsics are stored as a 4-element list:

```python
intrinsic = [cx, cy, fx, fy]
```

Where:
- `cx`: Principal point x-coordinate (pixels)
- `cy`: Principal point y-coordinate (pixels)
- `fx`: Focal length in x-direction (pixels)
- `fy`: Focal length in y-direction (pixels)

**Note**: For `PerspectiveCamera`, only these 4 parameters are needed. For `FisheyeCamera`, additional distortion parameters may be present but are not required by the base specification.

### 4. Ego Pose Format

The ego pose provides the transformation from ego to world coordinates:

```python
{
    "rotation": R,      # (3,3) rotation matrix: ego → world
    "translation": t    # (3,) translation vector: ego → world
}
```

To transform a point from ego to world:

```
p_world = R @ p_ego + t
```

### 5. 3D Bounding Box Format

Each 3D bounding box is defined by:

- **Center**: `translation` - (3,) position in ego coordinates
- **Size**: `size` - [length, width, height] in meters
- **Orientation**: `rotation` - (3,3) rotation matrix
- **Class**: `class` - Object class name (string)
- **Attributes**: `attr` - Object attributes (dict or list)
- **Velocity**: `velocity` - (3,) velocity vector (optional)
- **Track ID**: `track_id` - Unique identifier for tracking (optional)
- **Visibility**: `visibility` - Visibility score 0-1 (optional)

The box dimensions are defined as:
- **Length**: Size along the X-axis (forward direction)
- **Width**: Size along the Y-axis (left direction)
- **Height**: Size along the Z-axis (up direction)

### 6. 3D Polyline Format

Each 3D polyline is defined by:

- **Points**: `points` - (N,3) array of 3D points in ego coordinates, or (3,) for single-point polylines
- **Class**: `class` - Polyline class name (string)
- **Attributes**: `attr` - Polyline attributes (dict or list)
- **Track ID**: `track_id` - Unique identifier (optional)

### 7. LiDAR Point Cloud Format

LiDAR point clouds are stored in **PCD (Point Cloud Data)** format. The PCD file must contain at minimum:

- **x, y, z**: Point coordinates in sensor frame

Additional fields (intensity, timestamp, etc.) may be present but are optional.

---

## Reference Conventions

### Image References

Images are referenced via relative paths from the data root:

```python
"camera_image": {
    "CAM_FRONT": {
        "path": "scene_001/samples/CAM_FRONT/1234567890.jpg"
    }
}
```

### Camera Mask References

Camera masks (ego masks) are referenced via relative paths:

```python
"scene_info": {
    "camera_mask": {
        "CAM_FRONT": "scene_001/self_mask/CAM_FRONT.png"
    }
}
```

**Note**: Masks should be grayscale images where:
- Value 0 (or 0.0): Invalid/ego region
- Value 255 (or 1.0): Valid region

### LiDAR Point Cloud References

LiDAR point clouds are referenced via relative paths:

```python
"lidar_points": {
    "lidar_top": {
        "path": "scene_001/lidar_points/lidar_top/1234567890.pcd"
    }
}
```

**Note**: The LiDAR sensor ID (e.g., `lidar_top`, `LIDAR_TOP`) is dataset-dependent and not standardized.

---

## Notes and Deprecations

### Deprecated Fields

- **`meta_info`**: This field in the scene info is **deprecated** and will be removed in future versions. Do not rely on it.

### Important Notes

1. **Frame Pickle is Central**: Always start from the frame pickle file as the centralized data source. If external data is needed, the pickle will store paths pointing to the real data.

2. **Dataset Variations**: There may be minor differences between different data sources. This document only specifies the common format. Dataset-specific extensions are allowed but should be documented separately.

3. **Async Calibration Mode**: The documentation assumes async calibration mode where each camera stores its calibration independently. Sync mode (where calibration is stored in `scene_info`) is deprecated and will be retired.

4. **Path Resolution**: All paths in the pickle files are relative to the `data_root` directory.

5. **Coordinate System Consistency**: All geometric data (boxes, polylines, sensor extrinsics) must be expressed in the ego coordinate system unless explicitly stated otherwise.

6. **Right-Handed System**: The ego coordinate system is right-handed. Ensure all rotation matrices maintain proper orthogonality and right-handedness.

---

## Validation

To validate a dataset against this specification, use the provided validation script:

```bash
python tools/validate_dataset.py --data-root /path/to/data --info-pkl info.pkl
```

The script checks:
- Required field presence
- Data type correctness
- Path existence and readability
- Coordinate system conventions
- Camera intrinsic format

---

## Appendix: Complete Example

### Info Pickle Example

```python
{
    "scene-0001": {
        "frame_info": {
            "1531883530449": "scene-0001/frame_info_pkl/1531883530449.pkl",
            "1531883530949": "scene-0001/frame_info_pkl/1531883530949.pkl"
        }
    }
}
```

### Frame Pickle Example

```python
{
    "camera_image": {
        "CAM_FRONT": {
            "path": "scene-0001/samples/CAM_FRONT/image_001.jpg",
            "calibration": {
                "camera_type": "PerspectiveCamera",
                "intrinsic": [816.27, 491.51, 1266.42, 1266.42],
                "extrinsic": [
                    [[0.9999, -0.0123, 0.0034],
                     [0.0123, 0.9999, -0.0012],
                     [-0.0034, 0.0012, 0.9999]],
                    [1.5, 0.0, 1.2]
                ]
            }
        }
    },
    "ego_pose": {
        "rotation": [[0.9999, -0.0123, 0.0034],
                     [0.0123, 0.9999, -0.0012],
                     [-0.0034, 0.0012, 0.9999]],
        "translation": [1010.13, 610.81, 0.0]
    },
    "3d_boxes": [
        {
            "class": "car",
            "attr": {},
            "size": [4.5, 2.0, 1.8],
            "rotation": [[0.9999, -0.0123, 0.0034],
                         [0.0123, 0.9999, -0.0012],
                         [-0.0034, 0.0012, 0.9999]],
            "translation": [10.5, 2.3, 0.5],
            "velocity": [5.2, 0.1, 0.0],
            "track_id": "car_001"
        }
    ],
    "scene_info": {
        "camera_mask": {
            "CAM_FRONT": "scene-0001/self_mask/CAM_FRONT.png"
        }
    }
}
```
