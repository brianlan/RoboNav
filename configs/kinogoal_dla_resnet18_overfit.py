import datetime

experiment_name = "kinogoal_dla_resnet18_overfit"
today = datetime.datetime.now().strftime("%m%d")

custom_imports = dict(
    imports=[
        "prefusion",
        "robonav",
        "mmdet",
        "mmengine",
    ],
    allow_failed_imports=False,
)
default_scope = "prefusion"

backend_args = None

# Add this to enable unused parameter detection for DDP
find_unused_parameters = True

num_gpus = 1
batch_size = 2
num_epochs = 2000
possible_sequence_lengths = [20]

# Single source for the depth range: used by both CameraDepthTensor
# (target normalization) and the depth loss (meters reconstruction).
max_depth = 5

# Virtual Camera settings
camera_settings = {
    "FISHEYE": dict(
        cam_type="FisheyeCamera",
        resolution=(512, 384),
        euler_angles=[-90, 0, -90],
        intrinsic=[255.5, 191.5, 130.72, 130.72, 0.1, 0, 0, 0],
    ),
}

camera_mapping = [
    ("fisheye", "FISHEYE"),
]

transformables = dict(
    camera_images=dict(
        type="CameraImageSet",
        loader=dict(type="SyncCameraImageSetLoader", camera_mapping=camera_mapping),
        tensor_smith=dict(
            type="CameraImageTensor",
            means=[127.5, 127.5, 127.5],
            stds=[127.5, 127.5, 127.5],
        ),
    ),
    camera_depths=dict(
        type="CameraDepthSet",
        loader=dict(type="CameraDepthSetLoader", camera_mapping=camera_mapping),
        tensor_smith=dict(type="robonav.CameraDepthTensor", max_depth=max_depth),
    ),
    ego_poses=dict(type="EgoPoseSet"),
    goal=dict(
        type="robonav.Goal",
        loader=dict(type="robonav.GoalLoader"),
        tensor_smith=dict(type="robonav.GoalTensorSmith"),
    ),
    future_trajectory=dict(
        type="robonav.FutureTrajectory",
        loader=dict(type="robonav.FutureTrajectoryLoader"),
        tensor_smith=dict(type="robonav.FutureTrajectoryTensorSmith"),
    ),
    navigation_map_2d=dict(
        type="robonav.NavigationMap2D",
        loader=dict(type="robonav.NavigationMap2DLoader"),
        tensor_smith=dict(
            type="robonav.NavigationMap2DTensorSmith",
            output_range=(-1, -3, 5, 3),
            resolution=0.05,
        ),
    ),
)

model_feeder = dict(
    type="robonav.AquaModelFeeder",
    pe_downsample_factor=2,
    pe_range=(0, -5, 10, 5),
)

train_dataset = dict(
    type="StreamingSequenceBatchDataset",
    name="MvParkingTest",
    data_root="/ssd5/datasets/kino-goal-nav/prefusion",
    info_path="/ssd5/datasets/kino-goal-nav/prefusion/sage3d-839920-000018/info.pkl",
    model_feeder=model_feeder,
    transformables=transformables,
    transforms=[
        dict(type="BGR2RGB"),
        dict(type="RenderVirtualCamera", camera_settings=camera_settings),
        # dict(type="RandomRenderExtrinsic", angles=[2, 2, 2]),
        # dict(type="RandomTranslateSpace", translation=(1, 1, 0)),
        # dict(type="RandomRotateSpace", angles=(0, 0, 90), prob_inverse_cameras_rotation=0),
        # dict(type="RandomMirrorSpace"),
        # dict(type="RandomImageISP", prob=0.1),
        # dict(type="RandomSetIntrinsicParam", prob=0.1, jitter_ratio=0.01),
        # dict(type="RandomSetExtrinsicParam", prob=0.1, angle=1, translation=0.02),
    ],
    sequence_sampler=dict(
        type="ValIndexSequenceSampler",
        possible_sequence_lengths=possible_sequence_lengths,
        possible_frame_intervals=[1],
    ),
)

val_dataset = dict(
    type="SequenceBatchDataset",
    name="MvParkingTest",
    data_root="/ssd5/datasets/kino-goal-nav/prefusion",
    info_path="/ssd5/datasets/kino-goal-nav/prefusion/sage3d-839920-000018/info.pkl",
    model_feeder=model_feeder,
    transformables=transformables,
    transforms=[
        dict(type="BGR2RGB"),
        dict(type="RenderVirtualCamera", camera_settings=camera_settings),
    ],
    sequence_sampler=dict(
        type="ValIndexSequenceSampler",
        possible_sequence_lengths=possible_sequence_lengths,
        possible_frame_intervals=[1],
    ),
    batch_size=batch_size,
)

test_dataset = dict(
    type="SequenceBatchDataset",
    name="MvParkingTest",
    data_root="/ssd5/datasets/kino-goal-nav/prefusion",
    info_path="/ssd5/datasets/kino-goal-nav/prefusion/sage3d-839920-000018/info.pkl",
    model_feeder=model_feeder,
    transformables=transformables,
    transforms=[
        dict(type="BGR2RGB"),
        dict(type="RenderVirtualCamera", camera_settings=camera_settings),
    ],
    sequence_sampler=dict(type="SequentialSceneFrameSequenceSampler"),
    batch_size=1,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type="DefaultSampler", shuffle=False
    ),  # shuffle handled by batch_sampler
    batch_sampler=dict(
        type="AlignedTimestepBatchSampler", shuffle=True, drop_last=False, seed=0
    ),
    collate_fn=dict(type="StreamingCollate"),
    dataset=train_dataset,
)

val_dataloader = dict(
    num_workers=0,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=val_dataset,
    persistent_workers=False,
    pin_memory=True,
)

test_dataloader = dict(
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="collate_dict"),
    dataset=test_dataset,
    persistent_workers=False,
    pin_memory=True,
)

# Phase 1 loss: single editable source for all optimized term weights
# (docs/aquanet_loss_supervision_plan.md).
loss_weights = dict(
    traj_xy=1.0,
    traj_yaw=1.0,
    traj_unit=0.1,
    traj_vel=1.0,
    kin_pos=1.0,
    kin_yaw=1.0,
    depth=1.0,
)

model = dict(
    type="robonav.AquaNet",
    data_preprocessor=dict(
        type="robonav.FrameBatchMerger",
        device="cuda",
    ),
    backbone=dict(
        type="robonav.AquaResNet18D",
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4),
        fixbn=False,
        freeze=False,
        init_cfg=None,
    ),
    feature_modulation=dict(
        type="robonav.FeatureModulation",
        f4_chans=512,
        f3_chans=256,
        out_chans=256,
    ),
    temporal_fuser=dict(
        type="robonav.TemporalFuser",
        feat_chans=256,
        hidden_chans=256,
    ),
    depth_head=dict(
        type="robonav.DepthHead",
        f4_chans=512,
        f3_chans=256,
        f2_chans=128,
        f1_chans=64,
        decoder_chans=64,
    ),
    loss=dict(
        type="robonav.AquaLoss",
        loss_weights=loss_weights,
        depth_loss=dict(
            type="robonav.MultiScaleDepthLoss",
            max_depth=max_depth,
            log_offset=0.1,
            beta=0.1,
        ),
        delta_t=0.1,  # must match future trajectory sampling
        terminal_weight=5.0,  # terminal horizon step boost for L_xy
        xy_scale=1.0,  # m
        vel_scale=1.0,  # m/s
        omega_scale=1.0,  # rad/s
        v_stop_threshold=0.1,  # m/s
        omega_stop_threshold=0.1,  # rad/s
        eps=1e-6,
        beta_xy=1.0,
        beta_vel=1.0,
        beta_kin_pos=1.0,
    ),
    trajectory_head=dict(
        type="robonav.TrajectoryHead",
        feat_chans=256,
        hidden_chans=256,
        intermediate_chans=64,
        num_trajectory_steps=20,
        out_chans=7,
    ),
)

val_evaluator = dict(type="robonav.DummyAccuracyMetric")
test_evaluator = dict(type="robonav.DummyAccuracyMetric")


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

train_cfg = dict(
    type="robonav.StreamingSequenceBPTTTrainLoop",
    max_epochs=num_epochs,
    val_interval=-1,
)  # -1 note don't eval
val_cfg = dict(type="SequenceBatchValLoop")
test_cfg = dict(type="SequenceBatchInferLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4 * batch_size * num_gpus,  # total lr = base_lr * batch_size * num_gpus.
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(
                lr_mult=0.25
            ),  # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

## scheduler configs
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        end_factor=1,
        by_epoch=False,
        begin=0,
        end=500,
    ),  # warmup
    dict(
        type="CosineAnnealingLR", by_epoch=False, begin=500, eta_min=1e-5
    ),  # main LR Scheduler
    # dict(type='PolyLR', by_epoch=False, begin=0, eta_min=0, power=1.0) # main LR Scheduler
]


visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        # dict(type="AimVisBackend", init_kwargs=dict(repo="aim://10.243.28.41:5380")),
        dict(type="TensorboardVisBackend"),
    ],
)

# Smooth weighted loss_* terms and detached *_raw values over the same
# window (default pattern leaves *_raw as instantaneous values, so a
# weight-1 term appears to differ from its raw term); ordinary metrics
# such as ADE/FDE stay instantaneous.
log_processor = dict(
    type="robonav.StreamingSequenceBPTTLogProcessor",
    tabulate_ncols=5,
    mean_pattern=r".*(loss|time|data_time|grad_norm|gradnorm|_raw).*",
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", interval=100, save_best="accuracy", rule="greater"
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

custom_hooks = []

work_dir = f"./work_dirs/{experiment_name}_{today}"
# load_from = "./ckpts/wuhan_vov_pretrain_0601.pth"

resume = False
