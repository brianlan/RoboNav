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
        tensor_smith=dict(type="robonav.CameraDepthTensor", max_depth=5),
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
        # dict(type="RandomTranslateSpace", translation=(1, 1, 1)),
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
    neck=dict(
        type="robonav.TvFPN", in_channels_list=[64, 128, 256, 512], out_channels=256
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
    type="StreamingSequenceBatchTrainLoop", max_epochs=num_epochs, val_interval=-1
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

log_processor = dict(type="prefusion.SequenceAwareLogProcessor", tabulate_ncols=5)

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
