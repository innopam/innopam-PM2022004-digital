checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        122.7709,
        116.746,
        104.0937,
        122.7709,
        116.746,
        104.0937,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        68.5005,
        66.6322,
        70.3232,
        68.5005,
        66.6322,
        70.3232,
    ],
    test_cfg=dict(size_divisor=32),
    type='DualInputSegDataPreProcessor')
data_root = '/root/data/02_road/03_train/seoul_2020_2022/class1_tif_512_overlap_split'
dataset_type = 'LEVIR_CD_Dataset_Tif'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=4000, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        img_shape=(
            512,
            512,
            3,
        ),
        interval=1,
        type='CDVisualizationHook'))
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.75,
    1.0,
    1.25,
]
launcher = 'none'
load_from = '/root/open-cd/work_dirs/ban_vit-l14-georsclip_mit-b0_512x512_40k_road_seoul_overlap_split/best_mIoU_iter_4000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    asymetric_input=True,
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            122.7709,
            116.746,
            104.0937,
            122.7709,
            116.746,
            104.0937,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            68.5005,
            66.6322,
            70.3232,
            68.5005,
            66.6322,
            70.3232,
        ],
        test_cfg=dict(size_divisor=32),
        type='DualInputSegDataPreProcessor'),
    decode_head=dict(
        ban_cfg=dict(
            clip_channels=1024,
            fusion_index=[
                1,
                2,
                3,
            ],
            side_enc_cfg=dict(
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                drop_rate=0.0,
                embed_dims=32,
                in_channels=3,
                init_cfg=dict(
                    checkpoint=
                    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth',
                    type='Pretrained'),
                mlp_ratio=4,
                num_heads=[
                    1,
                    2,
                    5,
                    8,
                ],
                num_layers=[
                    2,
                    2,
                    2,
                    2,
                ],
                num_stages=4,
                out_indices=(
                    0,
                    1,
                    2,
                    3,
                ),
                patch_sizes=[
                    7,
                    3,
                    3,
                    3,
                ],
                qkv_bias=True,
                sr_ratios=[
                    8,
                    4,
                    2,
                    1,
                ],
                type='mmseg.MixVisionTransformer')),
        ban_dec_cfg=dict(
            align_corners=False,
            channels=128,
            dropout_ratio=0.1,
            in_channels=[
                32,
                64,
                160,
                256,
            ],
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=2,
            type='BAN_MLPDecoder'),
        loss_decode=dict(
            loss_weight=1.0, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        type='BitemporalAdapterHead'),
    encoder_resolution=dict(mode='bilinear', size=(
        336,
        336,
    )),
    image_encoder=dict(
        act_cfg=dict(type='mmseg.QuickGELU'),
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dims=1024,
        frozen_exclude=[],
        img_size=(
            336,
            336,
        ),
        in_channels=3,
        interpolate_mode='bicubic',
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-05, type='LN'),
        norm_eval=False,
        num_heads=16,
        num_layers=18,
        out_indices=(
            5,
            11,
            17,
        ),
        out_origin=False,
        output_cls_token=True,
        patch_bias=False,
        patch_pad=0,
        patch_size=14,
        pre_norm=True,
        qkv_bias=True,
        type='mmseg.VisionTransformer',
        with_cls_token=True),
    pretrained=
    'https://huggingface.co/likyoo/BAN/resolve/f529fbf79271afccf1bd2387d83d7e9199823cdf/pretrain/RS5M_ViT-L-14-336.pth?download=true',
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        256,
        256,
    )),
    train_cfg=dict(),
    type='BAN')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            img_encoder=dict(decay_mult=1.0, lr_mult=0.1),
            mask_decoder=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
pretrained = 'https://huggingface.co/likyoo/BAN/resolve/f529fbf79271afccf1bd2387d83d7e9199823cdf/pretrain/RS5M_ViT-L-14-336.pth?download=true'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test_filtered/A',
            img_path_to='test_filtered/B',
            seg_map_path='test_filtered/label'),
        data_root=
        '/root/data/02_road/03_train/seoul_2020_2022/class1_tif_512_overlap_split',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='MultiImgResize'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LEVIR_CD_Dataset_Tif'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='MultiImgResize'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path_from='train/A',
            img_path_to='train/B',
            seg_map_path='train/label'),
        data_root=
        '/root/data/02_road/03_train/seoul_2020_2022/class1_tif_512_overlap_split',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    512,
                    512,
                ),
                type='MultiImgRandomCrop'),
            dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
            dict(
                brightness_delta=10,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=10,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='MultiImgPhotoMetricDistortion'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LEVIR_CD_Dataset_Tif'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
    dict(
        cat_max_ratio=0.75, crop_size=(
            512,
            512,
        ), type='MultiImgRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(
        brightness_delta=10,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=10,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='MultiImgPhotoMetricDistortion'),
    dict(type='MultiImgPackSegInputs'),
]
tta_model = dict(type='mmseg.SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='MultiImgLoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True, scale_factor=0.75, type='MultiImgResize'),
                dict(keep_ratio=True, scale_factor=1.0, type='MultiImgResize'),
                dict(
                    keep_ratio=True, scale_factor=1.25, type='MultiImgResize'),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='MultiImgRandomFlip'),
            ],
            [
                dict(type='MultiImgLoadAnnotations'),
            ],
            [
                dict(type='MultiImgPackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test_filtered/A',
            img_path_to='test_filtered/B',
            seg_map_path='test_filtered/label'),
        data_root=
        '/root/data/02_road/03_train/seoul_2020_2022/class1_tif_512_overlap_split',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='MultiImgResize'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LEVIR_CD_Dataset_Tif'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    save_dir=
    '/root/data/open-cd/inference_results/ban_vit-l14-georsclip_mit-b0_512x512_40k_road_seoul_overlap_split',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = './work_dirs/ban_vit-l14-georsclip_mit-b0_512x512_40k_road_seoul_overlap_split'
