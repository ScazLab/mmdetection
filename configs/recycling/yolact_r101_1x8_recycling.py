_base_ = '../yolact/yolact_r50_1x8_coco.py'

data_root = '/home/dg777/project/recycling/Data/'
dataset_type = 'CocoDataset'
classes = ('Can', 'Bottle', 'Milk Jug', 'Cardboard')

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(num_classes=4),
    mask_head=dict(num_classes=4),
    segm_head=dict(num_classes=4),
    test_cfg=dict(iou_thr=0.1))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomCrop', crop_size=(720, 1280)),
    dict(type='Resize', img_scale=(720, 1280), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        #scale_factor = (1.0),
        img_scale=(720, 1280),
        flip=False,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            #dict(type='Pad', size_divisor=32),
            #dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'dense_mix/',
        classes=classes,
        ann_file=data_root + 'annotations/recycling_v1.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'dense_mix/',
        classes=classes,
        ann_file=data_root + 'annotations/recycling_v1.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'dense_mix/',
        classes=classes,
        ann_file=data_root + 'annotations/recycling_v1.json',
        pipeline=test_pipeline))
workflow = [('train', 1), ('val',1)]
load_from = '/home/dg777/project/recycling/mmdetection/checkpoints/yolact_r101_1x8_coco_20200908-4cbe9101.pth'
