_base_ = '../yolact/yolact_r50_1x8_coco.py'

data_root = '/home/debasmita/catkin_ws/src/hide_and_seek/hr_hide_seek/scripts/robot_detection/unity_heading_detection_data/'
dataset_type = 'CocoDataset'
classes = ('0', '45', '90', '135', '180', '225', '270', '315')

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(num_classes=8 #, 
                # anchor_generator=dict(
                #     type='AnchorGenerator',
                #     octave_base_scale=3,
                #     scales_per_octave=1,
                #     #base_sizes=[8, 16, 32, 64, 128],
                #     base_sizes=[4, 12, 22, 44, 86], # divide by 720/280 = 1.5
                #     ratios=[0.5, 1.0, 2.0],
                #     strides=[550.0 / x for x in [69, 35, 18, 9, 5]],
                #     centers=[(550 * 0.5 / x, 550 * 0.5 / x)
                #             for x in [69, 35, 18, 9, 5]])
            ),
    mask_head=dict(num_classes=8),
    segm_head=dict(num_classes=8),
    test_cfg=dict(iou_thr=0.4))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomCrop', crop_size=(480, 640)),
    dict(type='Resize', img_scale=(480, 640), keep_ratio=True), 
    dict(type='RandomFlip', flip_ratio=0.5), # do not comment, YolACT forces you to flip!
    # dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        # scale_factor = (1.0),
        img_scale=(480, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='Pad', size_divisor=32),
            # dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, # batch size
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'all_images',
        classes=classes,
        ann_file=data_root + 'unity_robot_detection_coco_json.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'all_images',
        classes=classes,
        ann_file=data_root + 'unity_robot_detection_coco_json.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'all_images',
        classes=classes,
        ann_file=data_root + 'unity_robot_detection_coco_json.json',
        pipeline=test_pipeline))
workflow = [('train', 1), ('val',1)]
load_from = '/home/debasmita/catkin_ws/src/mmdetection/checkpoints/yolact_r101_1x8_coco_20200908-4cbe9101.pth'
work_dir = '/home/debasmita/catkin_ws/src/hide_and_seek/hr_hide_seek/models/robot_detection_unity/'
