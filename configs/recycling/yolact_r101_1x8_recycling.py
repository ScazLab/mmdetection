_base_ = '../yolact/yolact_r50_1x8_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(num_classes=1),
    mask_head=dict(num_classes=1),
    segm_head=dict(num_classes=1))
