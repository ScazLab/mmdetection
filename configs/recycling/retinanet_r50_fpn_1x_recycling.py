_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# num_classes
model = dict(bbox_head = dict(num_classes=2),
             test_cfg = dict(min_bbox_size=40,
                             nms=dict(type='fast-nms', iou_threshold=0.5)))
