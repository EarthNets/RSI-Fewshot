# _base_ = [
#     '../_base_/models/faster_rcnn_r50_caffe_c4.py',
# ]
_base_ = ['../_base_/models/mask_rcnn_r50_fpn.py']
# model settings
model = dict(
    type='FSDetView',
    backbone=dict(type='ResNetWithMetaConv', frozen_stages=2),
    rpn_head=dict(
        feat_channels=512, loss_cls=dict(use_sigmoid=False, loss_weight=1.0)),
    roi_head=dict(
        type='FSDetViewRoIHead',
        # shared_head=dict(
        #     type='MetaRCNNResLayer',
        #     # pretrained=pretrained,
        #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        #     depth=50,
        #     stage=3,
        #     stride=2,
        #     dilation=1,
        #     style='pytorch',
        #     norm_cfg=dict(type='BN', requires_grad=False),
        #     norm_eval=True),
        bbox_head=dict(
            _delete_=True,
            type='MetaBBoxHead',
            with_avg_pool=False,
            in_channels=4096,
            roi_feat_size=1,
            num_classes=80,
            num_meta_classes=80,
            meta_cls_in_channels=2048,
            with_meta_cls_loss=True,
            loss_meta=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True),
                dict(
                    type='DifferenceAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True)
            ],
            init_cfg=[
                dict(
                    type='Normal',
                    layer=['Conv1d', 'Conv2d', 'Linear'],
                    mean=0.0,
                    std=0.001),
                dict(type='Normal', layer=['BatchNorm1d'], mean=1.0, std=0.02)
            ])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100)))
