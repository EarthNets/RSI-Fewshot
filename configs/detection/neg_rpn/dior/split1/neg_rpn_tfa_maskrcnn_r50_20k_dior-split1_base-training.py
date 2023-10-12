_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_dior.py',
    '../../../_base_/schedules/adamw_20k.py',
    '../../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))
# lr_config = dict(warmup_iters=100, step=[12000, 16000])
# runner = dict(max_iters=18000)
# model settings
model = dict(
    type='NegRPNTFA',
    rpn_head=dict(
        type='NegRPNHead'
    ),
    roi_head=dict(
        type='NegRPNRoIHead',
        mask_roi_extractor=None,
        mask_head=None,
        bbox_head=dict(num_classes=15)
    ),
    train_cfg=dict(
        rpn=dict(
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=1.0,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_pos=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=1.0,
                neg_pos_ub=0,
                add_gt_as_proposals=True),
            sampler_neg=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=1.0,
                neg_pos_ub=32,
                add_gt_as_proposals=False),
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            neg_filter_thre=0.3 # background boxes with scores higher than this threshold will be removed
        )
    )
)

# using regular sampler can get a better base model
use_infinite_sampler = False
