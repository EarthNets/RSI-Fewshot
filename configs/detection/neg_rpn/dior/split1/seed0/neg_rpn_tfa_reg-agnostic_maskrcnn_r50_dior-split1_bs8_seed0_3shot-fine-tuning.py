_base_ = [
    '../../../../_base_/datasets/fine_tune_based/few_shot_dior_bs8.py',
    '../../../../_base_/schedules/adamw_10k.py',
    '../../../tfa_maskrcnn_r50.py',
    '../../../../_base_/default_runtime.py'
]
seed = 0
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDIORDataset',
        # ann_cfg=[dict(method='TFA', setting='SPLIT1_3SHOT')],
        num_novel_shots=3,
        num_base_shots=3,
        classes='ALL_CLASSES_SPLIT1',
        save_dataset=True,
        save_dataset_path=f'work_dirs/data_infos/dior-split1_3shot_seed{seed}.json'
    ),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1')
)
model = dict(
    type='NegRPNTFA',
    frozen_parameters=[
        'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs',
        'roi_head.bbox_head.fc_reg'
    ],
    rpn_head=dict(
        type='NegRPNHead'
    ),
    roi_head=dict(
        type='NegRPNRoIHead',
        bbox_head=dict(reg_class_agnostic=True)
    ),
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
                neg_pos_ub=16,
                add_gt_as_proposals=False),
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            neg_filter_thre=0.4 # background boxes with scores higher than this threshold will be removed
        )
    )
)



evaluation = dict(
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1']
)
# checkpoint_config = dict(interval=12000)
# optimizer = dict(lr=0.001)
# lr_config = dict(
#     warmup_iters=10, step=[
#         11000,
#     ])
# runner = dict(max_iters=12000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ('work_dirs/tfa_reg-agnostic_maskrcnn_r50_20k_dior-split1_randomized_head/base_model_random_init_bbox_head.pth')

expr_name = 'neg_rpn_tfa_reg-agnostic_maskrcnn_r50_dior-split1_bs8_seed0_3shot_fine-tuning'
init_kwargs = {
    'project': 'rsi-fewshot',
    'entity': 'tum-tanmlh',
    'name': expr_name,
    'resume': 'never'
}
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='RSIDetWandbHook',
             init_kwargs=init_kwargs,
             interval=10,
             log_checkpoint=False,
             log_checkpoint_metadata=False,
             num_eval_images=30,
             bbox_score_thr=0.0,
             eval_after_run=True)
    ])
