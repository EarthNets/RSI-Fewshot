_base_ = [
    '../../../../_base_/datasets/fine_tune_based/few_shot_dior.py',
    '../../../../_base_/schedules/adamw_10k.py',
    '../../../tfa_r101_fpn.py',
    '../../../../_base_/default_runtime.py'
]
seed = 0
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDIORDataset',
        # ann_cfg=[dict(method='TFA', setting='SPLIT1_3SHOT')],
        num_novel_shots=5,
        num_base_shots=5,
        classes='ALL_CLASSES_SPLIT1',
        save_dataset=True,
        save_dataset_path=f'work_dirs/data_infos/dior-split1_3shot_seed{seed}.json'
    ),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1')
)
model = dict(
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
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=1.0,
                neg_pos_ub=0,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
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
load_from = ('work_dirs/tfa_r101_fpn_dior_fine-tuning/base_model_random_init_bbox_head.pth')

expr_name = 'tfa_r101_fpn_dior-split1_seed0_3shot-fine-tuning'
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
        # dict(type='RSIDetWandbHook',
        #      init_kwargs=init_kwargs,
        #      interval=10,
        #      log_checkpoint=False,
        #      log_checkpoint_metadata=False,
        #      num_eval_images=30,
        #      bbox_score_thr=0.3,
        #      eval_after_run=True)
    ])
