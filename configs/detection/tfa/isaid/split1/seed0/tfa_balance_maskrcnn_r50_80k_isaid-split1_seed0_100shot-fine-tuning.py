_base_ = [
    '../../../../_base_/datasets/fine_tune_based/few_shot_isaid.py',
    '../../../../_base_/schedules/adamw_10k.py',
    '../../../tfa_maskrcnn_r50.py',
    '../../../../_base_/default_runtime.py'
]
seed = 0
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotISAIDDataset',
        # ann_cfg=[dict(method='TFA', setting='SPLIT1_3SHOT')],
        num_novel_shots=100,
        # num_base_shots=100,
        num_base_shots=None,
        classes='ALL_CLASSES_SPLIT1',
        save_dataset=True,
        save_dataset_path=f'work_dirs/data_infos/isaid-split1_100shot_seed{seed}.json',
        balance_base_novel=True
    ),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1')
)
model = dict(
    type='TFA',
    backbone=dict(
        # depth=101,
        # frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=15
        ),
    ),
    frozen_parameters=[
        'backbone',
        # 'neck',
        # 'rpn_head.rpn_reg',
        'roi_head.bbox_head.shared_fcs',
        # 'rpn_head.rpn_conv.weight',
        # 'rpn_head.rpn_conv.bias',
        # 'rpn_head.rpn_cls.weight',
        # 'rpn_head.rpn_cls.bias',
        # 'rpn_head.rpn_reg.weight',
        # 'rpn_head.rpn_reg.bias',
        # 'roi_head.bbox_head', # freeze the teacher net
    ],
)

# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ('work_dirs/st_rpn_maskrcnn_r50_80k_isaid-split1_randomized_head/base_model_random_init_bbox_head.pth')

expr_name = 'tfa_balance_maskrcnn_r50_80k_isaid-split1_seed0_100shot_fine-tuning'
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
             interval=201,
             log_checkpoint=False,
             log_checkpoint_metadata=False,
             num_eval_images=200,
             # bbox_score_thr=0.0,
             # bbox_score_thr=[0.5] * 11 + [0.0] * 4,
             bbox_score_thr=[0.397, 0.625, 0.683, 0.699, 0.902, 0.542, 0.652, 0.526, 0.444, 0.231,
                             0.347, 0.098, 0.091, 0.065, 0.051],
             eval_after_run=True,
             without_mask=True)
    ])

evaluation = dict(
    interval=1000, metric=['bbox'], classwise=True,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'],
    # metric_items=['mAP_50'], iou_thrs=[0.5]
)