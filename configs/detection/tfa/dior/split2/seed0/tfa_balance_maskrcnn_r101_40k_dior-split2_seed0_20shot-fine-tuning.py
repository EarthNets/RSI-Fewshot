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
        num_novel_shots=20,
        num_base_shots=None,
        # num_base_shots=3,
        classes='ALL_CLASSES_SPLIT2',
        save_dataset=True,
        save_dataset_path=f'work_dirs/data_infos/dior-split2_20shot_seed{seed}.json',
        balance_base_novel=True
    ),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2')
)
model = dict(
    type='TFA',
    backbone=dict(
        depth=101,
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
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
load_from = ('work_dirs/st_rpn_maskrcnn_r101_40k_dior-split2_randomized_head/base_model_random_init_bbox_head.pth')

expr_name = 'tfa_balance_maskrcnn_r101_40k_dior-split2_seed0_20shot_fine-tuning'
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
             num_eval_images=100,
             bbox_score_thr=0.0,
             eval_after_run=True)
    ])

evaluation = dict(
    class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'],
)
