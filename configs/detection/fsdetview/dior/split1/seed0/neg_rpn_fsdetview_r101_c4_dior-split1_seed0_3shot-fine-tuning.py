_base_ = [
    '../../../../_base_/datasets/nway_kshot/few_shot_dior.py',
    '../../../../_base_/schedules/adamw_10k.py',
    '../../../fsdetview_r101_c4.py',
    '../../../../_base_/default_runtime.py'
]
seed = 0
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        save_dataset_path=f'work_dirs/data_infos/fsdet_dior-split1_3shot_seed{seed}.json',
        dataset=dict(
            type='FewShotDIORDataset',
            # type='FewShotDIORDefaultDataset',
            # ann_cfg=[dict(method='FSDetView', setting='SPLIT1_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1']
)
load_from = 'work_dirs/fsdetview_r101_c4_20k_dior-split1_base-training/latest.pth'

# model settings
model = dict(
    type='NegRPNMetaRCNN',
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ],
    rpn_head=dict(
        type='NegRPNHead'
    ),
    roi_head=dict(
        type='NegRPNFSDetViewRoIHead'
    ),
    train_cfg=dict(
        rcnn=dict(
            sampler_pos=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=1.0,
                neg_pos_ub=0,
                add_gt_as_proposals=True),
            sampler_neg=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=0.0,
                neg_pos_ub=16,
                add_gt_as_proposals=False),
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            neg_filter_thre=0.3 # background boxes with scores higher than this threshold will be removed
        )
    )
)

expr_name = 'neg_rpn_fsdetview_r101_c4_dior-split1_seed0_3shot-fine-tuning'
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
             bbox_score_thr=0.3,
             eval_after_run=True)
    ])
