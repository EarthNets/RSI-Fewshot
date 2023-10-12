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
# evaluation = dict(
#     interval=100, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
# checkpoint_config = dict(interval=100)
# optimizer = dict(lr=0.005)
# lr_config = dict(warmup=None)
# runner = dict(max_iters=500)
# load_from = 'path of base training model'
load_from = 'work_dirs/fsdetview_r101_c4_20k_dior-split1_base-training/latest.pth'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
])
