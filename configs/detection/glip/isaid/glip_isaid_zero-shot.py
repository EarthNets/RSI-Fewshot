_base_ = [
    '../../_base_/datasets/fine_tune_based/base_isaid_1k.py',
    '../../_base_/schedules/adamw_40k.py',
    '../../_base_/default_runtime.py'
]

# classes splits are predefined in FewShotDIORDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    # val=dict(classes='BASE_CLASSES_SPLIT1'),
    # test=dict(classes='BASE_CLASSES_SPLIT1')
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1')
)

num_classes = 15
use_infinite_sampler = False

expr_name = 'glip_40k_isaid_zero-shot'
init_kwargs = dict(
    project = 'rsi-fewshot',
    entity = 'tum-tanmlh',
    name = expr_name,
    resume = 'never',
    root = 'work_dirs',
    bbox_score_thr = [0.4] * num_classes,
    interval = 10
)
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
             bbox_score_thr=[0.4] * num_classes,
             eval_after_run=True)
    ])

model = dict(
    type='GLIPDetector',
    cfg=dict(
        model_file = '../RSGLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml',
        weight_file = '../RSGLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
    ),
    init_cfg=dict(),
    frozen_parameters=[
    ],
    class_names = [
        'Small_Vehicle', 'storage_tank', 'Swimming_pool', 'Harbor',
        'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field',
        'baseball_diamond', 'Bridge', 'basketball_court', 'Roundabout',
        'Helicopter', 'ship', 'plane', 'Large_Vehicle'
    ],
    wandb_cfg = init_kwargs
)

