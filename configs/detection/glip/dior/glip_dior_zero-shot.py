_base_ = [
    '../../_base_/datasets/fine_tune_based/base_dior_bs4_1k.py',
    '../../_base_/schedules/adamw_40k.py',
    '../../_base_/default_runtime.py'
]

# classes splits are predefined in FewShotDIORDataset
data = dict(
    train=dict(classes='ALL_CLASSES_SPLIT0'),
    val=dict(classes='ALL_CLASSES_SPLIT0'),
    test=dict(classes='ALL_CLASSES_SPLIT0')
)

num_classes = 20 + 1
use_infinite_sampler = False

expr_name = 'glip_40k_dior_zero-shot'
init_kwargs = dict(
    project = 'rsi-fewshot',
    entity = 'tum-tanmlh',
    name = expr_name,
    resume = 'never',
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
wandb_cfg = dict(
    **init_kwargs,
    root = 'work_dirs',
    bbox_score_thr = [0.4] * num_classes,
    interval = 10,
    train_vis_thr = 0.0
)
model = dict(
    type='GLIPDetector',
    cfg=dict(
        model_file = '../RSGLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml',
        weight_file = '../RSGLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
    ),
    init_cfg=dict(),
    frozen_parameters=[
        'model.backbone',
        'model.language_backbone'
    ],
    class_names = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
             'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield',
             'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation',
             'vehicle', 'windmill'],
    wandb_cfg = wandb_cfg
)

