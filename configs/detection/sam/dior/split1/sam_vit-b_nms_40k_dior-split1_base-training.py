_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_dior_vit.py',
    '../../../_base_/schedules/adamw_40k.py',
    # '../../../_base_/models/mask_rcnn_r50_fpn.py',
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

prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size
encoder_embed_dim = 768
encoder_depth = 12
encoder_num_heads = 12
encoder_global_attn_indexes = [2, 5, 8, 11]

# model = dict(
#     type='MaskRCNN',
#     backbone=dict(
#         depth=101,
#         # frozen_stages=1,
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
#     ),
#     roi_head=dict(
#         mask_roi_extractor=None,
#         mask_head=None,
#         bbox_head=dict(num_classes=15)))

model = dict(
    type='SAM',
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/checkpoints/sam_vit_b_01ec64.pth'),
    frozen_parameters=[
        'image_encoder',
        'prompt_encoder',
        'mask_decoder.transformer'
    ],
    image_encoder=dict(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        # norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    ),
    prompt_encoder=dict(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    ),
    mask_decoder=dict(
        num_multimask_outputs=3,
        transformer=dict(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_query=1024
    ),
    bbox_head=dict(
        type='SAMHead',
        num_classes=15,
        in_channels=2048,
        embed_dims=256,
        num_query=1024,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            )
        ),
        test_cfg=dict(
            score_thr=0.05,
            max_per_img=1024,
            nms=dict(type='nms', iou_threshold=0.7),
        )
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
    ),
    test_cfg=dict(max_per_img=1024)
)


expr_name = 'sam_vit-b_nms_40k_dior-split1_base-training'
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
