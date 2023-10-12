_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_dior_vit.py',
    '../../../_base_/schedules/adamw_40k_1e-5.py',
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
        test_cfg=dict(max_per_img=1024),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=1e-4,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=1024))
