python tools/detection/test.py configs/detection/st_tfa/isaid/split1/seed0/st_rpn_v6_no-roi/st_rpn_v6_no-roi_tfa_maskrcnn_r50_isaid-split1_seed0_10shot-fine-tuning.py work_dirs/st_rpn_v6_no-roi_tfa_maskrcnn_r50_isaid-split1_seed0_10shot-fine-tuning/iter_3000.pth --eval='bbox'
python tools/detection/test.py configs/detection/st_tfa/isaid/split1/seed0/st_rpn_v6_no-roi/st_rpn_v6_no-roi_tfa_maskrcnn_r50_isaid-split1_seed0_50shot-fine-tuning.py work_dirs/st_rpn_v6_no-roi_tfa_maskrcnn_r50_isaid-split1_seed0_50shot-fine-tuning/iter_5000.pth --eval='bbox'
python tools/detection/test.py configs/detection/st_tfa/isaid/split1/seed0/st_rpn_v6_no-roi/st_rpn_v6_no-roi_tfa_maskrcnn_r50_isaid-split1_seed0_100shot-fine-tuning.py work_dirs/st_rpn_v6_no-roi_tfa_maskrcnn_r50_isaid-split1_seed0_100shot-fine-tuning/iter_5000.pth --eval='bbox'