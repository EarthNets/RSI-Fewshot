# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
import copy
import re
import random

from typing import Any, Dict, List, Tuple

from rsidet.models.builder import DETECTORS, build_head
from rsidet.models.detectors import BaseDetector
from rsidet.core import bbox2result
from maskrcnn_benchmark.modeling.detector.generalized_vl_rcnn import GeneralizedVLRCNN
from maskrcnn_benchmark.config import cfg as glip_cfg
from maskrcnn_benchmark.engine.inference import create_queries_and_maps_from_dataset
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.od_to_grounding import convert_od_to_grounding_simple, check_for_positive_overflow, sanity_check_target_after_processing, convert_object_detection_to_grounding_optimized_for_od

@DETECTORS.register_module()
class GLIPDetector(BaseDetector):
    def __init__(
        self,
        cfg=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        class_names=None,
        wandb_cfg=None
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super(GLIPDetector, self).__init__(init_cfg)

        glip_cfg.merge_from_file(cfg['model_file'])
        glip_cfg.merge_from_list(["MODEL.WEIGHT", cfg['weight_file']])
        self.glip_cfg = glip_cfg
        self.wandb_cfg = wandb_cfg
        self.class_names=class_names
        self.class_names2id = {name:i for i, name in enumerate(class_names)}
        glip_cfg.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM = len(class_names) + 1
        glip_cfg.MODEL.DYHEAD.NUM_CLASSES = len(class_names) + 1
        glip_cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT = False
        glip_cfg.MODEL.DYHEAD.USE_CHECKPOINT = False
        glip_cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX = False
        # glip_cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000

        self.model = GeneralizedVLRCNN(glip_cfg)

    def init_weights(self):
        self._params_init_info = defaultdict(dict)
        for name, param in self.named_parameters():
            self._params_init_info[param]['init_info'] = f'initialized by weights from {glip_cfg.MODEL.WEIGHT}'
            self._params_init_info[param]['tmp_mean_value'] = param.data.mean()

        super(GLIPDetector, self).init_weights()
        checkpointer = DetectronCheckpointer(glip_cfg, self.model, save_dir='')
        _ = checkpointer.load(glip_cfg.MODEL.WEIGHT)


    def simple_test(self, imgs, img_metas, rescale=False):

        # captions = [all_queries[query_i] for ii in range(len(targets))]
        class FakeDataset:
            def __init__(self, names):
                # self.names = names
                self.names = {idx: name for idx, name in enumerate(names)}
            def categories(self):
                return self.names

        # fake_dataset = FakeDataset(self.class_names)
        # all_queries, all_positive_map_label_to_token = create_queries_and_maps_from_dataset(fake_dataset, self.glip_cfg)
        # query_time = len(all_queries)
        # num_classes = len(self.class_names)

        bbox_results = []


        caption, label_to_positions = self.create_caption_from_class_names(self.class_names)
        captions = [caption for i in range(len(imgs))]

        tokenized = self.tokenize(captions, imgs.device)
        # positive_map, tokenized = self.create_positive_map(captions, gt_labels, label_to_positions)
        label2tokens = self.create_label2tokens_map(tokenized, label_to_positions)
        lang_dict = self.create_language_dict_features(tokenized)

        # positive_map_label_to_token = all_positive_map_label_to_token[query_i]
        # captions = [all_queries[query_i] for ii in range(len(imgs))]
        # positive_map_label_to_token = all_positive_map_label_to_token[query_i]
        # imgs = imgs[:, [2, 1, 0]]
        output = self.model(imgs, captions=captions, positive_map=label2tokens, language_dict_features=lang_dict)

        for i in range(len(output)):
            labels = output[i].get_field('labels')
            scores = output[i].get_field('scores')
            bbox = output[i].convert('xyxy').bbox

            bbox_scores = torch.cat([bbox, scores.view(-1, 1)], dim=1)
            # labels[labels==num_classes] = 0

            output[i].add_field('labels', labels)
            output[i].bbox = bbox_scores

            bbox_results.append(
                bbox2result(bbox_scores, labels, len(self.class_names))
            )


            # bbox_results = [
            #     bbox2result(output[i].bbox, output[i].get_field('labels'), len(self.class_names))
            #     for i in range(len(output))
            # ]

        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise ValueError('aug_test function is not implemented in this model.')

    def extract_feat(self, img):
        raise ValueError('extract_feat function is not implemented in this model.')


    def forward_train(self,
                      imgs,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`rsidet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """


        targets = []
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            target = BoxList(gt_bbox, imgs.shape[2:], mode='xyxy')
            # if len(gt_label.shape) == 1:
            #     gt_label = F.one_hot(gt_label, len(self.class_names))
            target.add_field('labels', gt_label)
            targets.append(target)

        img_list = [img for img in imgs]

        device = imgs.device

        caption, label_to_positions = self.create_caption_from_class_names(self.class_names)
        captions = [caption for i in range(len(imgs))]

        tokenized = self.tokenize(captions, device)
        positive_map = self.create_positive_map(tokenized, gt_labels, label_to_positions)
        lang_dict = self.create_language_dict_features(tokenized)

        # loss_dict = self.model(imgs, targets=targets, captions=captions, positive_map=positive_map_label_to_token)
        loss_dict = self.model(img_list, targets=targets, positive_map=positive_map, language_dict_features=lang_dict)

        proposals = loss_dict.pop('proposals') if 'proposals' in loss_dict else None
        if proposals is not None:
            bbox_results = []
            for proposal in proposals:
                labels = proposal.get_field('labels')
                scores = proposal.get_field('scores')
                bbox = proposal.convert('xyxy').bbox
                # labels[labels==len(captions)] = 0
                bbox_scores = torch.cat([bbox, scores.view(-1, 1)], dim=1)

                inds = scores > self.wandb_cfg['train_vis_thr']

                bbox_results.append(
                    bbox2result(bbox_scores[inds], labels[inds], len(self.class_names))
                )
            loss_dict['states'] = {
                'else|img': imgs,
                'else|bbox_results': bbox_results
            }

        return loss_dict


    def create_caption_from_class_names(
        self,
        ori_class_names,
        separation_tokens=". ",
    ):
        """
        Convert object detection data into grounding data format, on the fly.
        """
        class_names = copy.deepcopy(ori_class_names)

        def clean_name(name):
            name = re.sub(r"\(.*\)", "", name)
            name = re.sub(r"_", " ", name)
            name = re.sub(r"  ", " ", name)
            return name

        label_to_positions = {}

        pheso_caption = ""

        for index, class_name in enumerate(class_names):

            start_index = len(pheso_caption)
            pheso_caption += clean_name(class_name)  # NOTE: slight change...
            end_index = len(pheso_caption)
            label_to_positions[self.class_names2id[class_name]] = [start_index, end_index]

            if index != len(class_names) - 1:
                pheso_caption += separation_tokens

        return pheso_caption, label_to_positions

    def tokenize(self, captions, device):
        tokenized = self.model.tokenizer.batch_encode_plus(
            captions,
            max_length=self.glip_cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
            padding='max_length' if self.glip_cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True
        ).to(device)
        return tokenized

    def create_label2tokens_map(self, tokenized, label_to_positions):
        # 1. Find out which chars are assigned to each label
        tokens_positive = []
        for label in range(len(self.class_names)):
            assert label in label_to_positions
            tokens_positive.append(label_to_positions[label])

        label2tokens = {}

        # 2. For each label, map the corresponding char position to token position
        for j, tok_list in enumerate(tokens_positive):
            beg, end = tok_list
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            # positive_map[j, beg_pos: end_pos + 1].fill_(1)
            label2tokens[j] = list(range(beg_pos, end_pos + 1))

        return label2tokens

    def create_positive_map(self, tokenized, gt_labels, label_to_positions):

        device = gt_labels[0].device

        # 1. Find out which chars are assigned to each gt_labels
        tokens_positive = []
        for labels_per_img in gt_labels:
            for gt_label in labels_per_img:
                assert gt_label.item() in label_to_positions
                tokens_positive.append(label_to_positions[gt_label.item()])


        """ construct a map such that positive_map[i,j] = True iff box i is associated to token j """
        positive_map = torch.zeros(
            (len(tokens_positive), tokenized['input_ids'].shape[1]),
            dtype=torch.float
        ).to(device)

        # 2. For each gt_label, map the corresponding char position to token position
        for j, tok_list in enumerate(tokens_positive):
            beg, end = tok_list
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)

        return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

    def create_language_dict_features(self, tokenized):

        input_ids = tokenized.input_ids
        mlm_labels = None

        tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}

        language_dict_features = self.model.language_backbone(tokenizer_input)

        # ONE HOT
        if self.glip_cfg.DATASETS.ONE_HOT:
            new_masks = torch.zeros_like(language_dict_features['masks'],
                                         device=language_dict_features['masks'].device)
            new_masks[:, :self.cfg.MODEL.DYHEAD.NUM_CLASSES] = 1
            language_dict_features['masks'] = new_masks

        # MASK ALL SPECIAL TOKENS
        if self.glip_cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL:
            language_dict_features["masks"] = 1 - tokenized.special_tokens_mask

        language_dict_features["mlm_labels"] = mlm_labels

        return language_dict_features


