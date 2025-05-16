from __future__ import annotations
import torch.nn as nn
import ssl
import ssl
import torch
import numpy as np
import torchvision
from collections import OrderedDict
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Dict, List, Union
from ..base.base_model import BaseModel

"""
RoIHeads https://github.com/pytorch/vision/blob/367e8514c5d5e8528330081f25efeb17dd5e7ebd/torchvision/models/detection/roi_heads.py#L550-L560
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets)
        losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

        keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)

        loss_keypoint = {}
        if self.training:
            gt_keypoints = [t["keypoints"] for t in targets]
            loss_keypoint = keypointrcnn_loss(
                keypoint_logits, keypoint_proposals,
                gt_keypoints, pos_matched_idxs)
            loss_keypoint = dict(loss_keypoint=loss_keypoint)
            
            
RPN https://github.com/pytorch/vision/blob/a06df0d9229e49bd859e2ff0355a48b3bddd1e10/torchvision/models/detection/rpn.py#L298

    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss
"""


class KeypointRCNN(BaseModel):
    def __init__(self, num_keypoints, num_classes,  pretrained_backbone=True,
                 trainable_backbone_layers=None, all_layers_trainable=False,
                 anchor_generator_sizes = None, anchor_generator_aspect_ratios=None):
        super().__init__("Resnet Keypoint model")
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        assert(self.num_classes >= 1), "Number of classes must be greater than 1- first class is background"
        ssl._create_default_https_context = ssl._create_unverified_context()
        """if pretrained:
            assert(num_keypoints == 17), "If pretained=true, num_keypoints should be set to 17"
            assert(num_classes == 2), "If pretained=true, num_classes should be set to 2
            weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        """
        weights_backbone = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None

        anchor_generator = None
        if anchor_generator_sizes is not None and anchor_generator_aspect_ratios is not None:
            anchor_generator = AnchorGenerator(sizes=anchor_generator_sizes, aspect_ratios=anchor_generator_aspect_ratios)
        print(weights_backbone)
        print(anchor_generator)
        self.model = keypointrcnn_resnet50_fpn(weights_backbone=weights_backbone,
                                               num_classes=num_classes, num_keypoints=num_keypoints,
                                               trainable_backbone_layers=trainable_backbone_layers,
                                               rpn_anchor_generator=anchor_generator)
        if all_layers_trainable:
            for param in self.model.parameters():
                param.requires_grad = True
        ssl._create_default_https_context = ssl.create_default_context()

    def forward(self, x, target=None):
        """
        :param x:
        :param target: target must be set when in training mode.
            boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format.
            labels (Int64Tensor[N]): the label for each bounding box. 0 always represents the background class.
            image_id (Int64Tensor[1]): an image identifier.
            area (Tensor[N]): the area of the bounding box.
            iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
            keypoints (FloatTensor[N, K, 3])
        :return:
        """
        
        if self.training:
            assert(target is not None), "When in training mode, target must also be passed"

        # in eval mode: 'boxes', 'labels', 'scores', 'keypoints', 'keypoints_scores'
        if self.training:
            output = self.model(x, target)

            return output
        else:
            return self.model(x)

    def process_predictions(self, output: Union[List[Dict[str, torch.FloatTensor]], Dict[str, torch.FloatTensor]],
                            score_boundary=0.7, nms_boundary=0.3):
        if isinstance(output, dict):
            scores = output['scores'].detach().cpu().numpy()
            if score_boundary is None:
                high_scores_idxs = np.where(scores > score_boundary)[0].tolist()
            else:
                high_scores_idxs = range(len(output['boxes']))

            if nms_boundary is not None:
                post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs],
                                                    output['scores'][high_scores_idxs],
                                                    nms_boundary).cpu().numpy()
            else:
                post_nms_idxs = range(len(output['boxes'][high_scores_idxs]))

            keypoints = []
            for kps in output['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append(kps)

            bboxes = []
            for bbox in output['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(bbox)

            classes = []
            for label in output['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                classes.append(label)
            return {"keypoints": keypoints, "boxes": bboxes, "classes": classes}
        elif isinstance(output, list):
            return [self.process_predictions(v) for v in output]
        else:
            raise Exception("Unknown type in process_predictions()")