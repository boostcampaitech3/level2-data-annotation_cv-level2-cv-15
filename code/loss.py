import torch
import torch.nn as nn


def get_dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)


def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map


class EASTLoss(nn.Module):
    def __init__(self, weight_angle=10):
        super().__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, roi_mask):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        classify_loss = get_dice_loss(gt_score, pred_score * roi_mask)
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        angle_loss *= self.weight_angle
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        geo_loss = angle_loss + iou_loss
        total_loss = classify_loss + geo_loss

        return total_loss, dict(cls_loss=classify_loss.item(), angle_loss=angle_loss.item(),
                                iou_loss=iou_loss.item())
