import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# mmseg
import numpy as np
from mmengine.fileio import load
import functools

# ddp
import torch.distributed as dist

# from DMINet
def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def cross_entropy_ddp(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')

    input = input.to(device)
    target = target.to(device)

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def cross_entropy_ddp_01(input, target, weight=None, reduction='mean',ignore_index=255, args=None):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """

    rank = dist.get_rank()
    device = torch.device(f'cuda:{args.gpu_ids[rank]}')

    input = input.to(device)
    target = target.to(device)

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
# from light ################################################################
def binarization(A):
    return (A > 0.5).float()
def n3c_loss(V1, V2, GT, alpha_coef=1.0):
    L_nc = 0
    L_mask = 0
    L = len(V1)
    
    epsilon = 1e-8  
    GTf = GT.float()

    for i in range(L):
        V1_i = V1[i]
        V2_i = V2[i]
        
        assert V1_i.shape == V2_i.shape, 'tensor shape cannot match'
        B, C, H, W = V2_i.shape

        # v1 = V1_i.contiguous().view(B, -1, C)
        # v2 = V2_i.contiguous().view(B, -1, C)
        v1 = V1_i.permute(0, 2, 3, 1).view(B, -1, C)
        v2 = V2_i.permute(0, 2, 3, 1).view(B, -1, C)
        
        
        sim_i = F.cosine_similarity(v1, v2, -1, epsilon).unsqueeze(dim=-1)
        dif_i = (torch.full(sim_i.shape, 1.).to(sim_i.device) - sim_i).permute(0, 2, 1).view(B, 1, H, W)
        
        sim_i[sim_i < 0.5] = 0
        
        sim_i_sum = torch.sum(sim_i >= 0.5) #nochange numbers
        if sim_i_sum == 0:
            alpha = 0
        else:
            alpha = H * W / sim_i_sum
        
        L_nc += alpha * F.l1_loss(v1 * sim_i, v2 * sim_i)
        
        dif_up = F.interpolate(dif_i, size=(256, 256), mode='bilinear', align_corners=False)
        L_mask += F.mse_loss(dif_up, GTf)
        #L_mask += F.l1_loss(dif_up, GT)
    
    
    L_mask *= alpha_coef
    return L_nc, L_mask
def dice_loss(input, target, weight=None, ignore_index=255):
    """
    dice loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.float()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    smooth = 1.
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    if weight is None:
        weight = torch.ones_like(input_flat)
    intersect = torch.dot(input_flat, target_flat)
    union = torch.sum(input_flat) + torch.sum(target_flat)
    loss = (2 * intersect + smooth) / (union + smooth)
    return 1 - loss


class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwards):
        super().__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = target.squeeze(1) # 
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss_mmseg(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss
def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = load(class_weight)

    return class_weight
