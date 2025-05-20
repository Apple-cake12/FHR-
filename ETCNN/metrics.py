import numpy as np
import torch
from torch import nn
from signal_processing import *

class Dice(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Initialize the DiceLoss class.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Calculate the Dice loss.

        Args:
            preds (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, signal_length).
            targets (torch.Tensor): Ground truth tensor of shape (batch_size, signal_length) with class labels.

        Returns:
            torch.Tensor: Dice loss.
        """
        num_classes = 3
        # Apply softmax to ensure predictions are in the probability space if logits are provided

        # Initialize dice score list
        dice_scores = []

        # Calculate Dice coefficient for each class
        for class_idx in range(num_classes):
            # Create binary masks for predictions and targets
            pred_class = (preds == class_idx).float()  # shape: (batch_size, signal_length)
            target_class = (targets == class_idx).float()  # shape: (batch_size, signal_length)

            # Calculate intersection and union
            intersection = torch.sum(pred_class * target_class, dim=1)  # sum over signal_length
            union = torch.sum(pred_class, dim=1) + torch.sum(target_class, dim=1)

            # Dice score for current class
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)

        # Stack and average scores
        dice_scores = torch.stack(dice_scores, dim=1)  # shape: (batch_size, num_classes)
        dice = dice_scores.mean()  # average over classes and batches

        return dice
class Dice_Acc(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Dice_Acc, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        # Apply softmax to ensure predictions are in the probability space if logits are provided
        #加速标签为1
        class_idx = 1
        # Create binary masks for predictions and targets
        pred_class = (preds == class_idx).float()  # shape: (batch_size, signal_length)
        target_class = (targets == class_idx).float()  # shape: (batch_size, signal_length)

        # Calculate intersection and union
        intersection = torch.sum(pred_class * target_class, dim=1)  # sum over signal_length
        union = torch.sum(pred_class, dim=1) + torch.sum(target_class, dim=1)

        # Dice score for current class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)


        return dice_score


class Dice_Dec(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Dice_Dec, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply softmax to ensure predictions are in the probability space if logits are provided
        # 减速标签为2
        class_idx = 2
        # Create binary masks for predictions and targets
        pred_class = (preds == class_idx).float()  # shape: (batch_size, signal_length)
        target_class = (targets == class_idx).float()  # shape: (batch_size, signal_length)

        # Calculate intersection and union
        intersection = torch.sum(pred_class * target_class, dim=1)  # sum over signal_length
        union = torch.sum(pred_class, dim=1) + torch.sum(target_class, dim=1)

        # Dice score for current class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return dice_score


class IOU(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IOU, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # preds: (B, C, L) 未经argmax的输出
        # labels: (B, L) 类别索引
        num_classes = 3
        labels = labels.long()

        iou_per_class = []
        for c in range(num_classes):
            pred_c = (preds == c)
            label_c = (labels == c)

            intersection = (pred_c & label_c).sum().float()
            union = (pred_c | label_c).sum().float()

            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_per_class.append(iou)

        miou = torch.mean(torch.stack(iou_per_class))
        return miou

class IOU_Acc(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IOU_Acc, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # preds: (B, C, L) 未经argmax的输出
        # labels: (B, L) 类别索引
        labels = labels.long()
        class_id = 1
        pred_c = (preds == class_id)
        label_c = (labels == class_id)

        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou

class IOU_Dec(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IOU_Dec, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # preds: (B, C, L) 未经argmax的输出
        # labels: (B, L) 类别索引
        labels = labels.long()
        class_id = 2
        pred_c = (preds == class_id)
        label_c = (labels == class_id)

        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou



def interpolate_fhr(fhr_signal):
    """
    Interpolate missing values (zeros) in the FHR signal.

    Parameters:
        fhr_signal (array-like): The input FHR signal.

    Returns:
        array-like: The interpolated FHR signal.
    """
    fhr = fhr_signal.copy()

    # Flatten the signal if it's multidimensional
    if fhr.ndim > 1:
        fhr = fhr.flatten()

    # Find the indices of valid (non-zero) values
    valid_indices = np.where(fhr > 0)[0]

    if valid_indices.size > 0:
        # Set initial zeros to the first valid value
        first_valid = valid_indices[0]
        fhr[:first_valid] = fhr[first_valid]

        # Interpolate over zero regions
        idx = first_valid
        while idx is not None and idx < len(fhr):#idx存在非零值 且 idx不是最后一位
            zero_start = np.where(fhr[idx:] == 0)[0]
            if zero_start.size == 0:
                break
            zero_start = zero_start[0] + idx #第一个0的位置 索引

            next_valid = np.where(fhr[zero_start:] > 0)[0]
            if next_valid.size == 0:
                break
            next_valid = next_valid[0] + zero_start #第一个有效值的位置

            # Linear interpolation for zero region (Fix: Remove +1)
            fhr[zero_start-1:next_valid+1] = np.linspace(fhr[zero_start - 1], fhr[next_valid], next_valid - zero_start+2)


            idx = next_valid

        # Set trailing zeros to the last valid value
        last_valid = valid_indices[-1]
        fhr[last_valid:] = fhr[last_valid]

    return fhr
def region_AD(label):
    # 查找加速事件的起始点
    Acc_list=[]
    Dec_list=[]
    #防止边界效应
    binsig = np.zeros(len(label)+2)
    binsig[1:-1] = label
    acc_starts = np.where((binsig[:-1] != 1) & (binsig[1:] == 1))[0]
    acc_end = np.where((binsig[:-1] == 1) & (binsig[1:] != 1))[0] - 1
    Acc_list.append(acc_starts)
    Acc_list.append(acc_end)



    # 查找减速事件的起始点
    dec_starts = np.where((binsig[:-1] != 2) & (binsig[1:] == 2))[0]
    dec_end = np.where((binsig[:-1] == 2) & (binsig[1:] != 2))[0] - 1
    Dec_list.append(dec_starts)
    Dec_list.append(dec_end)

    return np.transpose(Acc_list)/240, np.transpose(Dec_list)/240
def mstatscompare_P(fhr,ground_labels,pred_labels,ground_baseline):
    FHR = fhr.detach().cpu().squeeze().numpy().copy() * 255
    FHR = apply_butterworth_filter(FHR,sampling_rate=4,cutoff_frequency=0.3,filter_order=6)
    ground_baselines = ground_baseline.detach().cpu().squeeze().numpy().copy()
    ground_cls = ground_labels.detach().cpu().squeeze().numpy().copy()
    pred_cls = pred_labels.detach().cpu().squeeze().numpy().copy()  # (32,4800)  # Predicted labels

    Index_Dice = Dice()
    Index_Dice_Acc = Dice_Acc()
    Index_Dice_Dec = Dice_Dec()

    Index_Iou = IOU()
    Index_Iou_Acc = IOU_Acc()
    Index_Iou_Dec = IOU_Dec()
    stats = {}
    # Dice指标
    stats['Dice'] = Index_Dice(pred_labels, ground_labels).cpu().numpy()
    stats['Dice_Acc'] = Index_Dice_Acc(pred_labels, ground_labels).cpu().numpy()
    stats['Dice_Dec'] = Index_Dice_Dec(pred_labels, ground_labels).cpu().numpy()

    # Iou指标
    stats['Iou'] = Index_Iou(pred_labels, ground_labels).cpu().numpy()
    stats['Iou_Acc'] = Index_Iou_Acc(pred_labels, ground_labels).cpu().numpy()
    stats['Iou_Dec'] = Index_Iou_Dec(pred_labels, ground_labels).cpu().numpy()
    #Accuracy
    correct = (pred_cls == ground_cls).sum()
    stats['Accuracy'] = (correct / len(pred_cls))

    pred_baselines = get_baseline_P(FHR,pred_cls,4800,240,480)
    # MAE指标
    stats['MAE'] = np.mean(np.abs(pred_baselines - ground_baselines))

    # 遍历样本 查找加减速事件
    Acc_ground_region, Dec_ground_region = region_AD(ground_cls)
    Acc_pred_region, Dec_pred_region = region_AD(pred_cls)
    ground_region = []
    ground_region.append(Acc_ground_region)
    ground_region.append(Dec_ground_region)
    pred_region = []
    pred_region.append(Acc_pred_region)
    pred_region.append(Dec_pred_region)


    LDB1 = ground_baselines.copy()  # baseline
    LDB2 = pred_baselines.copy()  # prebaseline
    acc1 = ground_region.copy()  #
    acc2 = pred_region.copy()  #


    FHR = FHR.flatten()
    LDB1 = LDB1.flatten()
    LDB2 = LDB2.flatten()
    stats['MADI'] = evlMADI(FHR, LDB1, LDB2)
    invaidIndex = np.where(FHR != 0)[0]
    if invaidIndex is None:
        return stats
    L1 = LDB1[invaidIndex]
    L2 = LDB2[invaidIndex]
    acc1[0] = filteraccident(acc1[0], FHR)  # 过滤掉在acc1候选者中缺失的信号数据
    acc1[1] = filteraccident(acc1[1], FHR)
    acc2[0] = filteraccident(acc2[0], FHR)
    acc2[1] = filteraccident(acc2[1], FHR)

    # 基线差 RMSD
    stats['RMSD_bpm'] = np.sqrt(np.mean(np.square(L1 - L2)))
    # 超过15bpm的百分比
    stats['Diff_Over_15_bpm_prct'] = np.mean(abs(L1 - L2) > 15) * 100
    # 计算一致性指标
    stats['Index_Agreement'] = 1 - np.sum(np.square(L1 - L2)) \
                               / np.sum(np.square(np.abs(L1 - np.mean(L1)) + np.abs(L2 - np.mean(L2))))
    [T, trueMatch, Qd] = accMatch(acc1[1], acc2[1], np.zeros((0, 2)))  # realDec[:,0:2], preDec[:,0:2]
    [T, trueMatch, Qa] = accMatch(acc1[0], acc2[0], np.zeros((0, 2)))


    # synthetic baseline inconsistency index
    FHRi = interpolate_fhr(FHR)
    cFHR1 = np.cumsum(FHRi - LDB1) / 240  # min
    cFHR2 = np.cumsum(FHRi - LDB2) / 240  # min
    AccArea1 = cFHR1[np.round(acc1[0][:, 1] * 4 * 60).astype(int)] - cFHR1[
        np.round(acc1[0][:, 0] * 4 * 60 ).astype(int)]
    AccArea1 = np.concatenate((AccArea1, [0]), axis=0)

    AccArea2 = cFHR2[np.round(acc2[0][:, 1] * 4 * 60).astype(int)] - cFHR2[
        np.round(acc2[0][:, 0] * 4 * 60 ).astype(int)]
    AccArea2 = np.concatenate((AccArea2, [0]), axis=0)

    DecArea1 = -cFHR1[np.round(acc1[1][:, 1] * 4 * 60).astype(int)] + cFHR1[
        np.round(acc1[1][:, 0] * 4 * 60 ).astype(int)]
    DecArea1 = np.concatenate((DecArea1, [0]), axis=0)

    DecArea2 = -cFHR2[np.round(acc2[1][:, 1] * 4 * 60).astype(int)] + cFHR2[
        np.round(acc2[1][:, 0] * 4 * 60 ).astype(int)]
    DecArea2 = np.concatenate((DecArea2, [0]), axis=0)

    d = 0
    dmax = 0
    dc = 0
    dmaxc = 0
    if Qa is not None:
        for i in range(np.size(Qa, 0)):
            for j in range(np.size(Qa, 1)):
                if Qa[i, j] == 1:
                    d += np.square(AccArea1[i] - AccArea2[j])
                    dmax += np.square(np.max([AccArea1[i], AccArea2[j]]))
                    dc += np.abs(AccArea1[i] - AccArea2[j])
                    dmaxc += np.max([AccArea1[i], AccArea2[j]])
    if dmax == 0:
        stats['ASI_prct'] = 0
    else:
        stats['ASI_prct'] = np.sqrt(d) / np.sqrt(dmax) * 100
    d = 0
    dmax = 0
    if Qd is not None:
        for i in range(np.size(Qd, 0)):
            for j in range(np.size(Qd, 1)):
                if Qd[i, j] == 1:
                    d += np.square(DecArea1[i] - DecArea2[j])
                    dmax += np.square(np.max([DecArea1[i], DecArea2[j]]))
                    dc += np.abs(DecArea1[i] - DecArea2[j])
                    dmaxc += np.max([DecArea1[i], DecArea2[j]])
    if dmax == 0:
        stats['DSI_prct'] = 0
        # stats.DSIc_prct = 0
    else:
        stats['DSI_prct'] = np.sqrt(d) / np.sqrt(dmax) * 100
    stats['SI_prct'] = (stats['ASI_prct'] + 2 * stats['DSI_prct']) / 3

    return stats
def get_baseline_P(signal, baseline_marker, long_win_points=4800, long_win_step=240, smooth_kernel_size=480):
    """
    Calculate the baseline of a signal using predicted baseline_marker.

    Parameters:
        signal (array-like): The input signal.
        baseline_marker (array-like): Binary labels indicating baseline points.
        long_win_points (int): Length of the long window in points (default is 4800).
        long_win_step (int): Step size of slide window (long_win_points) (default is 240)
        smooth_kernel_size (int): Kernel size for final median filter to smooth the estimated baseline

    Returns:
        array-like: The calculated baseline signal.
    """
    #因为我的baseline——maker 0为基线 因此调一下
    baseline_marker = np.where(baseline_marker==1,3,baseline_marker)#不在乎加速
    baseline_marker = np.where(baseline_marker==0,1,baseline_marker)
    signal = signal.flatten() if len(signal.shape) > 1 else signal.copy()
    len_signal = len(signal)
    estimated_baseline = np.zeros(len_signal)

    # Adjust long window length if signal is shorter
    real_long_win_points = min(len_signal, long_win_points)
    if len_signal <= long_win_points:
        print('WARNING: len_signal <= long_win_points!')

    half_long_win_points = round(real_long_win_points / 2)

    # Add padding to the signal and baseline
    padding = np.zeros(half_long_win_points)
    #为了处理边界情况，对信号和基线标记进行填充。填充的长度为窗口长度的一半
    baseline_marker_padded = np.concatenate((padding, baseline_marker, padding))
    signal_padded = np.concatenate((padding, signal, padding))

    # Calculate preliminary baseline
    for i_start in range(0, len_signal, long_win_step):
        window_indices = range(i_start, i_start + real_long_win_points)
        long_win_signal = signal_padded[window_indices]
        long_win_baseline = baseline_marker_padded[window_indices]#窗口大小real_long_win_points

        baseline_indices = np.where(long_win_baseline == 1)[0]#基线的坐标
        if len(baseline_indices) > 0:
            baseline_points = long_win_signal[baseline_indices]
            if len(baseline_points) / real_long_win_points >= 0.1:#预测的基线占窗口的10%以上
                estimated_baseline[i_start:i_start + long_win_step] = np.median(baseline_points)#中位数

    # Interpolate missing values in the preliminary baseline
    if np.any(estimated_baseline == 0):
        if np.all(estimated_baseline == 0):
            estimated_baseline = np.median(signal) * np.ones(len_signal)#如果上述没有10%的，则用信号中位数来代替
        else:
            estimated_baseline = interpolate_signal(estimated_baseline)

    # Calculate the final baseline
    distance = np.abs(signal - estimated_baseline)
    valid_baseline_indices = np.where((baseline_marker == 1) & (distance < 15))[0]#找出标记为基线且与初步基线差异小于15的点作为有效基线点

    if len(valid_baseline_indices) <= 1:#如果有效的基线点太少了，则用中位值代替
        final_baseline = np.median(signal) * np.ones(len_signal)
    else:
        final_baseline = np.zeros(len_signal)
        final_baseline[valid_baseline_indices] = signal[valid_baseline_indices]
        final_baseline = interpolate_signal(final_baseline)
        final_baseline = smooth_signal(final_baseline, smooth_kernel_size+1) # kernel should be odd.

    return final_baseline

