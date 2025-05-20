import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Initialize the DiceLoss class.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
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
        # Apply softmax to ensure predictions are in the probability space if logits are provided
        preds = torch.softmax(preds, dim=1)

        # Initialize dice score list
        dice_scores = []

        # Calculate Dice coefficient for each class
        for class_idx in range(preds.shape[1]):
            # Extract predictions for the current class
            pred_class = preds[:, class_idx, :]  # Shape: (batch_size, signal_length)
            # Create a binary mask for the current class in targets
            target_class = (targets == class_idx).float()  # Shape: (batch_size, signal_length)
            # Calculate intersection and union
            intersection = torch.sum(pred_class * target_class, dim=1)  # Sum over the signal length
            union = torch.sum(pred_class, dim=1) + torch.sum(target_class, dim=1)  # Sum over the signal length

            # Dice score for the current class
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)

        # Stack dice scores for each class and average over classes and batches
        dice_scores = torch.stack(dice_scores, dim=1)  # Shape: (batch_size, num_classes)
        dice_loss = 1 - dice_scores.mean()  # Final Dice loss

        return dice_loss
