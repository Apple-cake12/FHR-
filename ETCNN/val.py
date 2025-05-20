import torch

from utils import AverageMeter


def validate(model, val_dataloader, criterion1,criterion2, device):
    """
    Perform validation with non-overlapping sliding windows on each sample.

    Args:
        model: The trained model.
        val_dataloader: DataLoader for the validation set, with batch_size=1.
        criterion: The loss function to compute validation loss.
        target_length: The length (in samples) of each sliding window, corresponding to 20 minutes.

    Returns:
        val_loss: Average validation loss.
        val_accuracy: Validation accuracy.
    """
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_samples = 0
    running_loss = AverageMeter()

    with torch.no_grad():
        for batch in val_dataloader:
            fhrs = batch['fhrs'].to(device)  # Shape: [1, 1, signal_length]
            labels = batch['labels'].to(device)  # Ground truth shape: [1, signal_length]

            # Forward pass
            preds,_ = model(fhrs)  # Output shape: [1, num_classes, target_length]

            # Calculate loss for this window
            loss = 0.5 * criterion1(preds, labels) + 0.5 * criterion2(preds, labels)
            running_loss.update(loss.item())

            # Calculate accuracy for this window
            pred_cls = torch.argmax(preds, dim=1)  # Shape: [1, target_length]
            correct = (pred_cls == labels).sum().item()  # Count correct predictions
            total_correct += correct
            total_samples += labels.numel()  # Total number of elements in labels (batch_size * signal_length)

    # Calculate average validation loss and accuracy
    val_loss = running_loss.get_average()
    val_accuracy = total_correct / total_samples if total_samples > 0 else 0

    model.train()

    return val_loss, val_accuracy

