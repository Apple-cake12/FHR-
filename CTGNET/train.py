# %%
import os
import numpy as np
from val import validate
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import *
from glob import glob
from unet_model import UNet
from loss import DiceLoss
from dataset import CTGDataset
import random
from torch.utils.tensorboard import SummaryWriter

summaryWriter = SummaryWriter("/root/tf-logs")
seed_value = 500  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution


# %%
# data_root = r'/root/JNU_DB-331'
# data_root = r'/root/GMU_DB-84'
data_root = r'/root/SMU_DB-234'

train_fhr_dir = os.path.join(data_root, 'train_sigs')  # 训练数据目录
train_label_dir = os.path.join(data_root, 'train_labels')  # 训练数据标签
val_fhr_dir = os.path.join(data_root, 'val_sigs')
val_label_dir = os.path.join(data_root, 'val_labels')

train_fhr_paths = glob(os.path.join(train_fhr_dir, "*.npy"))  # 数据都是npy后缀
train_label_paths = [os.path.join(train_label_dir, fname.split('/')[-1]) for fname in train_fhr_paths]
val_fhr_paths = glob(os.path.join(val_fhr_dir, "*.npy"))
val_label_paths = [os.path.join(val_label_dir, fname.split('/')[-1]) for fname in val_fhr_paths]

# %%
train_dataset = CTGDataset(train_fhr_paths, train_label_paths)  # 返回字典'fhrs'和'labels' （1，4800）
test_dataset = CTGDataset(val_fhr_paths, val_label_paths)

# Initialize DataLoaders for train and test sets
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNet(1, 3).to(device)
criterion1 = DiceLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

running_loss = AverageMeter()
running_dice = AverageMeter()
running_acc = AverageMeter()
running_best_acc = BestMeter("min")  # 跟踪最大值，即最高的验证准确率。

epochs = 200

# %%
model.train()
for epoch in range(epochs):
    total_correct = 0
    total_samples = 0

    # Iterate through the DataLoader
    for batch in train_dataloader:
        fhrs = batch['fhrs'].to(device)  # 输入（batchsize,1,4800）
        labels = batch['labels'].to(device)

        preds = model(fhrs)  # （batchsize,3,4800）
        loss = 0.5*criterion1(preds, labels) + 0.5*criterion2(preds, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item())

        pred_cls = torch.argmax(preds, dim=1)  # (32,4800)  # Predicted labels
        correct = (pred_cls == labels).sum().item()  # Count correct predictions
        total_correct += correct
        total_samples += labels.numel()  # Total number of elements in labels (batch_size * signal_length)

    train_loss = running_loss.get_average()
    running_loss.reset()
    # Calculate training accuracy for the epoch
    train_accuracy = total_correct / total_samples

    val_loss, val_accuracy = validate(model, test_dataloader, criterion1,criterion2,device)

    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")
    summaryWriter.add_scalars("loss", {"Train_loss": train_loss}, epoch + 1)
    summaryWriter.add_scalars("Accuracy", {"Train_Accuracy": train_accuracy}, epoch + 1)
    summaryWriter.add_scalars("loss", {"Val_loss": val_loss}, epoch + 1)
    summaryWriter.add_scalars("Accuracy", {"Val_Accuracy": val_accuracy}, epoch + 1)

    if val_loss < running_best_acc.get_best():
        running_best_acc.update(val_loss)
        torch.save(model.state_dict(), os.path.join('./trained_model', "model_SMU.pt"))
        print("Save model.")

# %%