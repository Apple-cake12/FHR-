# %%
from torch.utils.data import  Dataset
from torch.utils.data import DataLoader
from utils import *
from glob import glob
from unet_model import MAU_Net
import pandas as pd
import random
import numpy as np
from torch import  nn
# 设置随机种子
seed_value = 500  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution

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
class CTGDataset_test(Dataset):
    def __init__(self, fhr_paths, label_paths,baseline_paths):
        """
        Args:
            data_root (str): Root directory of the data.
            target_length (int): Target length for each sample in data points, set to 20 minutes at 4 Hz (20 * 240).
        """

        # Initialize lists to hold data
        self.fhrs_list = []
        self.labels_list = []
        self.baseline_list = []

        # Load each JSON file and process the data
        for fhr_path, label_path,baseline_path in zip(fhr_paths, label_paths,baseline_paths):#将数据和标签一一对应
            fhrs = np.load(fhr_path) / 255. #np.load函数来读取二进制信号数据 (4800,)归一化
            label = np.load(label_path)
            baseline = np.load(baseline_path)
            self.fhrs_list.append(fhrs)
            self.labels_list.append(label)
            self.baseline_list.append(baseline)

    def __len__(self):
        # Number of data samples
        return len(self.fhrs_list)

    def __getitem__(self, idx):
        """
        Retrieve data for a specific index from the preloaded lists.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            dict: Contains 'fhrs' and 'labels' as torch tensors, sampled and padded if necessary.
        """
        fhrs = self.fhrs_list[idx]
        labels = self.labels_list[idx]
        labels = np.where(labels == -1, 2, labels) #将CTU数据集的标签-1变成2
        baselines = self.baseline_list[idx]
        # Convert to PyTorch tensors

        fhrs_tensor = torch.tensor(fhrs, dtype=torch.float32).reshape(1, 4800)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        baseline_tensor = torch.tensor(baselines, dtype=torch.float32)

        return {
            'fhrs': fhrs_tensor,
            'labels': labels_tensor,
            'baselines':baseline_tensor
        }
# %%

# data_root = r'E:\暨南\小论文\Data\npy_pro\JNU_DB-331'
# data_root = r'E:\暨南\小论文\Data\npy_pro\GMU_DB-84'
# data_root = r'E:\暨南\小论文\Data\npy_pro\SMU_DB-234'
data_root = r'E:\暨南\小论文\Data\npy_pro\LCU_DB-66'

test_fhr_dir = os.path.join(data_root, 'test_sigs')  # 训练数据目录
test_label_dir = os.path.join(data_root, 'test_labels')  # 训练数据标签
test_baseline_dir = os.path.join(data_root, 'test_regression')  # 训练数据标签


test_fhr_paths = glob(os.path.join(test_fhr_dir, "*.npy"))  # 数据都是npy后缀
test_label_paths = [os.path.join(test_label_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]
test_baseline_paths = [os.path.join(test_baseline_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]

# %%
test_dataset = CTGDataset_test(test_fhr_paths, test_label_paths,test_baseline_paths)  # 返回字典'fhrs'和'labels' （1，4800）

# Initialize DataLoaders for train and test sets
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kernel_size=[21,31,61,81]
model_21 = MAU_Net(1, 3, kernel_size=kernel_size[0]).to(device)
model_21.load_state_dict(torch.load('./trained_model/model_LCU_21.pt'))

model_31 = MAU_Net(1, 3, kernel_size=kernel_size[1]).to(device)
model_31.load_state_dict(torch.load('./trained_model/model_LCU_31.pt'))

model_61 = MAU_Net(1, 3, kernel_size=kernel_size[2]).to(device)
model_61.load_state_dict(torch.load('./trained_model/model_LCU_61.pt'))

model_81 = MAU_Net(1, 3, kernel_size=kernel_size[3]).to(device)
model_81.load_state_dict(torch.load('./trained_model/model_LCU_81.pt'))
# %%
model_21.eval()
model_31.eval()
model_61.eval()
model_81.eval()

total_correct = 0
total_samples = 0
# 创建一个空的 DataFrame，用于存储评估指标
metrics_df = pd.DataFrame(columns=['Batch', 'Dice', 'IoU', 'Accuracy'])
# 定义一个计数器，用于记录batch的编号
batch_counter = 0
Index_Dice = Dice()
Index_Iou = IOU()
for batch in test_dataloader:
    fhrs = batch['fhrs'].to(device)  # 输入（batchsize,1,4800）
    labels = batch['labels'].to(device)
    baselines = batch['baselines'].to(device)
    with torch.no_grad():
        preds_21 = model_21(fhrs)  # （batchsize,3,4800）
        preds_31 = model_31(fhrs)  # （batchsize,3,4800）
        preds_61 = model_61(fhrs)  # （batchsize,3,4800）
        preds_81 = model_81(fhrs)
        # （batchsize,3,4800）
    preds_21 = torch.argmax(preds_21, dim=1)
    preds_31 = torch.argmax(preds_31, dim=1)
    preds_61 = torch.argmax(preds_61, dim=1)
    preds_81 = torch.argmax(preds_81, dim=1)
    preds = preds_21 & preds_31 & preds_61 & preds_81

    # 转换为CPU张量计算指标
    preds_tensor = preds.detach().cpu()  # 保持张量格式
    labels_tensor = labels.detach().cpu()

    # 计算指标（输入必须是PyTorch张量）
    dice_score = Index_Dice(preds_tensor, labels_tensor).item()
    iou_score = Index_Iou(preds_tensor, labels_tensor).item()

    # Accuracy计算
    correct = (preds_tensor == labels_tensor).sum().item()
    accuracy = correct / labels_tensor.numel()
    # 存储到DataFrame
    metrics_df.loc[batch_counter] = [batch_counter, dice_score, iou_score, accuracy]
    batch_counter += 1


    print('hhh')

# 保存到Excel文件
metrics_df.to_excel('batch_metrics_EMAU-Net.xlsx', index=False)





print('......SUMMARY......')

