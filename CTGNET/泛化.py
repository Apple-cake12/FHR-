# %%

from torch.utils.data import  Dataset
from torch.utils.data import DataLoader
from utils import *
from glob import glob
from unet_model import UNet
from metrics import *
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

# data_root = r'E:\暨南\小论文\Data\泛化\JNU_DB-331'
data_root = r'E:\暨南\小论文\Data\泛化\GMU_DB-84'
# data_root = r'E:\暨南\小论文\Data\泛化\SMU_DB-234'


test_fhr_dir = os.path.join(data_root, 'sigs')  # 训练数据目录
test_label_dir = os.path.join(data_root, 'labels')  # 训练数据标签
test_baseline_dir = os.path.join(data_root, 'baseline')  # 训练数据标签


test_fhr_paths = glob(os.path.join(test_fhr_dir, "*.npy"))  # 数据都是npy后缀
test_label_paths = [os.path.join(test_label_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]
test_baseline_paths = [os.path.join(test_baseline_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]

# %%
test_dataset = CTGDataset_test(test_fhr_paths, test_label_paths,test_baseline_paths)  # 返回字典'fhrs'和'labels' （1，4800）

# Initialize DataLoaders for train and test sets
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNet(1, 3).to(device)
model.load_state_dict(torch.load('./trained_model/model_LCU.pt'))


# %%
model.eval()
total_correct = 0
total_samples = 0
Dice = 0
Iou = 0
# Iterate through the DataLoader
MeasureList = {}  # 存储评估指标的特点
MeasureList['RMSD_bpm'] = []
MeasureList['Dice'] = []
MeasureList['Dice_Acc'] = []
MeasureList['Dice_Dec'] = []

MeasureList['Iou'] = []
MeasureList['Iou_Acc'] = []
MeasureList['Iou_Dec'] = []

MeasureList['Accuracy'] = []
MeasureList['SI_prct'] = []
MeasureList['MADI'] = []
MeasureList['MAE'] = []
# 初始化存储列表
MeasureList['Recall_Base'] = []
MeasureList['Recall_Acc'] = []
MeasureList['Recall_Dec'] = []
MeasureList['MacroF1'] = []
MeasureList['MCC'] = []
MeasureList['GMean'] = []

for batch in test_dataloader:
    fhrs = batch['fhrs'].to(device)  # 输入（batchsize,1,4800）
    labels = batch['labels'].to(device)
    baselines = batch['baselines'].to(device)
    with torch.no_grad():
        preds = model(fhrs)  # （batchsize,3,4800）
    stats = mstatscompare_P(fhr=fhrs,ground_labels=labels,pred_labels=preds,ground_baseline = baselines)

    RMSD_bpm = stats['RMSD_bpm']
    SI_prct = stats['SI_prct']
    MADI = stats['MADI'] * 100
    Dice = stats['Dice'] * 100
    Dice_Acc = stats['Dice_Acc'] * 100
    Dice_Dec = stats['Dice_Dec'] * 100

    Iou = stats['Iou'] * 100
    Iou_Acc = stats['Iou_Acc'] * 100
    Iou_Dec = stats['Iou_Dec'] * 100

    Accuracy = stats['Accuracy'] * 100
    MAE = stats['MAE']
    MeasureList['RMSD_bpm'] = np.append(MeasureList['RMSD_bpm'], RMSD_bpm)
    MeasureList['Accuracy'] = np.append(MeasureList['Accuracy'], Accuracy)

    MeasureList['Dice'] = np.append(MeasureList['Dice'], Dice)
    MeasureList['Dice_Acc'] = np.append(MeasureList['Dice_Acc'], Dice_Acc)
    MeasureList['Dice_Dec'] = np.append(MeasureList['Dice_Dec'], Dice_Dec)

    MeasureList['Iou'] = np.append(MeasureList['Iou'], Iou)
    MeasureList['Iou_Acc'] = np.append(MeasureList['Iou_Acc'], Iou_Acc)
    MeasureList['Iou_Dec'] = np.append(MeasureList['Iou_Dec'], Iou_Dec)


    MeasureList['SI_prct'] = np.append(MeasureList['SI_prct'], SI_prct)
    MeasureList['MADI'] = np.append(MeasureList['MADI'], MADI)
    MeasureList['MAE'] = np.append(MeasureList['MAE'], MAE)

    MeasureList['Recall_Base'] = np.append(MeasureList['Recall_Base'], stats['Recall_Base'] * 100)
    MeasureList['Recall_Acc'] = np.append(MeasureList['Recall_Acc'], stats['Recall_Acc'] * 100)
    MeasureList['Recall_Dec'] = np.append(MeasureList['Recall_Dec'], stats['Recall_Dec'] * 100)
    MeasureList['MacroF1'] = np.append(MeasureList['MacroF1'], stats['MacroF1'] * 100)
    MeasureList['MCC'] = np.append(MeasureList['MCC'], stats['MCC'] * 100)
    MeasureList['GMean'] = np.append(MeasureList['GMean'], stats['GMean'] * 100)

print('......SUMMARY......')
print('Median Dice_Acc = %0.4f\t' % (np.nanmedian(MeasureList['Dice_Acc'])),
      'Mean Dice_Acc = %0.4f\t' % (np.nanmean(MeasureList['Dice_Acc'])))
print('Median Dice_Dec = %0.4f\t' % (np.nanmedian(MeasureList['Dice_Dec'])),
      'Mean Dice_Dec = %0.4f\t' % (np.nanmean(MeasureList['Dice_Dec']))
      )
print('Median Dice = %0.4f\t' % (np.nanmedian(MeasureList['Dice'])),
      'Mean Dice = %0.4f\t' % (np.nanmean(MeasureList['Dice']))
      )
print('Median Iou_Acc = %0.4f\t' % (np.nanmedian(MeasureList['Iou_Acc'])),
      'Mean Iou_Acc = %0.4f\t' % (np.nanmean(MeasureList['Iou_Acc']))
      )
print('Median Iou_Dec = %0.4f\t' % (np.nanmedian(MeasureList['Iou_Dec'])),
      'Mean Iou_Dec = %0.4f\t' % (np.nanmean(MeasureList['Iou_Dec']))
      )
print('Median Iou = %0.4f\t' % (np.nanmedian(MeasureList['Iou'])),
      'Mean Iou = %0.4f\t' % (np.nanmean(MeasureList['Iou']))
      )
print('Median Accuracy = %0.4f\t' % (np.nanmedian(MeasureList['Accuracy'])),
      'Mean Accuracy = %0.4f\t' % (np.nanmean(MeasureList['Accuracy']))
      )
print('Median SI = %0.4f %%\t' % (np.nanmedian(MeasureList['SI_prct'])),
      'Mean SI = %0.4f %%\t' % (np.nanmean(MeasureList['SI_prct']))
      )
print('Median MADI = %0.4f %%\t' % (np.nanmedian(MeasureList['MADI'])),
      'Mean MADI = %0.4f %%\t' % (np.nanmean(MeasureList['MADI']))
      )
print('Median RMSD = %0.4f bpm\t' % (np.nanmedian(MeasureList['RMSD_bpm'])),# 计算置信区间0.4和0.6
      'Mean RMSD = %0.4f bpm\t' % (np.nanmean(MeasureList['RMSD_bpm']))
      )
print('Median MAE = %0.4f bpm\t' % (np.nanmedian(MeasureList['MAE'])),# 计算置信区间0.4和0.6
      'Mean MAE = %0.4f bpm\t' % (np.nanmean(MeasureList['MAE']))
      )
print('Median Recall_Base = %0.4f%%\t' % (np.nanmedian(MeasureList['Recall_Base'])),
      'Mean Recall_Base = %0.4f%%' % (np.nanmean(MeasureList['Recall_Base']))
      )
print('Median Recall_Acc = %0.4f%%\t' % (np.nanmedian(MeasureList['Recall_Acc'])),
      'Mean Recall_Acc = %0.4f%%' % (np.nanmean(MeasureList['Recall_Acc']))
      )
print('Median Recall_Dec = %0.4f%%\t' % (np.nanmedian(MeasureList['Recall_Dec'])),
      'Mean Recall_Dec = %0.4f%%' % (np.nanmean(MeasureList['Recall_Dec']))
      )
# 2. 宏F1
print('Median MacroF1 = %0.4f%%\t' % (np.nanmedian(MeasureList['MacroF1'])),
      'Mean MacroF1 = %0.4f%%' % (np.nanmean(MeasureList['MacroF1']))
      )

# 3. Matthews相关系数（MCC）
print('Median MCC = %0.4f%%\t' % (np.nanmedian(MeasureList['MCC'])),
      'Mean MCC = %0.4f%%' % (np.nanmean(MeasureList['MCC']))
      )

# 4. 几何平均（G-Mean）
print('Median GMean = %0.4f%%\t' % (np.nanmedian(MeasureList['GMean'])),
      'Mean GMean = %0.4f%%' % (np.nanmean(MeasureList['GMean']))
      )
print('......  END  ......')
