# %%
import os

from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import Config as config
from UCTransNet import UCTransNet
from torch.utils.data import  Dataset
from torch.utils.data import DataLoader
from utils import *
from glob import glob
from metrics import *
from utils import create_dir
import matplotlib

matplotlib.use('Agg')  # 或者尝试其他后端，如 'TkAgg', 'Qt5Agg'
import matplotlib.pyplot as plt

import random

seed_value = 500  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution
def region(label):
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

    return np.transpose(Acc_list), np.transpose(Dec_list)
def visual_acc_dec(fhr,ground_labels,pred_labels,ground_baseline,filename,save_dir=r'.\acc_dec_results'):
    create_dir([save_dir])  # 创建save_dir文件夹
    FHR = fhr.detach().cpu().squeeze().numpy().copy() * 255
    FHR = apply_butterworth_filter(FHR, sampling_rate=4, cutoff_frequency=0.3, filter_order=6)
    ground_baselines = ground_baseline.detach().cpu().squeeze().numpy().copy()
    ground_cls = ground_labels.detach().cpu().squeeze().numpy().copy()
    pred_cls = pred_labels.detach().cpu().squeeze().numpy().copy()  # (32,4800)  # Predicted labels

    pred_baselines = get_baseline_P(FHR, pred_cls, 4800, 240, 480)

    # Extract "real" (ground truth) accelerations/decelerations from the provided baseline
    # 遍历样本 查找加减速事件
    Acc_ground_region, Dec_ground_region = region(ground_cls)
    Acc_pred_region, Dec_pred_region = region(pred_cls)
    ground_region = []
    ground_region.append(Acc_ground_region)
    ground_region.append(Dec_ground_region)
    pred_region = []
    pred_region.append(Acc_pred_region)
    pred_region.append(Dec_pred_region)
    real_acc=ground_region[0]
    real_dec=ground_region[1]
    pred_acc=pred_region[0]
    pred_dec=pred_region[1]
    real_acc_region = np.zeros_like(FHR)
    pred_acc_region = np.zeros_like(FHR)
    real_dec_region = np.zeros_like(FHR)
    pred_dec_region = np.zeros_like(FHR)

    for idx in real_acc:
        real_acc_region[idx[0]:idx[1]] = 1

    for idx in pred_acc:
        pred_acc_region[idx[0]:idx[1]] = 1

    for idx in real_dec:
        real_dec_region[idx[0]:idx[1]] = 1

    for idx in pred_dec:
        pred_dec_region[idx[0]:idx[1]] = 1

    points_per_plot = 1000  # 每个子图中绘制的数据点数量
    plots_per_fig = 5  # 每个图像文件中所包含的子图数量

    # Generate plots
    total_points = len(FHR)  # 信号总长度
    num_plots = total_points // points_per_plot + int(total_points % points_per_plot > 0)  # 总共要绘制的个数

    file_name = filename.split('.')[0]
    for plot_idx in range(num_plots):
        start = plot_idx * points_per_plot
        end = min(start + points_per_plot, total_points)

        fig, axs = plt.subplots(plots_per_fig, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(f"File: {file_name}, Plot {plot_idx + 1}", fontsize=16)

        # Plot each signal in a subplot
        axs[0].plot(FHR[start:end], label='FHR Filtered')
        axs[0].legend()
        axs[0].set_ylabel("FHR")

        axs[1].plot(real_acc_region[start:end], label='Real Acceleration Region', color='orange')
        axs[1].legend()
        axs[1].set_ylabel("Region")

        axs[2].plot(pred_acc_region[start:end], label='Predicted Acceleration Region', color='green')
        axs[2].legend()
        axs[2].set_ylabel("Region")

        axs[3].plot(real_dec_region[start:end], label='Real Deceleration Region', color='red')
        axs[3].legend()
        axs[3].set_ylabel("Region")

        axs[4].plot(pred_dec_region[start:end], label='Predicted Deceleration Region', color='purple')
        axs[4].legend()
        axs[4].set_xlabel("Time")
        axs[4].set_ylabel("Region")

        # Save figure
        plt.tight_layout()
        print(file_name)
        save_path = os.path.join(save_dir, f"{file_name}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(f"Plots for file {filename}")


# %%
class CTGDataset_test(Dataset):
    def __init__(self, fhr_paths, label_paths,baseline_paths,file_name):
        """
        Args:
            data_root (str): Root directory of the data.
            target_length (int): Target length for each sample in data points, set to 20 minutes at 4 Hz (20 * 240).
        """

        # Initialize lists to hold data
        self.fhrs_list = []
        self.labels_list = []
        self.baseline_list = []
        self.file_name_list = []

        # Load each JSON file and process the data
        for fhr_path, label_path,baseline_path,file_name in zip(fhr_paths, label_paths,baseline_paths,file_name):#将数据和标签一一对应
            fhrs = np.load(fhr_path) / 255. #np.load函数来读取二进制信号数据 (4800,)归一化
            label = np.load(label_path)
            baseline = np.load(baseline_path)
            self.fhrs_list.append(fhrs)
            self.labels_list.append(label)
            self.baseline_list.append(baseline)
            self.file_name_list.append(file_name)

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
        filename = self.file_name_list[idx]
        # Convert to PyTorch tensors

        fhrs_tensor = torch.tensor(fhrs, dtype=torch.float32).reshape(1, 4800)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        baseline_tensor = torch.tensor(baselines, dtype=torch.float32)

        return {
            'fhrs': fhrs_tensor,
            'labels': labels_tensor,
            'baselines':baseline_tensor,
            'file_name':filename
        }
# %%
data_root = r'E:\暨南\小论文\Data\泛化\GMU_DB-84'
test_fhr_dir = os.path.join(data_root, 'sigs')  # 训练数据目录
test_label_dir = os.path.join(data_root, 'labels')  # 训练数据标签
test_baseline_dir = os.path.join(data_root, 'baseline')  # 训练数据标签


test_fhr_paths = glob(os.path.join(test_fhr_dir, "*.npy"))  # 数据都是npy后缀
test_label_paths = [os.path.join(test_label_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]
test_baseline_paths = [os.path.join(test_baseline_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]
file_name =[fname.split('\\')[-1] for fname in test_fhr_paths]
# %%
test_dataset = CTGDataset_test(test_fhr_paths, test_label_paths,test_baseline_paths,file_name)  # 返回字典'fhrs'和'labels' （1，4800）

# Initialize DataLoaders for train and test sets
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config_vit = config.get_CTranS_config()  # 包含模型transformer参数的字典
kernel_size=[21,31,61,81]
model_21 = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, kernel_size=int(kernel_size[0])).to(device)
model_21.load_state_dict(torch.load('./trained_model/model_LCU_21.pt'))

model_31 = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, kernel_size=int(kernel_size[1])).to(device)
model_31.load_state_dict(torch.load('./trained_model/model_LCU_31.pt'))

model_61 = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, kernel_size=int(kernel_size[2])).to(device)
model_61.load_state_dict(torch.load('./trained_model/model_LCU_61.pt'))

model_81 = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, kernel_size=int(kernel_size[3])).to(device)
model_81.load_state_dict(torch.load('./trained_model/model_LCU_81.pt'))
# %%
# %%
model_21.eval()
model_31.eval()
model_61.eval()
model_81.eval()

for batch in test_dataloader:
    fhrs = batch['fhrs'].to(device)  # 输入（batchsize,1,4800）
    labels = batch['labels'].to(device)
    baselines = batch['baselines'].to(device)
    file_name = batch['file_name'][0]
    with torch.no_grad():
        preds_21, _ = model_21(fhrs)  # （batchsize,3,4800）
        preds_31, _ = model_31(fhrs)  # （batchsize,3,4800）
        preds_61, _ = model_61(fhrs)  # （batchsize,3,4800）
        preds_81, _ = model_81(fhrs)
        # （batchsize,3,4800）
    preds_21 = torch.argmax(preds_21, dim=1)
    preds_31 = torch.argmax(preds_31, dim=1)
    preds_61 = torch.argmax(preds_61, dim=1)
    preds_81 = torch.argmax(preds_81, dim=1)
    preds = preds_21 & preds_31 & preds_61 & preds_81
    visual_acc_dec(fhr=fhrs,ground_labels=labels,pred_labels=preds,ground_baseline=baselines,filename=file_name)

# %%