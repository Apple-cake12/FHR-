import numpy as np
import pandas as pd
from scipy.signal import medfilt, butter, filtfilt


def get_accident(signal, baseline, diff_threshold=0.5, duration_threshold=60, amplitude_threshold=15):
    """
    Detect accelerations and decelerations in the signal relative to the baseline.

    Parameters:
        signal (array-like): The input signal.
        baseline (array-like): The baseline signal.
        diff_threshold (float): Minimum difference (in bpm) between signal and baseline for detection (default=0.5 bpm).
        duration_threshold (int): Minimum duration threshold in points (default=60 points. If freq=4, then this means 15 seconds).
        amplitude_threshold (float): Minimum amplitude threshold (in bpm, default=15 bpm).

    Returns:
        tuple: Two numpy arrays containing accelerations and decelerations. 
        The arrays have shape (N, 3), where N is the total numbers of acc or dec. 
        The first col is the start index. 
        The second col is the end index.
        The third col is the third index. 
    """
    signal = signal.flatten() if len(signal.shape) > 1 else signal.copy()
    baseline = baseline.flatten() if len(baseline.shape) > 1 else baseline.copy()

    # Detect accelerations
    acc_candidates = detect_candidates(signal, baseline, diff_threshold)#start_idx, end_idx - 1, np.array(max_idx)]
    #返回的是认证过后的加速区域stard_idx、end_idx、max_ids
    accelerations, _ = validate_accident(acc_candidates, signal - baseline, duration_threshold, amplitude_threshold)

    # Detect decelerations
    dec_candidates = detect_candidates(baseline, signal, diff_threshold)
    decelerations, _ = validate_accident(dec_candidates, baseline - signal, duration_threshold, amplitude_threshold)

    return np.transpose(accelerations)/240, np.transpose(decelerations)/240#返回的是转置 列表反转后变成每一个元素为一个区域


def validate_accident(candidates, signal_diff, duration_threshold, amplitude_threshold):
    """
    Validate acceleration or deceleration candidates based on duration and amplitude thresholds.

    Parameters:
        candidates (list): Start, end, and max indices of candidates.
        signal_diff (array-like): Difference between the signal and baseline.
        duration_threshold (int): Minimum duration threshold in points.
        amplitude_threshold (float): Minimum amplitude threshold in bpm.

    Returns:
        tuple: Validated candidates and rejected candidates.
    """
    start_idx, end_idx, max_idx = candidates

    # Validate duration
    valid_mask = (end_idx - start_idx) >= duration_threshold #利用布尔值筛选出时间＞duration_threshold的位置
    start_idx, end_idx, max_idx = start_idx[valid_mask], end_idx[valid_mask], max_idx[valid_mask]#只保留时间＞60的坐标

    if len(start_idx) == 0:
        return None, candidates

    # Validate amplitude
    amplitude_mask = np.array([
        np.max(signal_diff[start:end]) >= amplitude_threshold#监测波峰是否高于胎心率基线15bpm
        for start, end in zip(start_idx, end_idx)
    ])

    valid_candidates = [start_idx[amplitude_mask], end_idx[amplitude_mask], max_idx[amplitude_mask]]
    rejected_candidates = [start_idx[~amplitude_mask], end_idx[~amplitude_mask], max_idx[~amplitude_mask]]

    return valid_candidates, rejected_candidates#返回通过的区域valid_candidates和没有通过的区域rejected_candidates


def detect_candidates(signal1, signal2, diff_threshold):
    """
    Detect candidate regions for accelerations or decelerations.监测加减速区域

    Parameters:
        signal1 (array-like): The first signal.
        signal2 (array-like): The second signal.
        diff_threshold (float): Minimum difference threshold for detection in bpm.

    Returns:
        list: Start, end, and max indices of candidates.
    """
    binsig = np.zeros(len(signal1) + 2)
    binsig[1:-1] = (signal1 - signal2) > diff_threshold#筛选出信号幅度大于基线0.5bpm的
    #监测二值信号从0值跳为1的位置
    start_idx = np.where((binsig[1:] > 0) & (binsig[:-1] == 0))[0]
    # 监测二值信号从1值跳为0的位置
    end_idx = np.where((binsig[1:] == 0) & (binsig[:-1] > 0))[0]

    max_idx = [#argmax来寻找每个候选区域的最大差异值索引
        start + np.argmax(signal1[start:end] - signal2[start:end])
        for start, end in zip(start_idx, end_idx)
    ]#存储着差值最大的索引值

    return [start_idx, end_idx - 1, np.array(max_idx)]


def get_baseline(signal, baseline_marker, long_win_points=4800, long_win_step=240, smooth_kernel_size=480):
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


def interpolate_signal(signal):
    """
    Interpolate missing values (zeros) in the signal.

    Parameters:
        signal (array-like): The input signal with missing values (zeros).

    Returns:
        array-like: The interpolated signal.
    """
    signal = signal.flatten() if len(signal.shape) > 1 else signal.copy()
    non_zero_indices = np.where(signal > 0)[0]

    if len(non_zero_indices) > 0:
        # Fill leading zeros 将信号开头的零值 替换为第一个非零值
        signal[:non_zero_indices[0]] = signal[non_zero_indices[0]]
        
        # Interpolate between non-zero values
        for i in range(1, len(non_zero_indices)):
            start, end = non_zero_indices[i - 1], non_zero_indices[i]
            if end - start > 1:#如果两个非零值之间存在零值 则线性填充
                signal[start + 1:end] = np.linspace(signal[start], signal[end], end - start - 1)
        
        # Fill trailing zeros 末尾的零值则用最后一个非零值来代替
        signal[non_zero_indices[-1]:] = signal[non_zero_indices[-1]]

    return signal


def smooth_signal(signal, window_size):
    """
    Apply a median filter to smooth the signal.

    Parameters:
        signal (array-like): The input signal.
        window_size (int): The length of the smoothing window.

    Returns:
        array-like: The smoothed signal.
    """
    signal = signal.flatten() if len(signal.shape) > 1 else signal.copy()

    # Add padding to the signal
    padding_len = int(window_size)
    padding_start = np.ones(padding_len) * signal[0]#480
    padding_end = np.ones(padding_len) * signal[-1]
    signal_padded = np.concatenate((padding_start, signal, padding_end))

    # Apply median filter
    smoothed_signal = medfilt(signal_padded, kernel_size=window_size)#中值滤波
    smoothed_signal = smoothed_signal[padding_len:-padding_len]#去除填充部分

    return smoothed_signal


def apply_butterworth_filter(signal_data, sampling_rate, cutoff_frequency, filter_order=4):
    """
    Apply a Butterworth low-pass filter to the input signal.

    Parameters:
        signal_data (array-like): The input signal to be filtered.
        sampling_rate (float): Sampling rate of the signal (in Hz).
        cutoff_frequency (float): Cutoff frequency for the low-pass filter (in Hz).
        filter_order (int): The order of the Butterworth filter (default is 4).

    Returns:
        array-like: The filtered signal.
    """
    nyquist_frequency = sampling_rate / 2  # Nyquist frequency
    normalized_cutoff = cutoff_frequency / nyquist_frequency  # Normalized cutoff frequency

    # Design the filter
    b_coefficients, a_coefficients = butter(filter_order, normalized_cutoff, btype='low')

    # Apply zero-phase filtering
    filtered_signal = filtfilt(b_coefficients, a_coefficients, signal_data)

    # Clip extreme values to mitigate overshoots
    filtered_signal = np.clip(filtered_signal, np.min(signal_data), np.max(signal_data))

    return filtered_signal


def preprocess_signal(fhr_signal,fs=4):
    """
    Preprocess the FHR signal by removing unreliable parts and interpolating missing values.

    Parameters:
        fhr_signal (array-like): The input FHR signal.

    Returns:
        Interpolated FHR signal (array-like)
    """
    # Remove small parts from the signal
    fhr_processed = remove_unreliable_parts(fhr_signal.copy())

    # Interpolate missing values in the signal
    fhr_interpolated = interpolate_fhr(fhr_processed)
    #不进行插值处理的
    fhr  = fhr_signal.copy()
    fhr = remove_unreliable_parts(fhr)
    tmp = np.where(fhr == 0)
    if tmp is not None:
        fhr[tmp] = 0
    fhr[0:(15 * 60 * fs + 1)] = 0  # 开头15分钟数据
    fhr[(-1 * 60 * fs):] = 0  # 结尾一分钟数据置零

    return fhr_interpolated,fhr


def remove_unreliable_parts(fhr_signal, short_gap_threshold=20, anomaly_threshold=25):
    """
    Clean the FHR signal by replacing invalid values, short gaps, and anomalies with zeros.
    Parameters:
        fhr_signal (array-like): The input FHR signal.
        short_gap_threshold (int): Threshold in samples for short gaps to be removed (default is 20 samples).短间隔阈值
        anomaly_threshold (int): Threshold in FHR change to detect anomalies (default is 25 bpm).异常阈值

    Returns:
        array-like: The cleaned FHR signal.
    """
    fhr = fhr_signal.copy()

    # Remove values outside the valid range
    fhr[(fhr < 50) | (fhr > 220)] = 0#去除范围外的信号

    # Identify gaps in the signal fhr[1:]代表fhr哥哥信号点的下一个时间点信号
    #只有当前时间点为0，但是下个时间点不为0的信号点才会记录坐标 +1代表该点为从零值跃迁的点
    gap_starts = np.where((fhr[:-1] == 0) & (fhr[1:] > 0))[0] + 1

    # Remove short gaps
    for start in gap_starts:
        gap_end = np.where(fhr[start:] == 0)[0]
        if gap_end.size > 0 and gap_end[0] < short_gap_threshold:
            fhr[start:start + gap_end[0] + 1] = 0 #将零值区间缩为一个0

    # Detect and remove anomalies (e.g., doubling or halving patterns)
    for start in gap_starts:
        gap_end = np.where(fhr[start:] == 0)[0]
        if gap_end.size > 0 and gap_end[0] < 30 * 4:  # Longer gaps
            gap_length = gap_end[0]
            prev_valid = np.where(fhr[:start] > 0)[0]
            next_valid = np.where(fhr[start + gap_length:] > 0)[0]

            if prev_valid.size > 0 and next_valid.size > 0:
                prev_value = fhr[prev_valid[-1]]#获取了star前后的非零值
                next_value = fhr[start + gap_length + next_valid[0]]

                # Remove doubling or halving anomalies 如果一个点与前后点相差25bpm以上
                if (fhr[start] - prev_value < -anomaly_threshold) and (fhr[start + gap_length - 1] - next_value < -anomaly_threshold):
                    fhr[start:start + gap_length] = 0
                elif (fhr[start] - prev_value > anomaly_threshold) and (fhr[start + gap_length - 1] - next_value > anomaly_threshold):
                    fhr[start:start + gap_length] = 0

    return fhr


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
            fhr[zero_start:next_valid] = np.linspace(fhr[zero_start - 1], fhr[next_valid], next_valid - zero_start)
            # fhr[zero_start-1:next_valid+1] = np.linspace(fhr[zero_start - 1], fhr[next_valid], next_valid - zero_start+2)


            idx = next_valid

        # Set trailing zeros to the last valid value
        last_valid = valid_indices[-1]
        fhr[last_valid:] = fhr[last_valid]

    return fhr


def read_ctg_from_csv(file_path):
    """
    Reads a CTG (Cardiotocography) file and extracts the FHR, TOCO, and baseline data.

    Parameters:
        file_path (str): The full path to the CTG file.

    Returns:
        list of np.ndarray: A list containing three arrays: FHR (Fetal Heart Rate), TOCO (Uterine Contractions),
                            and baseline.
    """
    # Read the specified columns from the CSV file
    data = pd.read_csv(file_path, usecols=['fhr', 'toco', 'baseline'])
    
    # Extract individual columns as numpy arrays
    fhr = data['fhr'].to_numpy().reshape(-1, 1)
    toco = data['toco'].to_numpy().reshape(-1, 1)
    baseline = data['baseline'].to_numpy().reshape(-1, 1)

    return fhr, toco, baseline


def segment_signal_with_overlap(signal, segment_length_points=4800, step_size=None):
    """
    Splits the input signal into overlapping segments.

    Parameters:
        signal (array-like): The input 1D signal array.
        segment_length_points (int): The length of each segment in points (default is 4800).
        step_size (int): The step size for overlapping segments in number of points. If None, defaults to 1/3 of the segment length.

    Returns:
        list of np.ndarray: A list of overlapping signal segments.
    """
    import numpy as np

    # Ensure the signal is a 1D numpy array
    signal = np.asarray(signal).flatten()

    # Determine step size
    if step_size is None:
        step_size = segment_length_points // 3  # Default to 1/3 of segment length

    # Calculate the number of full segments
    total_samples = len(signal)
    #完整的segment_length_points片段个数
    num_full_segments = (total_samples - segment_length_points) // step_size + 1  #1是为了补上前面减去完整的片段

    segments = []

    # Generate overlapping segments
    for i in range(num_full_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_length_points
        segments.append(signal[start_idx:end_idx])

    # Handle the last segment if there are remaining samples
    remaining_samples = total_samples - (num_full_segments * step_size)
    if remaining_samples > 0:
        last_segment = signal[-remaining_samples:]  # Include all remaining points
        segments.append(last_segment)

    return segments


def splice_segments(signal, baselines, segment_length_points=4800, step_size=600):
    """
    Splices the predicted baselines from segments into a single array matching the original signal's length.

    Parameters:
        signal (array-like): The original 1D signal array.
        baselines (list of array-like): List of predicted baselines for each segment (binary values, same length as segments).
        segment_length_points (int): The length of each segment in points (default is 4800).
        step_size (int): The step size for overlapping segments in points (default is 600).

    Returns:
        np.ndarray: A 1D array of the spliced baseline matching the length of the original signal.
    """
    import numpy as np

    # Ensure signal and baselines are numpy arrays
    signal = np.asarray(signal).flatten()
    baselines = [np.asarray(baseline).flatten() for baseline in baselines]

    # Initialize the final baseline array and a weight array
    total_samples = len(signal)
    spliced_baseline = np.zeros(total_samples, dtype=np.float32)
    weight_array = np.zeros(total_samples, dtype=np.float32)

    # Reconstruct the baseline by adding overlapping contributions
    for i, baseline in enumerate(baselines):
        start_idx = i * step_size
        end_idx = start_idx + segment_length_points

        # Handle contributions to the spliced baseline and weights
        # 若大于切片长度，会自动调整为数组大小；但是适合用于最后一个元素不完整的情况，以此来防止超出原信号长度
        spliced_baseline[start_idx:end_idx] += baseline[:total_samples - start_idx]
        weight_array[start_idx:end_idx] += 1

    # Normalize overlapping regions by dividing by the weight array
    # Avoid division by zero by masking non-contributing regions
    valid_weights = weight_array > 0#叠加的权重数
    spliced_baseline[valid_weights] /= weight_array[valid_weights]

    # Convert to binary by rounding values
    spliced_baseline = np.round(spliced_baseline).astype(int)#四舍五入

    return spliced_baseline
def evlMADI(FHR0, LDB1_, LDB2_):
    FHR = FHR0.copy()
    LDB1 = LDB1_.copy()
    LDB2 = LDB2_.copy()
    if len(FHR.shape) > 1:
        FHR = FHR.flatten()
    if len(LDB1.shape) > 1:
        LDB1 = LDB1.flatten()
    if len(LDB2.shape) > 1:
        LDB2 = LDB2.flatten()
    MADI = -1
    invaidIndex = np.where(FHR != 0)[0]#筛选出非0值 没有双线性插值的地方也去掉了
    if invaidIndex is not None:
        FHRu = FHR[invaidIndex]
        L1 = LDB1[invaidIndex]
        L2 = LDB2[invaidIndex]

        Coef = np.ones(240) / 240#一个长度为 240 的数组，每个元素为 1/240，用于计算滑动平均。
        tmp = np.square(L1 - FHRu)
        D1 = np.convolve(tmp, Coef, mode='full')# 使用滑动平均（卷积）对 tmp 进行平滑处理
        D1 = D1[240:-239]
        tmp = np.where(D1 != 0)
        D1[tmp] = np.sqrt(D1[tmp])
        D1 += 3 # bpm #（参数D1=DL1/FHRu）

        tmp = np.square(L2 - FHRu)
        D2 = np.convolve(tmp, Coef, mode='full')
        D2 = D2[240:-239]
        tmp = np.where(D2 != 0)
        D2[tmp] = np.sqrt(D2[tmp])
        D2 += 3 # bpm（参数D2=DL2/FHRu）

        tmp = L1[120:-120] - L2[120:-120]
        D = np.square(tmp)
        MADI = np.mean(D/(D1 * D2 + D))
    return MADI
def filteraccident(acc_, FHR):
    if acc_ is None:
        return None
    acc = acc_.copy()
    cnt = np.size(acc, 0)
    keep = np.ones(cnt).astype(int)
    for i in range(cnt):
        #acc[i, 0] 和 acc[i, 1] 分别表示事故的开始和结束时间，乘以 240是将时间转换为样本点数
        s = FHR[ (int(acc[i, 0]*240)) : np.min([len(FHR), int(int(acc[i, 1]*240)+1)]) ] == 0
        if sum(s)/len(s) > 0.33333:#如果等于0的数在整体占据0.33333以上 则将这个事件置于0
            keep[i] = 0
    tmp = np.where(keep == 1)[0]
    if tmp is not None:
        acc = acc[tmp, :]
    else:
        acc = None
    return acc
def accMatch(a1_, a2_, e_):
    a1 = a1_.copy()
    a2 = a2_.copy()
    e = e_.copy()
    fs = 4

    M1 = [None] * np.size(a1, 0)
    M2 = [None] * np.size(a2, 0)#M1 和 M2 用于存储每个事件在另一组中的匹配索引
    Q = np.zeros((np.size(a1, 0), np.size(a2, 0)))#是一个布尔矩阵，用于标记匹配关系。
    trueMatch = np.zeros((np.min([np.size(a1, 0), np.size(a2, 0)]), 4))#用于存储匹配事件的详细信息
    n = 0
    T = np.zeros((4, 4))#T 是一个 4×4 的矩阵，用于统计不同类型的匹配情况。
    counted1 = np.zeros((np.size(a1, 0)))-100
    counted2 = np.zeros((np.size(a2, 0)))-100#标记每个事件的匹配状态。

    for i in range(np.size(a1, 0)):#遍历a1，寻找a2中的匹配项
        #等效于固定区间a1[i],检测a2中所有区域中与区域a1[i]发生重叠的编号
        M1[i] = np.where((a2[:,1] > (a1[i,0] + 5/60)) & ((a2[:,0] + 5/60) < a1[i,1]))[0]
        #a1[i,0] + 5/60 将a1[i]的起始时间扩展5min  第一个条件代表a2的结束时间，必须晚于a1[i]的起始时间
        #a2[:,0]+ 5/60 表示将a2的所有事件的起始时间扩展5min  第二条件代表a2的事件的起始时间必须早于a1[i]的结束时间
        #寻找a2中哪些时间与a1[i]事件有重叠
        if len(M1[i]) != 0:
            Q[i,M1[i]] = 1 #记录当前a1[i]与a2中重叠的事件位置+1  Q的行代表a1事件  列代表a2事件  若有重叠区域则会置于1
            #检查是否有特殊区域
        else: # 没有重叠区域，即a1检测出来，a2没有检测出来
            # e的作用需要确认，假设为0，此时的为，a1[i]出现在e附近15s的动态变化范围内?，因为实际e应该为空，因此下面的if始终不满足
            if np.any((abs(a1[i,0] - e[:,0]) < 15/60) & (abs(a1[i,1] - e[:,1]) < 15/60)):
                T[0,1] += 1 #属于独立检测，但是可能位于不可靠段？
                counted1[i] = -2
            else: # 确定a1[i]满足独立检测条件，+1
                T[0,2] += 1
                counted1[i] = -1

    for i in range(np.size(a2, 0)):#遍历a2，寻找a1中的匹配项
        #等效于固定区间a2[i],检测a1中所有区域中与区域a1[i]发生重叠的编号
        M2[i] = np.where((a1[:,1] > a2[i,0] + 5/60) & (a1[:,0] + 5/60 < a2[i,1]))[0]
        #a2[i]的起始时间扩展5min后必须早于a1的结束时间
        #a2[i]的结束时间缩短5min后必须晚于a1的起始时间
        if len(M2[i]) != 0:
            Q[M2[i],i] = 1#记录当前a2[i]与a1中重叠的事件位置+1
        else:# 没有重叠区域，即a2检测出来，a1没有检测出来
            if np.any((abs(a2[i,0] - e[:,0]) < 15/60) & (abs(a2[i,1] - e[:,1]) < 15/60)):
                T[1,0] += 1#属于独立检测，但是可能位于不可靠段？
                counted2[i] = -2
            else:# 确定a2[i]满足独立检测条件，+1
                T[2,0] += 1
                counted2[i] = -1
        if len(M2[i]) == 1:
            # tmp = np.array(M1)[M2[i]]
            tmp = [M1[j] for j in M2[i]]
            if len(tmp[0]) == 1:#彼此相互匹配（完全一致）
                T[0,0] += 1
                counted2[i] = M2[i]#保存其匹配项索引 a2[i] >> M2[i]
                counted1[M2[i]] = i#保存其匹配项索引 a1[M2[i]] >> a2[i]
                tmp = a1[M2[i],:]
                # print(n)
                trueMatch[n,:] = np.concatenate((tmp[0], a2[i,:]),axis=0)
                n += 1

    trueMatch = trueMatch[0:-1,:]#丢弃最后一行？
    while np.any(counted1 == -100) or np.any(counted2 == -100):
        for i in range(np.size(a1, 0)):
            if (len(M1[i]) == 0) and (counted1[i] == -100):
                T[0,3] += 1 #acc1在acc2中无匹配项
                counted1[i] = -3
            elif (len(M1[i]) == 1) and (counted1[i] == -100):#唯一匹配，但是不满足条件？
                ind = [M2[j] for j in M1[i]][0].flatten()
                if ind is not None:
                    for k in ind:
                        if k != i:
                            tmp = np.where(M1[k] != M1[i])[0]#是否需要[0]需要确认
                            if tmp is not None:
                                M1[k] = M1[k][tmp]
                # np.array(M2)[M1[i]] = i
                for idx in M1[i]:
                    if idx < len(M2) and len(M2[idx]) == 0:  # 检查索引有效且 M1[idx] 为空
                        M2[idx] = [i]  # 更新 M1[idx]
                    elif idx < len(M2):  # 如果 M1[idx] 不为空，直接赋值
                        M2[idx][0] = i  # 假设我们只关心第一个匹配项

                if -100 != counted2[M1[i]]:
                    print('Erreur 1')
                T[0,0] += 1
                counted1[i] = M1[i]
                counted2[M1[i]] = i

        for i in range(np.size(a2, 0)):
            if (len(M2[i]) == 0) and (counted2[i] == -100):
                T[3,0] += 1
                counted2[i] = -3
            elif (len(M2[i]) == 1) and (counted2[i] == -100):
                # ind = np.array(M1)[M2[i]][0].flatten()
                ind = [M1[j] for j in M2[i]][0].flatten()
                if ind is not None:
                    for k in ind:
                        if k != i:
                            tmp = np.where(M2[k] != M2[i])[0]
                            if tmp is not None:
                                M2[k] = M2[k][tmp]
                # np.array(M1)[M2[i]] = i
                # M1 = [[j if len(j) > 0 else i for j in M1_i] for M1_i in M1]
                # 假设 M2[i] 包含需要更新的索引
                for idx in M2[i]:
                    if idx < len(M1) and len(M1[idx]) == 0:  # 检查索引有效且 M1[idx] 为空
                        M1[idx] = [i]  # 更新 M1[idx]
                    elif idx < len(M1):  # 如果 M1[idx] 不为空，直接赋值
                        M1[idx][0] = i  # 假设我们只关心第一个匹配项
                if -100 != counted1[M2[i]]:
                    print('Erreur 1')
                T[0,0] += 1
                counted2[i] = M2[i]
                counted1[M2[i]] = i
    q1 = np.where(counted1 != -2)[0].flatten()
    q2 = np.where(counted2 != -2)[0].flatten()
    if len(q1) != 0 and len(q2) != 0:
        Q = Q[q1,:]
        Q = Q[:,q2]
        tmp = [(np.sum(Q, axis=0) == 0)]
        Q = np.concatenate((Q, tmp), axis=0)
        tmp = [(np.sum(Q, axis=1) == 0)]
        tmp = np.transpose(tmp)
        Q = np.concatenate((Q, tmp), axis=1)
    else:
        Q = None
    return [T, trueMatch, Q]#, counted1, counted2]
    #: 一个 4×4 的统计矩阵，表示不同类型的匹配情况：
    #[0,0]: 完全匹配的事件数量。
    #[0,1] 和 [1,0]: 可能不可靠的独立检测事件数量。
    #[0,2] 和 [2,0]: 独立检测事件数量。
    #[0,3] 和 [3,0]: 未匹配事件数量。

def mstatscompare(FHR_, LDB1_, LDB2_, acc1_, acc2_, overshoots=None):
    FHR = FHR_.copy()
    LDB1 = LDB1_.copy()#baseline
    LDB2 = LDB2_.copy()#prebaseline
    acc1 = acc1_.copy()#[realAcc[:,0:2], realDec[:,0:2]]
    acc2 = acc2_.copy()#[preAcc[:,0:2], preDec[:,0:2]]
    stats = {}
    FHR = FHR.flatten()
    LDB1 = LDB1.flatten()
    LDB2 = LDB2.flatten()
    stats['MADI'] = evlMADI(FHR, LDB1, LDB2)
    invaidIndex = np.where(FHR != 0)[0]
    if invaidIndex is None:
        return stats
    FHRu = FHR[invaidIndex]
    L1 = LDB1[invaidIndex]
    L2 = LDB2[invaidIndex]
    acc1[0] = filteraccident(acc1[0], FHR)#过滤掉在acc1候选者中缺失的信号数据
    acc1[1] = filteraccident(acc1[1], FHR)
    acc2[0] = filteraccident(acc2[0], FHR)
    acc2[1] = filteraccident(acc2[1], FHR)
    overshoots = filteraccident(overshoots, FHR)

    #基线差
    stats['RMSD_bpm'] = np.sqrt(np.mean(np.square(L1 - L2)))
    #超过15bpm的百分比
    stats['Diff_Over_15_bpm_prct'] = np.mean(abs(L1 - L2) > 15) * 100
    #计算一致性指标
    stats['Index_Agreement'] = 1 - np.sum(np.square(L1 - L2))\
                      / np.sum(np.square(np.abs(L1 - np.mean(L1)) + np.abs(L2 - np.mean(L2))))
    # Decelerations:
    [T, trueMatch, Qd] = accMatch(acc1[1], acc2[1], np.zeros((0, 2))) #realDec[:,0:2], preDec[:,0:2]
    stats['Dec_Match'] = T[0,0] #Number of common deceleration
    stats['Dec_Only_1'] = T[0,2] + T[0,3] #Number of deceleration detected only by Analyse 1
    stats['Dec_Only_2'] = T[2,0] + T[3,0] #Number of deceleration detected only by Analyse 2
    stats['Dec_Doubled_On_1'] = T[0,3] #Number of deceleration detected on Analyse 2 corresponding to two decelerations on Analyse 1
    stats['Dec_Doubled_On_2'] = T[3,0] #Number of deceleration detected on Analyse 1 corresponding to two decelerations on Analyse 2
    if T[0,0] + T[0,2] + T[0,3] == 0:
        stats['Dec_Only_1_Rate'] = 0
    else:
        stats['Dec_Only_1_Rate'] = (T[0,2] + T[0,3]) / (T[0,0] + T[0,2] + T[0,3])
    if T[0,0] + T[2,0] + T[3,0] == 0:
        stats['Dec_Only_2_Rate'] = 0
    else:
        stats['Dec_Only_2_Rate'] = (T[2,0] + T[3,0]) / (T[0,0] + T[2,0] + T[3,0])
    stats['Dec_Fmes'] = (1 - stats['Dec_Only_1_Rate']) * (1 - stats['Dec_Only_2_Rate'])\
               / ((1 - stats['Dec_Only_1_Rate']) + (1 - stats['Dec_Only_2_Rate']))
    stats['Dec_Start_RMSD_s'] = 60 * np.sqrt(np.mean(np.square(trueMatch[:,0] - trueMatch[:,2])))
    stats['Dec_Start_Avg_2_M_1_s'] = -60 * np.mean(trueMatch[:,0]-trueMatch[:,2])
    stats['Dec_Length_RMSD_s'] = 60 * np.sqrt(np.mean(
        np.square(trueMatch[:,1] - trueMatch[:,0] - trueMatch[:,3] + trueMatch[:,2])))
    stats['Dec_Length_Avg_2_M_1_s'] = -60 * np.mean(
        trueMatch[:,1]-trueMatch[:,0]-trueMatch[:,3]+trueMatch[:,2])

    # Accelerations:
    [T, trueMatch, Qa] = accMatch(acc1[0], acc2[0], np.zeros((0, 2)))
    stats['Acc_Match'] = T[0,0]
    stats['Acc_Only_1'] = T[0,2] + T[0,3]
    stats['Acc_Only_2'] = T[2,0] + T[3,0]
    stats['Acc_Doubled_On_1'] = T[0,3]
    stats['Acc_Doubled_On_2'] = T[3,0]
    if T[0,0] + T[0,2] + T[0,3] == 0:
        stats['Acc_Only_1_Rate'] = 0
    else:
        stats['Acc_Only_1_Rate'] = (T[0, 2] + T[0, 3]) / (T[0, 0] + T[0, 2] + T[0, 3])
    if T[0,0] + T[2,0] + T[3,0] == 0:
        stats['Acc_Only_2_Rate'] = 0
    else:
        stats['Acc_Only_2_Rate'] = (T[2,0] + T[3,0]) / (T[0,0] + T[2,0] + T[3,0])
    stats['Acc_Fmes'] = (1 - stats['Acc_Only_1_Rate']) * (1 - stats['Acc_Only_2_Rate'])\
               / ((1 - stats['Acc_Only_1_Rate']) + (1 - stats['Acc_Only_2_Rate']))
    stats['Acc_Start_RMSD_s'] = 60 * np.sqrt(np.mean(np.square(trueMatch[:,0] - trueMatch[:,2])))
    stats['Acc_Start_Avg_2_M_1_s'] = -60 * np.mean(trueMatch[:,0]-trueMatch[:,2])
    stats['Acc_Length_RMSD_s'] = 60 * np.sqrt(np.mean(
        np.square(trueMatch[:,1] - trueMatch[:,0] - trueMatch[:,3] + trueMatch[:,2])))
    stats['Acc_Length_Avg_2_M_1_s'] = -60 * np.mean(
        trueMatch[:,1]-trueMatch[:,0]-trueMatch[:,3]+trueMatch[:,2])

    #synthetic baseline inconsistency index
    FHRi = interpolate_fhr(FHR)
    cFHR1 = np.cumsum(FHRi - LDB1) / 240 #min
    cFHR2 = np.cumsum(FHRi - LDB2) / 240 #min
    AccArea1 = cFHR1[np.round(acc1[0][:,1] * 4 * 60).astype(int)] -cFHR1[np.round(acc1[0][:,0] * 4 * 60 + 1).astype(int)]
    AccArea1 = np.concatenate((AccArea1, [0]),axis=0)

    AccArea2 = cFHR2[np.round(acc2[0][:,1] * 4 * 60).astype(int)]-cFHR2[np.round(acc2[0][:,0] * 4 * 60 + 1).astype(int)]
    AccArea2 = np.concatenate((AccArea2, [0]),axis=0)

    DecArea1 = -cFHR1[np.round(acc1[1][:,1] * 4 * 60).astype(int)]+cFHR1[np.round(acc1[1][:,0] * 4 * 60 + 1).astype(int)]
    DecArea1 = np.concatenate((DecArea1, [0]), axis=0)

    DecArea2 = -cFHR2[np.round(acc2[1][:,1] * 4 * 60).astype(int)]+cFHR2[np.round(acc2[1][:,0] * 4 * 60 + 1).astype(int)]
    DecArea2 = np.concatenate((DecArea2, [0]), axis=0)

    d = 0
    dmax = 0
    dc = 0
    dmaxc = 0
    if Qa is not None:
        for i in range(np.size(Qa, 0)):
            for j in range(np.size(Qa, 1)):
                if Qa[i,j] == 1:
                    d += np.square(AccArea1[i] - AccArea2[j])
                    dmax +=  np.square(np.max([AccArea1[i], AccArea2[j]]))
                    dc += np.abs(AccArea1[i] - AccArea2[j])
                    dmaxc += np.max([AccArea1[i], AccArea2[j]])
    if dmax == 0:
        stats['ASI_prct'] = 0
        # stats.ASIc_prct = 0
    else:
        stats['ASI_prct'] = np.sqrt(d) / np.sqrt(dmax) * 100
        # stats.ASIc_prct = dc / dmaxc * 100
    d = 0
    dmax = 0
    if Qd is not None:
        for i in range(np.size(Qd, 0)):
            for j in range(np.size(Qd, 1)):
                if Qd[i,j] == 1:
                    d += np.square(DecArea1[i] - DecArea2[j])
                    dmax += np.square(np.max([DecArea1[i], DecArea2[j]]))
                    dc += np.abs(DecArea1[i] - DecArea2[j])
                    dmaxc += np.max([DecArea1[i], DecArea2[j]])
    if dmax == 0:
        stats['DSI_prct'] = 0
        # stats.DSIc_prct = 0
    else:
        stats['DSI_prct'] = np.sqrt(d) / np.sqrt(dmax) * 100
        # stats.DSIc_prct = dc / dmaxc * 100
    stats['SI_prct'] = (stats['ASI_prct'] + 2 * stats['DSI_prct']) / 3
    # stats.SIc_prct = (stats.ASIc_prct + 2 * stats.DSIc_prct) / 3;
    return stats
