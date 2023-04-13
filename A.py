import numpy as np
import torch

def compute(file):
    # 读取数据集
    frequency = np.loadtxt('/home/caiyuzhu/caiyuzhu_space/workspace/CDF-zipf/Fit&Frequency/' + file + '_frequency.txt')
    error,C,s = np.loadtxt('/home/caiyuzhu/caiyuzhu_space/workspace/CDF-zipf/Fit&Frequency/' + file + '_Result.txt')
    sample_list = np.loadtxt('/home/caiyuzhu/caiyuzhu_space/workspace/CDF-zipf/Fit&Frequency/' + file + '_sampleList.txt')
    sampleSum = np.sum(sample_list)

    # 将数据转换为GPU张量
    C_tensor = torch.tensor(C, dtype=torch.float, device='cuda:6')
    s_tensor = torch.tensor(s, dtype=torch.float, device='cuda:6')
    frequency_tensor = torch.tensor(frequency, dtype=torch.float, device='cuda:6')
    sampleList_tensor = torch.tensor(sample_list, dtype=torch.float, device='cuda:6')
    sampleSum_tensor = torch.tensor(sampleSum, dtype=torch.float, device='cuda:6')

    # 计算Dr
    D_r = [0]
    sum_r_tensor = sampleList_tensor[0]
    for r in range(1, len(sample_list) + 1):
        CDF_sample = sum_r_tensor / sampleSum_tensor
        Dr = torch.abs(C_tensor * r ** s_tensor - CDF_sample)
        if Dr > D_r[-1]:
            D_r.append(Dr)
        # 打印进度
        if r % 1000000 == 0:
            print(f"Progress: {r}/{len(sample_list)}")
        if r == len(sample_list):
            break
        sum_r_tensor += sampleList_tensor[r]
    D_r = [float(x) for x in D_r]

    # 计算严格KS test所需的阈值D_alpha
    DS = np.sum(sample_list)
    D_01 = 1.63 / np.sqrt(DS)
    D_05 = 1.36 / np.sqrt(DS)

    # alpha = 0.01
    path_01 = '/home/caiyuzhu/caiyuzhu_space/workspace/CDF-zipf/result/' + file + '_KS_strict_0.01_Result.txt'
    for r in range(1, len(sample_list) + 1):
        # 判断是否通过KS检验
        if D_r[r] > D_01:
            with open(path_01, "w") as f:
                f.write("{}, {}, {}".format(D_r[r], r, int(sample_list[r-1])))
            break

    # alpha = 0.05
    path_05 = '/home/caiyuzhu/caiyuzhu_space/workspace/CDF-zipf/result/' + file + '_KS_strict_0.05_Result.txt'
    for r in range(1, len(sample_list) + 1):
        # 判断是否通过KS检验
        if D_r[r] > D_05:
            with open(path_05, "w") as f:
                f.write("{}, {}, {}".format(D_r[r], r, int(sample_list[r-1])))
            break

file_name = ['000webhost', 'chegg', 'Dodonew', 'duelingnetwork', 'rockyou', 'tianya_clean', 'wishbone', 'yahoo']
for file in file_name:
    compute(file=file)
        