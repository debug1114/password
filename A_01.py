# 验证|Cr^s-CDF(sampleList[r])|是否随着r单调递增
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
    D_r_temp = 0
    sum_r_tensor = sampleList_tensor[0]
    for r in range(1, len(sample_list) + 1):
        CDF_sample = sum_r_tensor / sampleSum_tensor
        Dr = torch.abs(C_tensor * r ** s_tensor - CDF_sample)
        print("Dr:{},Dr':{}".format(Dr, D_r_temp))
        if Dr <= D_r_temp:
            return False
        D_r_temp = Dr
        # 打印进度
        if r % 1000000 == 0:
            print(f"Progress: {r}/{len(sample_list)}")
        if r == len(sample_list):
            break
        sum_r_tensor += sampleList_tensor[r]
    return True


file_name = ['000webhost', 'chegg', 'Dodonew', 'duelingnetwork', 'rockyou', 'tianya_clean', 'wishbone', 'yahoo']
for file in file_name:
    print("Processing:{}...".format(file))
    if(compute(file=file)):
        print("{} is monotonically incremental!\n".format(file))
    else:
        print("{} IS NOT monotonically incremental!\n".format(file))
        