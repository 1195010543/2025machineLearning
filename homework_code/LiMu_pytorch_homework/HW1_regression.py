import torch

# 数据生成函数
def synthetic_data(w,b,num_examples):
    # 定义x 从均值为 0、标准差为 1 的正态分布中采样。；
    # 形状为 (num_examples, len(w))，即 num_examples  1000个样本，每个样本有 len(w) 2 个特征。
    #x就是1000行2列的矩阵
     x = torch.normal(0, 1, (num_examples, len(w)))
    #torch.matmul(x, w) 执行矩阵乘法：形状 (num_examples, len(w)) 与 (len(w),) 相乘 → 形状 (num_examples,)。
     y = torch.matmul(x, w) + b
    # 添加噪声-使数据更贴近真实场景，避免完全线性可分。
