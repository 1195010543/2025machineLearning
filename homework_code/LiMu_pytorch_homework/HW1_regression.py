import torch
import random
# 数据生成函数
def synthetic_data(w,b,num_examples):
    # 定义x 从均值为 0、标准差为 1 的正态分布中采样。；
    # 形状为 (num_examples, len(w))，即 num_examples  1000个样本，每个样本有 len(w) 2 个特征。
    #x就是1000行2列的矩阵
     x = torch.normal(0, 1, (num_examples, len(w)))
    #torch.matmul(x, w) 执行矩阵乘法：形状 (num_examples, len(w)) 与 (len(w),) 相乘 → 形状 (num_examples,)。
     y = torch.matmul(x, w) + b
    # 添加噪声-使数据更贴近真实场景，避免完全线性可分。生成和 y 形状一样的随机数（每个样本一个噪声），均值为 0，标准差 0.01（很小的噪声）。
     y += torch.normal(0, 0.01, y.shape)
    # 第 4 步：返回特征和标签;
    #y.reshape((-1, 1))：把 y 从一维数组（长度 num_examples）变成列向量（num_examples 行，1 列）。
    #不 reshape 时 y 是一维张量 (1000,)。后面如果和预测值（也是列向量）做减法，PyTorch 可能自动广播但容易出错。保持明确形状 (1000,1) 能让代码更清晰、不易出 bug。
     return x,y.reshape(-1,1)


#设计构造数据
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
print("features:",features[0],'\nlabels:',labels[0])


# 在训练时，我们不会一次性把所有 1000 个样本都扔进模型（内存可能不够，而且梯度下降的“随机性”也不够）。我们每次只取一小批（例如 10 个样本），计算损失和梯度，更新参数。这个函数就是负责把数据集切成小批量，并且每次打乱顺序
#features形状是 (1000, 2)。是所有的样本（x1,x2）每个都有1000个值（样本）
#labels：形状 (1000, 1)

def data_iter(batch_size, features, labels):
    # features 的形状是 (1000, 2)，len(features) 就是第一维的大小，即样本总数 1000。
    num_examples = len(features)
    # 生成一个列表 [0, 1, 2, ..., 999]，表示样本的原始序号。
    indices = list(range(num_examples))
    # 随机打乱这个列表，比如变成 [345, 12, 789, ..., 42]。这样后面取小批量时，每个批次的样本就是随机抽取的，实现随机梯度下降的随机性。
    random.shuffle(indices)
    # i 从 0 开始，每次增加 batch_size（比如 10），所以 i = 0, 10, 20, ..., 990。
    for i in range(0, num_examples, batch_size):
        # 可能会超出样本个数，所以到了最后会拿末尾的一个
        # indices[i : i+batch_size] 从打乱后的索引列表中切出一段，长度为 batch_size（最后一段可能不足）。
        # 例如：第一次循环 i=0，取 indices[0:10]，得到 [345, 12, 789, ...]（10 个随机索引）。
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        #  features[batch_indices] 就是：按照 batch_indices 里索引的顺序，取 features 的对应行，拼成一个小批量的特征矩阵。
        # 即取345行、12行、789行;labels 形状 (1000, 1)，取同样索引的行。
        yield features[batch_indices], labels[batch_indices]
    '''带有yield的函数在Python中被称之为generator(生成器)，也就是说，
        当你调用这个函数的时候，函数内部的代码并不立即执行 ，
        这个函数只是返回一个生成器(Generator Iterator)。'''


batch_size = 10

# data_iter是一个生成器，每次调用会在上一个状态的基础上再次调用
for x, y in data_iter(batch_size, features, labels):
    print(x, '\n', y)
    #     break是因为只取了一个小批量
    break


# 定义初始化模型参数
# 权重，形状 (2, 1)，表示两个特征对应两个权重。初始值从均值为0、标准差0.01的正态分布随机采样。这样做的目的是打破对称性，让不同神经元（这里只是两个权重）初始不同。
w = torch.normal(0,0.01,size = (2,1),requires_grad = True)
'''
b：偏置，形状 (1,)，初始为0。偏置通常可以初始化为0 ;创建一个一维张量，里面有一个元素，值为 0。
为什么不用标量？
在 PyTorch 中，标量是 torch.tensor(0.0)，形状为 ()。
但为了后续能与 torch.matmul(X, w) 的结果（形状 (batch_size, 1)）相加，PyTorch 的广播机制要求：'''
b = torch.zeros(1,requires_grad = True)


#定义线性回归模型  linear regression
def linreg(X,w,b):
#     线性回归模型
    return torch.matmul(X,w)+b

#定义损失函数  y_hat 预测值， y 真实值
def squared_loss(y_hat,y):
    '''均方损失，统一 y 的形状'''
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


#定义优化算法
'''params里面包含了w，b          lr为学习率'''
def  sgd(params,lr,batch_size):
    '''小批量梯度下降'''
    with torch.no_grad():
        for param in params:
#       之前求损失函数没有计算均值，所以这里除以batch_size，   w，b都需要计算梯度
            param -= lr*param.grad/batch_size
#       梯度手动置为0
            param.grad.zero_()


# 训练过程

'''指定超参'''
lr = 0.03  # 学习率
num_epochs = 3  # 扫描数据次数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)  # 预测和实际的小批量损失
        # 因为 l 的形状是（batch_size,1)，是一个长为批量大小的向量，而不是一个标量。
        # 并以此计算关于 w，b的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 这里注意批量大小，如果取到最后了，那么可能会比批量小

        # 把更新一次参数后，把整个数据放进去过一遍计算损失
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1},loss:{float(train_l.mean()):f}')
