import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 不需要显示窗口，直接保存图片
import time

# ==========================================
# 设备检测
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 数据结构和辅助类
# ==========================================

class Params:
    """内层梯度下降参数"""
    def __init__(self, n_iter, lr):
        self.n_iter = n_iter
        self.lr = lr

class Inits:
    """初始化值"""
    def __init__(self, u0, v0):
        self.u0 = u0
        self.v0 = v0

class Dataset:
    """数据集封装"""
    def __init__(self, X, Y, alpha, x_hat, N, K, d):
        self.X = X          # (N, d) 特征矩阵
        self.Y = Y          # (N,) 标签
        self.alpha = alpha  # 正则化系数
        self.x_hat = x_hat  # (K, d) 按类别聚合的特征
        self.N = N          # 样本数
        self.K = K          # 类别数
        self.d = d          # 特征维度

# ==========================================
# Douglas-Rachford 核心算法 (GPU 版本)
# ==========================================

def grad_F(u, ds):
    """
    计算 F(u) 的梯度
    F(u) = (alpha * N / 2) ||u||^2 - <x_hat, u>
    grad_F(u) = alpha * N * u - x_hat
    """
    # return ds.alpha * ds.N * u - ds.x_hat
    return ds.alpha * u - (ds.x_hat / ds.N)
    

def inner_gd(u_last, v_last, ds, params, rng):
    """
    内层梯度下降，求解子问题 (GPU 加速)
    """
    batch_size = len(rng)
    rng_tensor = torch.tensor(list(rng), device=device)
    
    # 将全局解复制 batch_size 次，初始化并行子问题
    u = u_last.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, K, d)
    
    # 预取当前批次的数据和对偶变量
    X_batch = ds.X[rng_tensor]  # (batch_size, d)
    v_batch = v_last[rng_tensor]  # (batch_size, K, d)
    
    # 预计算 grad_F(u_last)
    grad_F_u_last = grad_F(u_last, ds)  # (K, d)
    
    for i in range(params.n_iter):
        # X_j 是 x_j 重复 K 次，打包为 (batch_size, K, d)
        Xj = X_batch.unsqueeze(1).expand(-1, ds.K, -1)  # (batch_size, K, d)
        
        # 计算 <x_j, u_k> 对所有 k
        _XjU = torch.einsum('nkd, nkd -> nk', Xj, u)  # (batch_size, K)
        
        # LSE 梯度: softmax(logits) * x_j
        _grad_LSE = Xj * _XjU.softmax(dim=1).unsqueeze(-1)  # (batch_size, K, d)
        
        # 总梯度 (Douglas-Rachford 形式)
        grad_H = grad_F(u, ds) + v_batch - grad_F_u_last + _grad_LSE
        
        # 梯度下降
        u = u - params.lr * grad_H
    
    return u

def parallel_douglas_rachford_gpu(tau, inits, n_iter, inner_params, ds, X_test, Y_test):
    """
    并行 Douglas-Rachford 分裂算法 (GPU 加速版本)
    增加了每个 iteration 的测试集评估
    """
    u_t = inits.u0.clone().to(device)  # (K, d) 全局权重
    v_t = inits.v0.clone().to(device)  # (N, K, d) 对偶变量
    batch_size = 1000                   # 批处理大小
    
    hist_train_acc = []
    hist_train_loss = []
    hist_test_acc = []   # 新增：测试集准确率历史
    hist_test_loss = []  # 新增：测试集损失历史
    
    # 评估训练集
    train_acc, train_loss = evaluate_model(u_t, ds)
    hist_train_acc.append(train_acc)
    hist_train_loss.append(train_loss)
    
    # 评估测试集（每个 iteration）
    test_acc, test_loss = evaluate_on_test(u_t, X_test, Y_test)
    hist_test_acc.append(test_acc)
    hist_test_loss.append(test_loss)

    print(f"Starting Parallel DR on {device} (N={ds.N}, Alpha={ds.alpha}, Tau={tau})...")
    start_time = time.time()
    
    for t in range(n_iter):
        current_tau = tau * (0.9 ** t) # 每轮乘0.95
        # current_tau = tau 
        
        # Step 1 & 2: 分批次并行求解 N 个 argmin 子问题，并更新 v
        for start in range(0, ds.N, batch_size):
            end = min(start + batch_size, ds.N)
            rng = range(start, end)
            rng_tensor = torch.tensor(list(rng), device=device)
            
            start_time1 = time.time()
            # 求解当前批次的子问题 (在 GPU 上并行计算)
            u_batch = inner_gd(u_t, v_t, ds, inner_params, rng)  # (batch_size, K, d)
            elapsed1 = time.time() - start_time1
            # 更新对偶变量 v
            v_t[rng_tensor] = v_t[rng_tensor] + current_tau * (grad_F(u_batch, ds) - grad_F(u_t, ds))
        
        # Step 3: 更新全局变量 u (闭式解)
        u_t = (1.0 / (ds.alpha * ds.N)) * (ds.x_hat + v_t.sum(dim=0))
        
        # 评估训练集
        train_acc, train_loss = evaluate_model(u_t, ds)
        hist_train_acc.append(train_acc)
        hist_train_loss.append(train_loss)
        
        # 评估测试集（每个 iteration）
        test_acc, test_loss = evaluate_on_test(u_t, X_test, Y_test)
        hist_test_acc.append(test_acc)
        hist_test_loss.append(test_loss)
        
        print(f"  DR Iter {t+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"DR Finished on {device}. Time: {elapsed:.2f}s\n")
    
    # 将结果移回 CPU
    return u_t.cpu(), hist_train_acc, hist_train_loss, hist_test_acc, hist_test_loss

# ==========================================
# 评估函数
# ==========================================

def evaluate_model(weights, ds):
    """评估模型准确率和损失（训练集）"""
    with torch.no_grad():
        logits = torch.matmul(ds.X, weights.T)  # (N, K)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == ds.Y).float().mean().item() * 100
        
        # 计算 Cross-Entropy Loss
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fn(logits, ds.Y).item()
    
    return acc, loss

def evaluate_on_test(weights, X_test, Y_test):
    """在测试集上评估模型"""
    with torch.no_grad():
        logits = torch.matmul(X_test, weights.T)  # (N_test, K)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == Y_test).float().mean().item() * 100
        
        # 计算 Cross-Entropy Loss
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fn(logits, Y_test).item()
    
    return acc, loss

# ==========================================
# Baseline SGD (也支持 GPU)
# ==========================================

def baseline_sgd(ds, X_test, Y_test, n_epochs=30, lr=0.01, momentum=0.9, batch_size=64):
    """标准 SGD 求解多分类逻辑回归 (GPU 加速)
    增加了每个 epoch 的测试集评估
    """
    print(f"Starting Baseline SGD on {device} (Epochs={n_epochs})...")
    start_time = time.time()
    
    w = torch.zeros(ds.K, ds.d, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([w], lr=lr, momentum=momentum, weight_decay=ds.alpha)
    loss_fn = nn.CrossEntropyLoss()
    
    hist_train_acc = []
    hist_train_loss = []
    hist_test_acc = []   # 新增：测试集准确率历史
    hist_test_loss = []  # 新增：测试集损失历史
    

    # 评估训练集
    with torch.no_grad():
        train_acc, train_loss = evaluate_model(w.detach(), ds)
        hist_train_acc.append(train_acc)
        hist_train_loss.append(train_loss)
        
        # 评估测试集（每个 epoch）
        test_acc, test_loss = evaluate_on_test(w.detach(), X_test, Y_test)
        hist_test_acc.append(test_acc)
        hist_test_loss.append(test_loss)

    for epoch in range(n_epochs):
        # 随机打乱数据
        perm = torch.randperm(ds.N, device=device)
        X_shuffled = ds.X[perm]
        Y_shuffled = ds.Y[perm]
        
        # Mini-batch 训练
        for i in range(0, ds.N, batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = Y_shuffled[i:i+batch_size]
            
            logits = xb @ w.T
            loss = loss_fn(logits, yb)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 评估训练集
        with torch.no_grad():
            train_acc, train_loss = evaluate_model(w.detach(), ds)
            hist_train_acc.append(train_acc)
            hist_train_loss.append(train_loss)
            
            # 评估测试集（每个 epoch）
            test_acc, test_loss = evaluate_on_test(w.detach(), X_test, Y_test)
            hist_test_acc.append(test_acc)
            hist_test_loss.append(test_loss)
            
            print(f"  SGD Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"SGD Finished on {device}. Time: {elapsed:.2f}s\n")
    return w.detach().cpu(), hist_train_acc, hist_train_loss, hist_test_acc, hist_test_loss

# ==========================================
# 数据准备
# ==========================================

def prepare_data(n_samples=2000):
    """加载 MNIST 数据并预计算 x_hat，移动到 GPU"""
    print("Loading MNIST Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    
    # 随机采样
    indices = torch.randperm(len(train_dataset))[:n_samples]
    data_subset = torch.utils.data.Subset(train_dataset, indices)
    loader = torch.utils.data.DataLoader(data_subset, batch_size=n_samples, shuffle=False)
    
    X_list, Y_list = [], []
    for x, y in loader:
        X_list.append(x)
        Y_list.append(y)
    
    X = torch.cat(X_list, dim=0).view(n_samples, -1)  # (N, 784)
    Y = torch.cat(Y_list, dim=0)  # (N,)
    
    # 添加 bias
    bias = torch.ones(n_samples, 1)
    X = torch.cat([X, bias], dim=1)  # (N, 785)
    
    N, d = X.shape
    K = 10  # 10 个类别
    
    # 计算 x_hat: 按类别聚合的特征
    x_hat = torch.zeros(K, d)
    for k in range(K):
        mask = (Y == k)
        if mask.sum() > 0:
            x_hat[k] = X[mask].sum(dim=0)
    
    # 将数据移动到 GPU
    X = X.to(device)
    Y = Y.to(device)
    x_hat = x_hat.to(device)
    
    print(f"Data Loaded: N={N}, d={d}, K={K}")
    print(f"Data moved to: {device}")
    print(f"x_hat computed: shape={x_hat.shape}\n")
    return X, Y, x_hat, N, K, d


def prepare_test_data(n_test_samples=300):
    """加载 MNIST 测试集"""
    print("Loading MNIST Test Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    indices = torch.randperm(len(test_dataset))[:n_test_samples]
    test_subset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(test_subset, batch_size=n_test_samples, shuffle=False)

    # loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    X_list, Y_list = [], []
    for x, y in loader:
        X_list.append(x)
        Y_list.append(y)
    
    X_test = torch.cat(X_list, dim=0).view(n_test_samples, -1)  # (10000, 784)
    Y_test = torch.cat(Y_list, dim=0)  # (n_test_samples,)
    
    # 添加 bias
    bias = torch.ones(n_test_samples, 1)
    X_test = torch.cat([X_test, bias], dim=1)  # (10000, 785)
    
    # 移动到 GPU
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    print(f"Test Data Loaded: N={len(test_dataset)}, d={X_test.shape[1]}\n")
    return X_test, Y_test

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    # 参数设置
    N_SAMPLES = 60000
    N_TEST_SAMPLES = 1000
    ALPHA = 1       # 正则化系数
    N_ITER = 10        # 外层迭代次数
    
    # 准备训练数据 (自动移到 GPU)
    X, Y, x_hat, N, K, d = prepare_data(N_SAMPLES)
    ds = Dataset(X, Y, ALPHA, x_hat, N, K, d)
    
    # 准备测试数据
    X_test, Y_test = prepare_test_data(N_TEST_SAMPLES)
    
    # ==========================================
    # 方法 1: 并行 Douglas-Rachford (GPU)
    # ==========================================
    print("="*60)
    print("Method 1: Parallel Douglas-Rachford (GPU)")
    print("="*60)
    
    # # 初始化
    # std = torch.sqrt(torch.tensor(2.0 / (K + d)))
    # u0 = torch.randn(K, d, device=device) * std
    # v0 = torch.randn(N, K, d, device=device) * 0.01
    # 使用零初始化
    # u0 = torch.zeros(K, d, device=device)
    # v0 = torch.zeros(N, K, d, device=device)
    # inits = Inits(u0, v0)

    u0 = torch.zeros(K, d, device=device)

    # v^0_j = (1/N) * ∇F(u^0)
    v0_template = grad_F(u0, ds) / ds.N  # (K, d)
    v0 = v0_template.unsqueeze(0).repeat(ds.N, 1, 1)  # (N, K, d)

    inits = Inits(u0, v0)

    inner_params = Params(n_iter=20, lr=1e-3)
    
    # 运行 DR 算法（传入测试集）
    w_dr, hist_train_acc_dr, hist_train_loss_dr, hist_test_acc_dr, hist_test_loss_dr = parallel_douglas_rachford_gpu(
        tau=15,
        inits=inits,
        n_iter=N_ITER,
        inner_params=inner_params,
        ds=ds,
        X_test=X_test,
        Y_test=Y_test
    )
    
    # ==========================================
    # 方法 2: 标准 SGD Baseline (GPU)
    # ==========================================
    print("="*60)
    print("Method 2: Baseline SGD (GPU)")
    print("="*60)
    
    w_sgd, hist_train_acc_sgd, hist_train_loss_sgd, hist_test_acc_sgd, hist_test_loss_sgd = baseline_sgd(
        ds,
        X_test,
        Y_test,
        n_epochs=N_ITER,
        lr=1e-4,
        momentum=0.9,
        batch_size=64
    )
    
    # ==========================================
    # 可视化对比（包含测试集曲线）
    # ==========================================
    print("="*60)
    print("Generating Comparison Plot...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 左上：训练集准确率
    axes[0, 0].plot(hist_train_acc_dr, 'b-o', label='Parallel DR', linewidth=2, markersize=3)
    axes[0, 0].plot(hist_train_acc_sgd, 'r--s', label='Baseline SGD', linewidth=2, markersize=3)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title(f'Training Accuracy (N={N_SAMPLES})', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 右上：测试集准确率
    axes[0, 1].plot(hist_test_acc_dr, 'b-o', label='Parallel DR', linewidth=2, markersize=3)
    axes[0, 1].plot(hist_test_acc_sgd, 'r--s', label='Baseline SGD', linewidth=2, markersize=3)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title(f'Test Accuracy (N={N_TEST_SAMPLES})', fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 左下：训练集损失
    axes[1, 0].plot(hist_train_loss_dr, 'b-o', label='Parallel DR', linewidth=2, markersize=3)
    axes[1, 0].plot(hist_train_loss_sgd, 'r--s', label='Baseline SGD', linewidth=2, markersize=3)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title(f'Training Loss (N={N_SAMPLES})', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 右下：测试集损失
    axes[1, 1].plot(hist_test_loss_dr, 'b-o', label='Parallel DR', linewidth=2, markersize=3)
    axes[1, 1].plot(hist_test_loss_sgd, 'r--s', label='Baseline SGD', linewidth=2, markersize=3)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title(f'Test Loss (N={N_TEST_SAMPLES})', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to 'final_comparison.png'")
    
    # 打印最终结果
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    print(f"Device: {device}")
    print(f"\nTraining Set (N={N_SAMPLES}):")
    print(f"  Parallel DR  - Final Acc: {hist_train_acc_dr[-1]:.2f}%, Final Loss: {hist_train_loss_dr[-1]:.4f}")
    print(f"  Baseline SGD - Final Acc: {hist_train_acc_sgd[-1]:.2f}%, Final Loss: {hist_train_loss_sgd[-1]:.4f}")
    print(f"\nTest Set (N={N_TEST_SAMPLES}):")
    print(f"  Parallel DR  - Test Acc: {hist_test_acc_dr[-1]:.2f}%, Test Loss: {hist_test_loss_dr[-1]:.4f}")
    print(f"  Baseline SGD - Test Acc: {hist_test_acc_sgd[-1]:.2f}%, Test Loss: {hist_test_loss_sgd[-1]:.4f}")
    print("="*60)
    
    plt.show()

#放图