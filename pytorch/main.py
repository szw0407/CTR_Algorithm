import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from model.DeepFM import DeepFM
from model.xDeepFM import xDeepFM
from model.FM import FactorizationMachine
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt # Add matplotlib import

# 1. 加载并预处理数据
# 读取预处理后的Amazon Book数据集
raw_data = pd.read_csv('data/amazon-books-100k-preprocessed.csv', index_col=0)

# 2. 特征和标签
# 最后一列为label
X = raw_data.iloc[:,:-1]
y = raw_data.iloc[:,-1].values

# 3. 计算每个特征的取值个数（fields）
fields = X.max().values + 1

# 4. 划分数据集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# 5. 转为Tensor，并放到设备上

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.from_numpy(X_train.values).long().to(device)
X_val = torch.from_numpy(X_val.values).long().to(device)
X_test = torch.from_numpy(X_test.values).long().to(device)
y_train = torch.from_numpy(y_train).long().to(device)
y_val = torch.from_numpy(y_val).long().to(device)
y_test = torch.from_numpy(y_test).long().to(device)

train_set = Data.TensorDataset(X_train, y_train)
val_set = Data.TensorDataset(X_val, y_val)
train_loader = Data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
val_loader = Data.DataLoader(dataset=val_set, batch_size=64, shuffle=False)

def train(model, patience=8, use_amp=True):
    """
    训练模型

    :param model: 模型
    :param patience: 早停的耐心值
    :param use_amp: 是否使用自动混合精度
    :return: 每个epoch的训练损失列表, 每个epoch的验证损失列表，最佳验证AUC
    """
    model = model.to(device)
    epoches = 50
    best_auc = 0
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    scaler = torch.amp.GradScaler() if use_amp else None

    epoch_train_losses = [] # Store average train loss per epoch
    epoch_val_losses = []   # Store average validation loss per epoch

    for epoch in range(epoches):
        current_epoch_train_batch_losses = [] # Renamed from train_loss to avoid confusion
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        model.train()
        for batch, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoches} [Train]")):
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                ):
                    pred = model(x).squeeze(-1)
                    loss = criterion(pred, y.float().detach())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x).squeeze(-1)
                loss = criterion(pred, y.float().detach())
                loss.backward()
                optimizer.step()
            current_epoch_train_batch_losses.append(loss.item())
        
        avg_epoch_train_loss = sum(current_epoch_train_batch_losses)/len(current_epoch_train_batch_losses) if current_epoch_train_batch_losses else 0
        epoch_train_losses.append(avg_epoch_train_loss)

        model.eval()
        current_epoch_val_batch_losses = [] # Renamed from val_loss to avoid confusion
        prediction = []
        y_true = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epoches} [Val]")):
                if scaler is not None:
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu",
                    ):
                        pred = model(x).squeeze(-1)
                        loss = criterion(pred, y.float().detach())
                else:
                    pred = model(x).squeeze(-1)
                    loss = criterion(pred, y.float().detach())
                current_epoch_val_batch_losses.append(loss.item())
                prediction.extend(torch.sigmoid(pred).tolist())
                y_true.extend(y.tolist())
        
        avg_epoch_val_loss = sum(current_epoch_val_batch_losses)/len(current_epoch_val_batch_losses) if current_epoch_val_batch_losses else float('inf')
        epoch_val_losses.append(avg_epoch_val_loss)
        
        val_auc = roc_auc_score(y_true=y_true, y_score=prediction)
        print(f"EPOCH {epoch+1} train loss : {avg_epoch_train_loss:.5f}   validation loss : {avg_epoch_val_loss:.5f}   validation auc is {val_auc:.5f}")
        
        # Early stopping
        if val_auc > best_auc or (val_auc == best_auc and avg_epoch_val_loss < best_loss):
            best_auc = val_auc
            best_loss = avg_epoch_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
    return epoch_train_losses, epoch_val_losses, best_auc

# 6. 评估DeepFM
deepfm = DeepFM(feature_fields=fields, embed_dim=8, mlp_dims=(48,24), dropout=0.25)
print('Training DeepFM...')
_, deepfm_val_epoch_losses, _ = train(deepfm)

# 6. 评估xDeepFM
xdeepfm = xDeepFM(feature_fields=fields, embed_dim=8, mlp_dims=(48,24), dropout=0.25, cross_layer_sizes=(24,16), split_half=True)
print('Training xDeepFM...')
_, xdeepfm_val_epoch_losses, _ = train(xdeepfm)

# 6. 评估FM
fm = FactorizationMachine(feature_fields=fields, embed_dim=8)
print('Training FM...')
_, fm_val_epoch_losses, _ = train(fm)

# 8. 测试
def test(model):
    model.eval()
    test_loss = []
    prediction = []
    y_true = []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch, (x, y) in enumerate(tqdm(val_loader, desc=f"Testing")):
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y.float().detach())
            test_loss.append(loss.item())
            prediction.extend(torch.sigmoid(pred).tolist())
            y_true.extend(y.tolist())
    test_auc = roc_auc_score(y_true=y_true, y_score=prediction)
    print(f"Test loss : {sum(test_loss)/len(test_loss):.5f}   Test auc is {test_auc:.5f}")
# 测试DeepFM
print('Testing DeepFM...')
test(deepfm)
# 测试xDeepFM
print('Testing xDeepFM...')
test(xdeepfm)
# 测试FM
print('Testing FM...')
test(fm)

# 9. 保存模型
torch.save(deepfm.state_dict(), 'deepfm.pth')
torch.save(xdeepfm.state_dict(), 'xdeepfm.pth')
torch.save(fm.state_dict(), 'fm.pth')

# 10. 绘制验证损失曲线
plt.figure(figsize=(12, 7))

# 确保有数据点才绘制
if deepfm_val_epoch_losses:
    epochs_deepfm = range(1, len(deepfm_val_epoch_losses) + 1)
    plt.plot(epochs_deepfm, deepfm_val_epoch_losses, label='DeepFM Validation Loss', marker='o')

if xdeepfm_val_epoch_losses:
    epochs_xdeepfm = range(1, len(xdeepfm_val_epoch_losses) + 1)
    plt.plot(epochs_xdeepfm, xdeepfm_val_epoch_losses, label='xDeepFM Validation Loss', marker='s')

if fm_val_epoch_losses:
    epochs_fm = range(1, len(fm_val_epoch_losses) + 1)
    plt.plot(epochs_fm, fm_val_epoch_losses, label='FM Validation Loss', marker='^')

plt.title('Model Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')

# 设置x轴刻度为整数
max_epochs = 0
if deepfm_val_epoch_losses:
    max_epochs = max(max_epochs, len(deepfm_val_epoch_losses))
if xdeepfm_val_epoch_losses:
    max_epochs = max(max_epochs, len(xdeepfm_val_epoch_losses))
if fm_val_epoch_losses:
    max_epochs = max(max_epochs, len(fm_val_epoch_losses))

if max_epochs > 0:
    plt.xticks(ticks=range(1, max_epochs + 1))

plt.legend()
plt.grid(True)
plt.tight_layout()
plot_filename = 'validation_loss_comparison.png'
plt.savefig(plot_filename)
print(f"Validation loss comparison plot saved to {plot_filename}")
# plt.show() # 取消注释以在支持的环境中显示绘图
