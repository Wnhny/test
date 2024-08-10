import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inverse_standardize(tensor, mean, std):
    return tensor * std + mean


# 反标准化

# DataLoader部分
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = pd.read_csv('data1ramp0.4dt.csv')
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        # 分离输入和输出
        X = self.data.iloc[:, :5].values
        Y = self.data.iloc[:, 5:12].values

        # 标准化
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.Y_scaled = self.scaler_Y.fit_transform(Y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.X_scaled[idx]
        y = self.Y_scaled[idx]
        x = x.reshape(1, -1)  # 重塑为 (1, 5)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_scaler(self):
        return self.scaler_X, self.scaler_Y


class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=7, num_layers=2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层，用于生成输出
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU前向传播
        out, _ = self.gru(x, h0)

        # 我们只需要最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        input_scaled = custo.scaler_X.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        output_scaled = model(input_tensor)
        output = scaler_Y.inverse_transform(output_scaled.cpu().numpy())
    return output

# 创建模型实例
model = GRUModel(input_size=5, hidden_size=64, output_size=7, num_layers=2).to(device)


# 打印模型结构
print(model)


# 数据加载
dataset = CustomDataset('data1ramp0.4dt.csv')
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

# 数据划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 使用slice而不是random_split
train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)

# Unit
# model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam
criterion = nn.MSELoss()

# 学习率动态调整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
num_epochs = 10
# Early stopping
n_epochs_stop = 10
epochs_no_improve = 0
min_val_loss = float('inf')

predicted_vals = []
actual_vals = []

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 确保输入形状正确：(batch_size, sequence_length, input_size)
        # 在这里，sequence_length = 1
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')


    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        average_loss = total_loss / len(val_loader)
        print(f'Test Loss: {average_loss:.4f}')

# scaler = dataset.

predictions = []
for inputs, _ in dataloader:
    inputs = inputs.to(device)
    outputs = model(inputs)
    # 将预测结果转移到 CPU 并转换为 NumPy 数组
    outputs = outputs.detach().cpu().numpy()
    # 反标准化
    original_scale_predictions = dataset.scaler_Y.inverse_transform(outputs)
    predictions.extend(original_scale_predictions)

plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predicted')
# plt.plot(actual_vals, label='Actual t0')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.title('Comparison of Predicted Results and Actual t0 Values')
plt.show()

# plt.figure(figsize=(12,6))
# plt.plot(predict.detach().cpu().numpy(), label='Predicted')
# plt.plot(actual_vals, label='Actual t0')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Comparison of Predicted Results and Actual t0 Values')
# plt.show()
