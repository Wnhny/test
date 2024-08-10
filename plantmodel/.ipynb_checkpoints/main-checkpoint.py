import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1064881791906100)
print(torch.initial_seed())
device = torch.device("cuda:0")


def inverse_standardize(tensor, mean, std):
    return tensor * std + mean


# 反标准化

# DataLoader部分
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.mean = self.data.iloc[:, 1:6].mean()
        self.std = self.data.iloc[:, 1:6].std()
        self.mean_t0 = self.data.iloc[:, 0].mean()
        self.std_t0 = self.data.iloc[:, 0].std()
        self.data.iloc[:, 1:6] = (self.data.iloc[:, 1:6] - self.mean) / self.std

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        tm_values = self.data.iloc[idx + 1, 1:6].values
        # tm_values = tm_values.astype(np.float32)
        t0_value = self.data.iloc[idx, 0]

        # 检查数据类型并尝试转换
        try:
            tm = torch.tensor(tm_values, dtype=torch.float32, device=device)
            t0 = torch.tensor(t0_value, dtype=torch.float32, device=device)
        except TypeError:
            raise ValueError(f"Error converting data at index {idx}. tm_values: {tm_values}, t0_value: {t0_value}")

        return tm, t0


# 定义物理层
class PhysicalLayer(nn.Module):
    def __init__(self):
        super(PhysicalLayer, self).__init__()
        self.G = nn.Parameter(torch.randn(5))
        self.Phi = nn.Parameter(torch.randn(5))
        self.tau = nn.Parameter(torch.tensor(0.01))

    def forward(self, theta_m, tm):
        result2 = torch.matmul(tm, self.Phi) * self.tau
        t1 = theta_m + result2
        # 末尾应该补充上上一步的预测输出,此处为公式中第二项
        return t1



class T_RNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=5, num_layers=1):
        super(T_RNN, self).__init__()
        self.gru_T = nn.GRU(input_size=5, hidden_size=5, num_layers=1, bidirectional=False, bias=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)# 一个RNN,为了使得
        # self.fc = nn.Sequential(nn.Linear(5, 10), nn.Tanh(), nn.Linear(10, 5))  # 一个全连接层
        self.linear_1 = nn.Linear(5,5)

    def forward(self, x, amount=0):
        # Calculations for theta_m
        output, hn = self.gru_T(x)
        h = hn[-1]
       # T = torch.Tensor(T)
        T = torch.cat(h.split(1), dim=-1).squeeze(0)
        T = self.linear_1(T)
        # 惯性环节求和的同时加入了公式中的第三项

        return T


class InertialRNN(nn.Module):
    def __init__(self):
        super(InertialRNN, self).__init__()
        self.G = nn.Parameter(torch.randn(5))
        self.Phi = nn.Parameter(torch.randn(5))
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.K = nn.Parameter(torch.randn(5, 5))
        self.T = nn.Parameter(torch.randn(5))
        self.physical_layer = PhysicalLayer()
        self.h_0 = torch.Tensor(5,10)
        self.Y1 = [torch.Tensor(5).to(device)]
        self.T_nn = T_RNN()

    def forward(self, x):
        # Calculations for theta_m
        theta_m = torch.matmul(self.physical_layer.G , x)
        # self.T, h1= self.gru(self.Y1[amount],self.h_0)
        # 计算矩阵theta_m
        # Calculations moved from PhysicalLayer
        T_out = self.T_nn(x)
        exp_term = 1 - torch.exp(-self.physical_layer.tau / T_out)
        # 计算矩阵中的exp项
        # intermediate_product = torch.matmul(theta_m, self.K)
        out = torch.matmul(theta_m , self.K)
        out = torch.matmul(out,exp_term)
        # 惯性环节量K，Tau，系数，计算矩阵中第一项
        # RNN part
        # theta_m_processed = theta_m_processed.unsqueeze(1)
        # out, _ = self.rnn(theta_m_processed)
        # out = self.fc(out.squeeze(1))
        # 导入RNN+全连接层
        # 导入源项
        output = self.physical_layer(out, x)




        # 惯性环节求和的同时加入了公式中的第三项

        return output

    # def get_params(vocab_size, num_hiddens, device):
    #     num_inputs = num_outputs = vocab_size
    #
    #     def normal(shape):
    #         return torch.randn(size=shape, device=device) * 0.01
    #
    #     def three():
    #         return (normal((num_inputs, num_hiddens)),
    #                 normal((num_hiddens, num_hiddens)),
    #                 torch.zeros(num_hiddens, device=device))
    #
    #     W_xz, W_hz, b_z = three()  # 更新门参数
    #     W_xr, W_hr, b_r = three()  # 重置门参数
    #     W_xh, W_hh, b_h = three()  # 候选隐状态参数
    #     # 输出层参数
    #     W_hq = normal((num_hiddens, num_outputs))
    #     b_q = torch.zeros(num_outputs, device=device)
    #     # 附加梯度
    #     params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    #     for param in params:
    #         param.requires_grad_(True)
    #     return params
    #
    # def init_gru_state(batch_size, num_hiddens, device):
    #     return (np.zeros(shape=(batch_size, num_hiddens), ctx=device),)
    #
    # def gru(inputs, state, params):
    #     W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    #     H, = state
    #     outputs = []
    #     for X in inputs:
    #         Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
    #         R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
    #         H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
    #         H = Z * H + (1 - Z) * H_tilda
    #         Y = np.dot(H, W_hq) + b_q
    #         outputs.append(Y)
    #     return np.concatenate(outputs, axis=0), (H,)


# 数据加载
dataset = CustomDataset('result3.csv')
dataloader = DataLoader(dataset, batch_size=5, shuffle=False, drop_last=True)

# 数据划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 使用slice而不是random_split
train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, drop_last=True)

# Unit
model = InertialRNN()
print(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam
criterion = nn.MSELoss()

# 学习率动态调整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# Early stopping
n_epochs_stop = 10
epochs_no_improve = 0
min_val_loss = float('inf')

predicted_vals = []
actual_vals = []

for epoch in range(50):  # max epoch
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        tm, t0 = data
        optimizer.zero_grad()
        outputs = model(tm)
        # 反标准化预测的输出
        outputs_inv_std = inverse_standardize(outputs, dataset.mean[0], dataset.std[0])
        # t0_inv_std = inverse_standardize(t0, dataset.mean_t0, dataset.std_t0)

        loss = criterion(outputs_inv_std, t0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()



    # Validation loss,Maybe it is ok
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(tm)
            # data取的不对，这不是x——》Y的数据集

            # 反标准化预测的输出
            outputs_inv_std = inverse_standardize(outputs, dataset.mean[0], dataset.std[0])
            # t0_inv_std = inverse_standardize(t0, dataset.mean_t0, dataset.std_t0)
            # t0也是标准化后的，应该反标准化
            # 更改了初始化部分，t0未标准化
            loss = criterion(outputs_inv_std, t0)
            val_loss += loss.item()

            # # 预测与实际，方便出图
            # predicted_vals = outputs_inv_std
            # actual_vals = t0_inv_std
            #
        # if epoch == 9:
        #     predicted_vals.extend(outputs_inv_std.cpu().numpy().tolist())
        #     actual_vals.extend(t0.cpu().numpy().tolist())

    scheduler.step(val_loss)
    InertialRNN.Y1 = [torch.Tensor(5)]
    # 早停条件
    if val_loss < min_val_loss:
        epochs_no_improve = 0
        min_val_loss = val_loss
    else:
        epochs_no_improve += 1
    if epochs_no_improve == n_epochs_stop:
        print("Early stopping!")
        break
    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")

for i, data in enumerate(train_loader, 0):
    tm, t0 = data

    # 反标准化预测的输出
    outputs = model(tm)
    outputs_inv_std = inverse_standardize(outputs, dataset.mean[0], dataset.std[0])
    # t0_inv_std = inverse_standardize(t0, dataset.mean_t0, dataset.std_t0)

    predicted_vals.extend(outputs_inv_std.detach().cpu().numpy().tolist())
    actual_vals.extend(t0.cpu().detach().numpy().tolist())
plt.figure(figsize=(12, 6))
plt.plot(predicted_vals, label='Predicted')
plt.plot(actual_vals, label='Actual t0')
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
