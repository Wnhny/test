# 假设这是你的预测脚本 predict.py

import torch
import torch.nn as nn
import joblib

# 定义相同的模型结构
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # 定义与训练时相同的模型结构
        ...

    def forward(self, x):
        # 定义前向传播
        ...

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 如果你保存了整个模型
model = torch.load('full_model.pth', map_location=device)

# 或者如果你只保存了模型参数
model = YourModel()
model.load_state_dict(torch.load('model_params.pth', map_location=device))
model.to(device)
model.eval()  # 设置为评估模式

# 加载 scaler（如果你保存了的话）
scaler = joblib.load('scaler.pkl')

# 使用模型进行预测
def predict(model, input_data, scaler):
    # 确保输入数据的形状正确
    if input_data.dim() == 2:
        input_data = input_data.unsqueeze(1)
    
    with torch.no_grad():
        output = model(input_data)
    
    # 如果需要，进行反标准化
    if scaler:
        output = scaler.inverse_transform(output.cpu().numpy())
    
    return output

# 示例预测
input_data = torch.randn(1, 1, 10).to(device)  # 假设输入是 (batch_size, sequence_length, input_features)
prediction = predict(model, input_data, scaler)
print("Prediction:", prediction)
