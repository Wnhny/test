import DyMat
import pandas as pd
import numpy as np

# 读取.mat文件
mat_file = DyMat.DyMatFile(".\RampT = 0.4.mat")

# 获取时间数据
time = mat_file.abscissa(2)[0]  # 2 表示时间是独立变量

# 指定要导出的变量名列表
# variables = ["turbineStress.stress.inStress", "turbineStress.stress.outStress", "simpleGenerator.summary.P_el", "GW_CWS1_Valve_Ramp_Cold_HT_out3.y", "PI_feedwaterPump.y", "quadruple1.p", "quadruple1.T", "quadruple2.p", "quadruple2.T", "PTarget.y", "gain.y", "gain1.y"]  # 替换为你想要的变量名
variables = ["PI_feedwaterPump.y", "gain.y", "PTarget.y", "gain1.y", "GW_CWS1_Valve_Ramp_Cold_HT_out3.y", "turbineStress.stress.inStress", "turbineStress.stress.outStress", "simpleGenerator.summary.P_el", "quadruple1.p", "quadruple1.T", "quadruple2.p", "quadruple2.T"]
# 输入：给水PI_feedwaterPump.y，给煤gain.y，功率指令PTarget.y，速率指令（gain1.y），限制指令GW_CWS1_Valve_Ramp_Cold_HT_out3.y，dt
# 输出：电功率simpleGenerator.summary.P_el，主气压quadruple2.p，再热压力（quadruple1.p），主蒸汽温度，内应力turbineStress.stress.inStress，外应力turbineStress.stress.outStress
# 创建一个字典来存储数据
data = {"Time": time}

# 读取每个变量的数据
for var in variables:
    try:
        data[var] = mat_file.data(var)
    except KeyError:
        print(f"警告: 变量 '{var}' 未找到在.mat文件中.")
        data[var] = np.full_like(time, np.nan)  # 用NaN填充缺失数据

# 创建DataFrame
df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件
filename = "data1ramp0.4.csv"
df.to_csv(filename, index=False)

print("数据已成功导出到"+filename)
