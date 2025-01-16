import pandas as pd
import numpy as np

# S-Thd
# csv转换
data = pd.read_csv('ddCD_RH_P2_LH_P3_S.csv')  # 导入csv表格
# 转换为mat文件的功能可以用保存为pkl（pickle文件）代替，或者直接保存为csv，MATLAB与Python都支持读取csv
data.to_pickle('ddCD_RH_P2_LH_P3_S.pkl')  # 转换为pkl文件
data = pd.read_pickle('ddCD_RH_P2_LH_P3_S.pkl')
D = data
wavelength = D['Wavelength']  # 通过列名称获取数据

# 原始数据
RH_S_P2 = D['RH_S_P2']
RH_DMSO_P2 = D['RH_DMSO_P2']
LH_S_P3 = D['LH_S_P3']
LH_DMSO_P3 = D['LH_DMSO_P3']

# 对每个光谱都二范数，再取ddW
# Step 2: 对波长范围 800-1000 nm 进行二范数计算
valid_indices = (wavelength >= 800) & (wavelength <= 1000)
RH_S_P2_filtered = RH_S_P2[valid_indices]
RH_DMSO_P2_filtered = RH_DMSO_P2[valid_indices]
LH_S_P3_filtered = LH_S_P3[valid_indices]
LH_DMSO_P3_filtered = LH_DMSO_P3[valid_indices]

L2_RH_S_P2 = np.linalg.norm(RH_S_P2_filtered, 2)
L2_RH_DMSO_P2 = np.linalg.norm(RH_DMSO_P2_filtered, 2)
L2_LH_S_P3 = np.linalg.norm(LH_S_P3_filtered, 2)
L2_LH_DMSO_P3 = np.linalg.norm(LH_DMSO_P3_filtered, 2)

ddW_all_L2_S = -(L2_RH_S_P2 - L2_RH_DMSO_P2) + (L2_LH_S_P3 - L2_LH_DMSO_P3)
print(f'S-Thd——对每个光谱都二范数，再取ddW: {ddW_all_L2_S:.4f}')
print(f'S-Thd——对每个光谱都二范数，再取dW_RH: {L2_RH_S_P2 - L2_RH_DMSO_P2:.4f}')
print(f'S-Thd——对每个光谱都二范数，再取dW_LH: {L2_LH_S_P3 - L2_LH_DMSO_P3:.4f}')

# R-Thd
# csv转换
data = pd.read_csv('ddCD_RH_P3_LH_P1_R.csv')  # 导入csv表格
data.to_pickle('ddCD_RH_P3_LH_P1_R.pkl')  # 转换为pkl文件
data = pd.read_pickle('ddCD_RH_P3_LH_P1_R.pkl')
D = data
wavelength = D['Wavelength']  # 通过列名称获取数据

# 原始数据
RH_R_P3 = D['RH_R_P3']
RH_DMSO_P3 = D['RH_DMSO_P3']
LH_R_P1 = D['LH_R_P1']
LH_DMSO_P1 = D['LH_DMSOimport pandas as pd
import numpy as np

# S-Thd
# csv转换
data = pd.read_csv('ddCD_RH_P2_LH_P3_S.csv')  # 导入csv表格
# 转换为mat文件的功能可以用保存为pkl（pickle文件）代替，或者直接保存为csv，MATLAB与Python都支持读取csv
data.to_pickle('ddCD_RH_P2_LH_P3_S.pkl')  # 转换为pkl文件
data = pd.read_pickle('ddCD_RH_P2_LH_P3_S.pkl')
D = data
wavelength = D['Wavelength']  # 通过列名称获取数据

# 原始数据
RH_S_P2 = D['RH_S_P2']
RH_DMSO_P2 = D['RH_DMSO_P2']
LH_S_P3 = D['LH_S_P3']
LH_DMSO_P3 = D['LH_DMSO_P3']

# 对每个光谱都二范数，再取ddW
# Step 2: 对波长范围 800-1000 nm 进行二范数计算
valid_indices = (wavelength >= 800) & (wavelength <= 1000)
RH_S_P2_filtered = RH_S_P2[valid_indices]
RH_DMSO_P2_filtered = RH_DMSO_P2[valid_indices]
LH_S_P3_filtered = LH_S_P3[valid_indices]
LH_DMSO_P3_filtered = LH_DMSO_P3[valid_indices]

L2_RH_S_P2 = np.linalg.norm(RH_S_P2_filtered, 2)
L2_RH_DMSO_P2 = np.linalg.norm(RH_DMSO_P2_filtered, 2)
L2_LH_S_P3 = np.linalg.norm(LH_S_P3_filtered, 2)
L2_LH_DMSO_P3 = np.linalg.norm(LH_DMSO_P3_filtered, 2)

ddW_all_L2_S = -(L2_RH_S_P2 - L2_RH_DMSO_P2) + (L2_LH_S_P3 - L2_LH_DMSO_P3)
print(f'S-Thd——对每个光谱都二范数，再取ddW: {ddW_all_L2_S:.4f}')
print(f'S-Thd——对每个光谱都二范数，再取dW_RH: {L2_RH_S_P2 - L2_RH_DMSO_P2:.4f}')
print(f'S-Thd——对每个光谱都二范数，再取dW_LH: {L2_LH_S_P3 - L2_LH_DMSO_P3:.4f}')

# R-Thd
# csv转换
data = pd.read_csv('ddCD_RH_P3_LH_P1_R.csv')  # 导入csv表格
data.to_pickle('ddCD_RH_P3_LH_P1_R.pkl')  # 转换为pkl文件
data = pd.read_pickle('ddCD_RH_P3_LH_P1_R.pkl')
D = data
wavelength = D['Wavelength']  # 通过列名称获取数据

# 原始数据
RH_R_P3 = D['RH_R_P3']
RH_DMSO_P3 = D['RH_DMSO_P3']
LH_R_P1 = D['LH_R_P1']
LH_DMSO_P1 = D['LH_DMSO_P1']

# 对每个光谱都二范数，再取ddW
# Step 2: 对波长范围 800-1000 nm 进行二范数计算
valid_indices = (wavelength >= 800) & (wavelength <= 1000)
RH_S_P2_filtered = RH_R_P3[valid_indices]
RH_DMSO_P2_filtered = RH_DMSO_P3[valid_indices]
LH_S_P3_filtered = LH_R_P1[valid_indices]
LH_DMSO_P3_filtered = LH_DMSO_P1[valid_indices]

L2_RH_S_P2 = np.linalg.norm(RH_S_P2_filtered, 2)
L2_RH_DMSO_P2 = np.linalg.norm(RH_DMSO_P2_filtered, 2)
L2_LH_S_P3 = np.linalg.norm(LH_S_P3_filtered, 2)
L2_LH_DMSO_P3 = np.linalg.norm(LH_DMSO_P3_filtered, 2)

ddW_all_L2_R = -(L2_RH_S_P2 - L2_RH_DMSO_P2) + (L2_LH_S_P3 - L2_LH_DMSO_P3)
print(f'R-Thd——对每个光谱都二范数，再取ddW: {ddW_all_L2_R:.4f}')
print(f'R-Thd——对每个光谱都二范数，再取dW_RH: {L2_RH_S_P2 - L2_RH_DMSO_P2:.4f}')
print(f'R-Thd——对每个光谱都二范数，再取dW_LH: {L2_LH_S_P3 - L2_LH_DMSO_P3:.4f}')_P1']

# 对每个光谱都二范数，再取ddW
# Step 2: 对波长范围 800-1000 nm 进行二范数计算
valid_indices = (wavelength >= 800) & (wavelength <= 1000)
RH_S_P2_filtered = RH_R_P3[valid_indices]
RH_DMSO_P2_filtered = RH_DMSO_P3[valid_indices]
LH_S_P3_filtered = LH_R_P1[valid_indices]
LH_DMSO_P3_filtered = LH_DMSO_P1[valid_indices]

L2_RH_S_P2 = np.linalg.norm(RH_S_P2_filtered, 2)
L2_RH_DMSO_P2 = np.linalg.norm(RH_DMSO_P2_filtered, 2)
L2_LH_S_P3 = np.linalg.norm(LH_S_P3_filtered, 2)
L2_LH_DMSO_P3 = np.linalg.norm(LH_DMSO_P3_filtered, 2)

ddW_all_L2_R = -(L2_RH_S_P2 - L2_RH_DMSO_P2) + (L2_LH_S_P3 - L2_LH_DMSO_P3)
print(f'R-Thd——对每个光谱都二范数，再取ddW: {ddW_all_L2_R:.4f}')
print(f'R-Thd——对每个光谱都二范数，再取dW_RH: {L2_RH_S_P2 - L2_RH_DMSO_P2:.4f}')
print(f'R-Thd——对每个光谱都二范数，再取dW_LH: {L2_LH_S_P3 - L2_LH_DMSO_P3:.4f}')