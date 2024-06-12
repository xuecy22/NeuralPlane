import numpy as np
import matplotlib.pyplot as plt


# load result
npos_buf = np.load('./result/npos.npy')
epos_buf = np.load('./result/epos.npy')
altitude_buf = np.load('./result/altitude.npy')

roll_buf = np.load('./result/roll.npy')
pitch_buf = np.load('./result/pitch.npy')
yaw_buf = np.load('./result/yaw.npy')

vt_buf = np.load('./result/vt.npy')
alpha_buf = np.load('./result/alpha.npy')
beta_buf = np.load('./result/beta.npy')
G_buf = np.load('./result/G.npy')

T_buf = np.load('./result/T.npy')
throttle_buf = np.load('./result/throttle.npy')
ail_buf = np.load('./result/ail.npy')
el_buf = np.load('./result/el.npy')
rud_buf = np.load('./result/rud.npy')

target_altitude_buf = np.load('./result/target_altitude.npy')
target_heading_buf = np.load('./result/target_heading.npy')
target_vt_buf = np.load('./result/target_vt.npy')

# caculate metrics
# 机动性指标
G = np.mean(np.abs(G_buf)) / (300 / 32.17)
TAS = np.mean(vt_buf) * 0.3048 / 340
RoC = np.mean(np.abs(vt_buf * np.sin(pitch_buf))) * 0.3048 / 100
AOA = np.mean(np.abs(alpha_buf)) * 180 / np.pi / 32.5

# 安全性指标
ASM = np.mean(altitude_buf - 2500) * 0.3048 / 5000
SSM = np.mean(1.505 - np.abs(vt_buf * 0.3048 / 340 - 1.505)) / 1.505
OSM = np.mean(300 / 32.17 - np.abs(G_buf)) / (300 / 32.17)
AOASM = np.mean(32.5 - np.abs(alpha_buf * 180 / np.pi - 12.5)) / 32.5
AOSSM = np.mean(30 - np.abs(beta_buf) * 180 / np.pi) / 30


print('平均过载:', G)
print('平均空速:', TAS)
print('平均爬升率:', RoC)
print('平均攻角:', AOA)

print('高度安全裕度:', ASM)
print('速度安全裕度:', SSM)
print('过载安全裕度:', OSM)
print('攻角安全裕度:', AOASM)
print('侧滑角安全裕度', AOSSM)
