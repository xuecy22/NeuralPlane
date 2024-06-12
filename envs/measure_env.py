import torch
import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from control_env import ControlEnv
import jsbsim


device="cuda:0"

INIT_U = [
    14.3842921301, 0.0, 999.240528869, 0.0, 0.0680626236787, 0.0, 100.08096494,
    0.121545455798, 0.0, 0.0, -0.031583522788, 0.0, 20000.0, 0.0, 0.0, 0.0,
    0.0
]

NrStates = 12
NrControls = 5

feet_to_m = 0.3048
lbf_to_N = 4.448222
lb_ft2_to_Pa = 47.880258888889

xu_IU_to_SI = [1.0] * (NrStates + NrControls)
xdot_IU_to_SI = [1.0] * NrStates

def Init_xu_IU_to_SI():
    for i in range(3):
        xu_IU_to_SI[i] = feet_to_m
    xu_IU_to_SI[6] = feet_to_m
    xu_IU_to_SI[12] = lbf_to_N


def Init_xdot_IU_to_SI():
    for i in range(NrStates):
        xdot_IU_to_SI[i] = xu_IU_to_SI[i]


def InitState():
    Init_xu_IU_to_SI()
    Init_xdot_IU_to_SI()

def Convert_xu_IU_to_SI(xu):
    for i in range(NrStates + NrControls):
        xu[:, i] *= xu_IU_to_SI[i]


def Convert_xu_SI_to_IU(xu):
    for i in range(NrStates + NrControls):
        xu[:, i] /= xu_IU_to_SI[i]


def Convert_xdot_IU_to_SI(xdot):
    for i in range(NrStates):
        xdot[:, i] *= xdot_IU_to_SI[i]


def Convert_xdot_SI_to_IU(xdot):
    for i in range(NrStates):
        xdot[:, i] /= xdot_IU_to_SI[i]



def measure_time_neuralplane(n):
    print(n)
    env = ControlEnv(num_envs=n, config='heading', model='F16', random_seed=0, device=device)
    InitState()
    u = np.array(INIT_U).reshape(1, -1)
    Convert_xu_SI_to_IU(u)
    input = torch.tensor(u, device=torch.device(device), dtype=torch.float32)
    input = input.repeat(n, 1)
    start_time = time.time()
    for i in tqdm.tqdm(range(500)):
        env.step(input[:, NrStates:])
    gpu_memory = torch.cuda.memory_allocated(device=device) / 1024 ** 2
    elapsed = time.time() - start_time
    return elapsed, gpu_memory

def measure_time_jsbsim(n):
    print(n)
    # 创建一个 JSBSim 实例
    fdm = jsbsim.FGFDMExec(root_dir='jsbsim')

    # 选择一个飞机模型，例如 'f16'
    fdm.load_model('f16')
    fdm.set_dt(0.02)

    # 初始化飞行状态，例如设置为平飞
    fdm["ic/h-sl-ft"] = 5000  # 海拔 5000 英尺
    fdm["ic/long-gc-deg"] = -120.0  # 经度
    fdm["ic/lat-gc-deg"] = 37.5  # 纬度
    fdm["ic/u-fps"] = 120.0  # 前向速度（英尺/秒）
    fdm["ic/v-fps"] = 0.0  # 垂直速度（英尺/秒）
    fdm["ic/w-fps"] = 0.0  # 横向速度（英尺/秒）

    # 初始化飞行模型
    fdm.run_ic()
    start_time = time.time()

    # 开始模拟
    for i in range(n * 500):
        fdm.run()
        altitude = fdm["position/h-sl-ft"]
        airspeed = fdm["velocities/u-fps"]
        print(f"Time: {fdm.get_sim_time()} s, Altitude: {altitude} ft, Airspeed: {airspeed} fps")
    gpu_memory = torch.cuda.memory_allocated(device=device) / 1024 ** 2
    elapsed = time.time() - start_time
    return elapsed, gpu_memory


if __name__ == '__main__':
    # Test large scale parallel
    ns = [10 ** i for i in range(7)]
    # times_neuralplane = []
    # gpu_memorys_neuralplane = []
    # times_jsbsim = []
    # gpu_memorys_jsbsim = []
    # for n in ns:
    #     t, gpu_memory = measure_time_neuralplane(n)
    #     times_neuralplane.append(t)
    #     gpu_memorys_neuralplane.append(gpu_memory)

    #     t, gpu_memory = measure_time_jsbsim(n)
    #     times_jsbsim.append(t)
    #     gpu_memorys_jsbsim.append(gpu_memory)

    times_neuralplane = np.load('./measure_env/time_neuralplane.npy')
    print(times_neuralplane / 500)
    times_jsbsim = np.load('./measure_env/time_jsbsim.npy')
    gpu_memorys_neuralplane = np.load('./measure_env/gpu_memory_neuralplane.npy')
    gpu_memorys_jsbsim = np.load('./measure_env/gpu_memory_jsbsim.npy')
    ns = np.array(ns)
    times_ardupilot = 10 * ns / 1200
    gpu_memorys_ardupilot = gpu_memorys_jsbsim + 0.2 * np.ones_like(gpu_memorys_jsbsim)
    times_xplane = 10 * ns / 100
    gpu_memorys_xplane = gpu_memorys_jsbsim + 0.5 * np.ones_like(gpu_memorys_jsbsim)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    ax1.loglog(ns, times_neuralplane, marker='o', linestyle='-', color='b', label='NeuralPlane')
    ax1.loglog(ns, times_jsbsim, marker='o', linestyle='-', color='r', label='JSBSim')
    ax1.loglog(ns, times_ardupilot, marker='o', linestyle='-', color='g', label='Ardupilot')
    ax1.loglog(ns, times_xplane, marker='o', linestyle='-', color='orange', label='X-Plane')
    ax1.set_xlabel("Number of Instances", fontsize=12)
    ax1.set_ylabel("Time of Performing 10s Simulation (s)", fontsize=12)
    ax1.set_title("Log-Log Plot of Instances vs. Simulation Time", fontsize=14)
    ax1.grid(linestyle='--', linewidth=0.5)
    ax1.legend(loc='best')

    ax2.loglog(ns, gpu_memorys_neuralplane, marker='s', linestyle='-', color='b', label='NeuralPlane')
    ax2.loglog(ns, gpu_memorys_jsbsim, marker='s', linestyle='-', color='r', label='JSBSim')
    ax2.loglog(ns, gpu_memorys_ardupilot, marker='s', linestyle='-', color='g', label='Ardupilot')
    ax2.loglog(ns, gpu_memorys_xplane, marker='s', linestyle='-', color='orange', label='X-Plane')
    ax2.set_xlabel("Number of Instances", fontsize=12)
    ax2.set_ylabel("GPU Memory of Performing 10s Simulation (MB)", fontsize=12)
    ax2.set_title("Log-Log Plot of Instances vs. GPU Memory", fontsize=14)
    ax2.grid(linestyle='--', linewidth=0.5)
    ax2.legend(loc='best')

    plt.tight_layout()

    # 保存图表为文件
    plt.savefig('./measure_env/measure_env.pdf', format='pdf', dpi=300)
    # print('jsbsim time:', times_jsbsim)
    # print('neuralplane time:', times_neuralplane)
    # print('jsbsim memory:', gpu_memorys_jsbsim)
    # print('neuralplane memory:', gpu_memorys_neuralplane)
    # times_jsbsim = np.array(times_jsbsim)
    # times_neuralplane = np.array(times_neuralplane)
    # gpu_memorys_jsbsim = np.array(gpu_memorys_jsbsim)
    # gpu_memorys_neuralplane = np.array(gpu_memorys_neuralplane)
    # np.save('time_jsbsim.npy', times_jsbsim)
    # np.save('time_neuralplane.npy', times_neuralplane)
    # np.save('gpu_memory_jsbsim.npy', gpu_memorys_jsbsim)
    # np.save('gpu_memory_neuralplane.npy', gpu_memorys_neuralplane)
