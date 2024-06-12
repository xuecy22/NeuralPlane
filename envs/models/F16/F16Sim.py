# from envs.models.F16Model import F16Model
# import numpy as np
# import torch
# import time

# NrStates = 12
# NrControls = 5
# n = 10
# Dynamics = F16Model(n)

# feet_to_m = 0.3048
# lbf_to_N = 4.448222
# lb_ft2_to_Pa = 47.880258888889

# device = "cuda:0"

# xu_IU_to_SI = [1.0] * (NrStates + NrControls)
# xdot_IU_to_SI = [1.0] * NrStates


# def Init_xu_IU_to_SI():
#     for i in range(3):
#         xu_IU_to_SI[i] = feet_to_m
#     xu_IU_to_SI[6] = feet_to_m
#     xu_IU_to_SI[12] = lbf_to_N


# def Init_xdot_IU_to_SI():
#     for i in range(NrStates):
#         xdot_IU_to_SI[i] = xu_IU_to_SI[i]
#     # xdot_IU_to_SI[16] = lb_ft2_to_Pa
#     # xdot_IU_to_SI[17] = lb_ft2_to_Pa


# def InitState():
#     Init_xu_IU_to_SI()
#     Init_xdot_IU_to_SI()


# def Convert_xu_IU_to_SI(xu):
#     for i in range(NrStates):
#         xu[:, i] *= xu_IU_to_SI[i]


# def Convert_xu_SI_to_IU(xu):
#     for i in range(NrStates):
#         xu[:, i] /= xu_IU_to_SI[i]


# def Convert_xdot_IU_to_SI(xdot):
#     for i in range(NrStates):
#         xdot[:, i] *= xdot_IU_to_SI[i]


# def Convert_xdot_SI_to_IU(xdot):
#     for i in range(NrStates):
#         xdot[:, i] /= xdot_IU_to_SI[i]


# def Convert_IU_to_SI(xu, xdot):
#     Convert_xu_IU_to_SI(xu)
#     Convert_xdot_IU_to_SI(xdot)


# def Convert_SI_to_IU(xu, xdot):
#     Convert_xu_SI_to_IU(xu)
#     Convert_xdot_SI_to_IU(xdot)


# def UpdateSimulation(xu):
#     Dynamics.update(xu[:, NrStates:])
#     xdot = Dynamics.nlplant_(xu[:, :NrStates])
#     return xdot


# def UpdateSimulation_plus(xu):
#     Convert_xu_SI_to_IU(xu)
#     xdot = UpdateSimulation(xu)
#     return xdot


# def main(input):
#     InitState()
#     xdot = UpdateSimulation_plus(input)
#     return xdot


# if __name__ == '__main__':
#     import time
#     xu = [14.3842921301, 0.0, 999.240528869, 0.0, 0.0680626236787, 0.0, 100.08096494, 0.121545455798, 0.0,
#                      0.0, -0.031583522788, 0.0, 20000.0, 0.0, 0.0, 0.0, 0.0]
#     xu = np.array(xu).reshape(1, -1)
#     input = torch.tensor(xu, device=torch.device(device), dtype=torch.float32)
#     input = input.repeat(n, 1)
#     start_time = time.time()
#     xdot = main(input)
#     # Convert_IU_to_SI(input, xdot)
#     print(xdot)
#     print(time.time() - start_time)
