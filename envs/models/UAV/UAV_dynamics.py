import torch
import torch.nn as nn

class UAVDynamics(nn.Module):
    def __init__(self):
        super().__init__()
    
    def compute_extended_state(self, x):
        return self.nlplant(x)

    def forward(self, t, x):
        es = self.compute_extended_state(x)
        return es
    
    def nlplant(self, x):
        """
        model state(dim 12):
            0. ego_north_position      (unit: feet)
            1. ego_east_position       (unit: feet)
            2. ego_altitude            (unit: feet)
            3. ego_roll                (unit: rad)
            4. ego_pitch               (unit: rad)
            5. ego_yaw                 (unit: rad)
            6. ego_u                   (unit: feet/s)
            7. ego_v                   (unit: feet/s)
            8. ego_w                   (unit: feet/s)
            9. ego_P                   (unit: rad/s)
            10. ego_Q                  (unit: rad/s)
            11. ego_R                  (unit: rad/s)

        model control(dim 3)
            0. ego_Fx                  (unit: lbf)
            1. ego_Fy                  (unit: lbf)
            2. ego_Fz                  (unit: lbf)
        """
        UAV_M = 300                       
        UAV_FLAY_RANGE = 30            
        M, N, L_bar = 1.0, 1.0, 1.0   
        I_x, I_y, I_z, I_xz = 1.0, 1.0, 1.0, 0
        g = 9.81

        xdot = torch.zeros_like(x)
        alt = x[:, 2]
        phi = x[:, 3]
        theta = x[:, 4]
        psi = x[:, 5]
        U = x[:, 6]
        V = x[:, 7]
        W = x[:, 8]
        P = x[:, 9]
        Q = x[:, 10]
        R = x[:, 11]
        F_x = x[:, 12]
        F_y = x[:, 13]
        F_z = x[:, 14]

        st = torch.sin(theta)
        ct = torch.cos(theta)
        tt = torch.tan(theta)
        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        spsi = torch.sin(psi)
        cpsi = torch.cos(psi)
        # 计算状态量的导数
        xdot[:, 0] = U * (ct * cpsi) + V * (sphi * st * cpsi - cphi * spsi) + W * (sphi * spsi + cphi * st * cpsi)
        xdot[:, 1] = U * (ct * spsi) + V * (sphi * st * spsi + cphi * cpsi) + W * (-sphi * cpsi + cphi * st * spsi)
        xdot[:, 2] = U * st - V * (sphi * ct) - W * (cphi * ct)

        xdot[:, 3] = P + (R * cphi + Q * sphi) * tt
        xdot[:, 4] = Q * cphi - R * sphi
        xdot[:, 5] = (R * cphi + Q * sphi) / ct

        xdot[:, 6] = V * R - W * Q - g * st + F_x / UAV_M
        xdot[:, 7] = -U * R + W * P  + g * ct * sphi + F_y / UAV_M
        xdot[:, 8] = U * Q - V * P  + g * ct * cphi + F_z / UAV_M

        b0 = L_bar - Q * R * (I_z - I_y) + P * Q * I_xz
        b1 = N - P * Q * (I_y - I_x) - Q * R * I_xz
        b2 = M - P * R * (I_x - I_z) - (P ** 2 - R ** 2) * I_xz
        xdot[:, 9] = (b0 * I_z + b1 * I_xz) / (I_z * I_x - I_xz ** 2)
        xdot[:, 10] = b2 / I_y
        xdot[:, 11] = (b0 * I_xz + b1 * I_x) / (I_z * I_x - I_xz ** 2)

        return xdot
