import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from hifi_F16_AeroData import hifi_F16
import copy


class F16Dynamics(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.hifi_F16 = hifi_F16(device=device)

    def compute_extended_state(self, x):
        return self.nlplant(x)

    def forward(self, t, x):
        es = self.compute_extended_state(x)
        return es
    
    def atmos(self, alt, vt):
        # 根据高度和速度计算动压、马赫数、静压
        rho0 = 2.377e-3
        tfac = 1 - .703e-5 * (alt)
        temp = 519.0 * tfac
        temp = (alt >= 35000.0) * 390 + (alt < 35000.0) * temp
        rho = rho0 * pow(tfac, 4.14)
        mach = (vt) / torch.sqrt(1.4 * 1716.3 * temp)
        qbar = .5 * rho * pow(vt, 2)
        ps = 1715.0 * rho * temp

        ps = (ps == 0) * 1715 + (ps != 0) * ps

        return (mach, qbar, ps)
    
    def nlplant(self, x):
        """
        model state(dim 12):
            0. ego_north_position      (unit: feet)
            1. ego_east_position       (unit: feet)
            2. ego_altitude            (unit: feet)
            3. ego_roll                (unit: rad)
            4. ego_pitch               (unit: rad)
            5. ego_yaw                 (unit: rad)
            6. ego_vt                  (unit: feet/s)
            7. ego_alpha               (unit: rad)
            8. ego_beta                (unit: rad)
            9. ego_P                   (unit: rad/s)
            10. ego_Q                  (unit: rad/s)
            11. ego_R                  (unit: rad/s)

        model control(dim 5)
            0. ego_T                  (unit: lbf)
            1. ego_el                 (unit: deg)
            2. ego_ail                (unit: deg)
            3. ego_rud                (unit: deg)
            4. ego_lef                (unit: deg)
        """
        xdot = torch.zeros_like(x)
        g = 32.17
        m = 636.94
        B = 30.0
        S = 300.0
        cbar = 11.32
        xcgr = 0.35
        xcg = 0.30
        Heng = 0.0
        pi = torch.pi

        Jy = 55814.0
        Jxz = 982.0
        Jz = 63100.0
        Jx = 9496.0

        r2d = 180.0 / pi

        # States
        alt = x[:, 2]
        phi = x[:, 3]
        theta = x[:, 4]
        psi = x[:, 5]

        vt = x[:, 6]
        alpha = x[:, 7] * r2d
        beta = x[:, 8] * r2d
        P = x[:, 9]
        Q = x[:, 10]
        R = x[:, 11]

        sa = torch.sin(x[:, 7])
        ca = torch.cos(x[:, 7])
        sb = torch.sin(x[:, 8])
        cb = torch.cos(x[:, 8])

        st = torch.sin(theta)
        ct = torch.cos(theta)
        tt = torch.tan(theta)
        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        spsi = torch.sin(psi)
        cpsi = torch.cos(psi)

        vt = (vt <= 0.01) * 0.01 + (vt > 0.01) * vt

        # Control inputs

        T = x[:, 12]
        el = x[:, 13]
        ail = x[:, 14]
        rud = x[:, 15]
        lef = x[:, 16]

        dail = ail / 21.5
        drud = rud / 30.0
        dlef = (1 - lef / 25.0)

        # Atmospheric effects
        # sets dynamic pressure and mach number

        temp = self.atmos(alt, vt)
        mach = temp[0]
        qbar = temp[1] # dynamic pressure
        ps = temp[2]

        # Dynamics
        # Navigation Equations

        U = vt * ca * cb
        V = vt * sb
        W = vt * sa * cb

        xdot[:, 0] = U * (ct * cpsi) + V * (sphi * cpsi * st - cphi * spsi) + W * (cphi * st * cpsi + sphi * spsi)
        xdot[:, 1] = U * (ct * spsi) + V * (sphi * spsi * st + cphi * cpsi) + W * (cphi * st * spsi - sphi * cpsi)
        xdot[:, 2] = U * st - V * (sphi * ct) - W * (cphi * ct)
        xdot[:, 3] = P + tt * (Q * sphi + R * cphi)
        xdot[:, 4] = Q * cphi - R * sphi
        xdot[:, 5] = (Q * sphi + R * cphi) / ct

        temp = self.hifi_F16.hifi_C(alpha, beta, el)
        Cx = temp[0]
        Cz = temp[1]
        Cm = temp[2]
        Cy = temp[3]
        Cn = temp[4]
        Cl = temp[5]

        temp = self.hifi_F16.hifi_damping(alpha)
        Cxq = temp[0]
        Cyr = temp[1]
        Cyp = temp[2]
        Czq = temp[3]
        Clr = temp[4]
        Clp = temp[5]
        Cmq = temp[6]
        Cnr = temp[7]
        Cnp = temp[8]

        temp = self.hifi_F16.hifi_C_lef(alpha, beta)
        delta_Cx_lef = temp[0]
        delta_Cz_lef = temp[1]
        delta_Cm_lef = temp[2]
        delta_Cy_lef = temp[3]
        delta_Cn_lef = temp[4]
        delta_Cl_lef = temp[5]

        temp = self.hifi_F16.hifi_damping_lef(alpha)
        delta_Cxq_lef = temp[0]
        delta_Cyr_lef = temp[1]
        delta_Cyp_lef = temp[2]
        delta_Clr_lef = temp[4]
        delta_Clp_lef = temp[5]
        delta_Cmq_lef = temp[6]
        delta_Cnr_lef = temp[7]
        delta_Cnp_lef = temp[8]

        temp = self.hifi_F16.hifi_rudder(alpha, beta)
        delta_Cy_r30 = temp[0]
        delta_Cn_r30 = temp[1]
        delta_Cl_r30 = temp[2]

        temp = self.hifi_F16.hifi_ailerons(alpha, beta)
        delta_Cy_a20 = temp[0]
        delta_Cy_a20_lef = temp[1]
        delta_Cn_a20 = temp[2]
        delta_Cn_a20_lef = temp[3]
        delta_Cl_a20 = temp[4]
        delta_Cl_a20_lef = temp[5]

        temp = self.hifi_F16.hifi_other_coeffs(alpha, el)
        delta_Cnbeta = temp[0]
        delta_Clbeta = temp[1]
        delta_Cm = temp[2]
        eta_el = temp[3]
        delta_Cm_ds = temp[4]
        
        dXdQ = (cbar / (2 * vt)) * (Cxq + delta_Cxq_lef * dlef)
        Cx_tot = Cx + delta_Cx_lef * dlef + dXdQ * Q
        dZdQ = (cbar / (2 * vt)) * (Czq + delta_Cz_lef * dlef)
        Cz_tot = Cz + delta_Cz_lef * dlef + dZdQ * Q
        dMdQ = (cbar / (2 * vt)) * (Cmq + delta_Cmq_lef * dlef)
        Cm_tot = Cm * eta_el + Cz_tot * (xcgr - xcg) + delta_Cm_lef * dlef + dMdQ * Q + delta_Cm + delta_Cm_ds
        dYdail = delta_Cy_a20 + delta_Cy_a20_lef * dlef
        dYdR = (B / (2 * vt)) * (Cyr + delta_Cyr_lef * dlef)
        dYdP = (B / (2 * vt)) * (Cyp + delta_Cyp_lef * dlef)
        Cy_tot = Cy + delta_Cy_lef * dlef + dYdail * dail + delta_Cy_r30 * drud + dYdR * R + dYdP * P
        dNdail = delta_Cn_a20 + delta_Cn_a20_lef * dlef
        dNdR = (B / (2 * vt)) * (Cnr + delta_Cnr_lef * dlef)
        dNdP = (B / (2 * vt)) * (Cnp + delta_Cnp_lef * dlef)
        Cn_tot = Cn + delta_Cn_lef * dlef - Cy_tot * (xcgr - xcg) * (cbar / B) + dNdail * dail + delta_Cn_r30 * drud + dNdR * R + dNdP * P + delta_Cnbeta * beta
        dLdail = delta_Cl_a20 + delta_Cl_a20_lef * dlef
        dLdR = (B / (2 * vt)) * (Clr + delta_Clr_lef * dlef)
        dLdP = (B / (2 * vt)) * (Clp + delta_Clp_lef * dlef)
        Cl_tot = Cl + delta_Cl_lef * dlef + dLdail * dail + delta_Cl_r30 * drud + dLdR * R + dLdP * P + delta_Clbeta * beta
        Udot = R * V - Q * W - g * st + qbar * S * Cx_tot / m + T / m
        Vdot = P * W - R * U + g * ct * sphi + qbar * S * Cy_tot / m
        Wdot = Q * U - P * V + g * ct * cphi + qbar * S * Cz_tot / m
        xdot[:, 6] = (U * Udot + V * Vdot + W * Wdot) / vt
        xdot[:, 7] = (U * Wdot - W * Udot) / (U * U + W * W)
        xdot[:, 8] = (Vdot * vt - V * xdot[:, 6]) / (vt * vt * cb)
        L_tot = Cl_tot * qbar * S * B
        M_tot = Cm_tot * qbar * S * cbar
        N_tot = Cn_tot * qbar * S * B
        denom = Jx * Jz - Jxz * Jxz
        xdot[:, 9] = (Jz * L_tot + Jxz * N_tot - (Jz * (Jz - Jy) + Jxz * Jxz) * Q * R + Jxz * (Jx - Jy + Jz) * P * Q + Jxz * Q * Heng) / denom
        xdot[:, 10] = (M_tot + (Jz - Jx) * P * R - Jxz * (P * P - R * R) - R * Heng) / Jy
        xdot[:, 11] = (Jx * N_tot + Jxz * L_tot + (Jx * (Jx - Jy) + Jxz * Jxz) * P * Q - Jxz * (Jx - Jy + Jz) * Q * R + Jx * Q * Heng) / denom

        return xdot