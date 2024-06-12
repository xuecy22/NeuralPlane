import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import envs.models.F16.hifi_F16_AeroData as hifi

device = "cuda:0"

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        out_dim = out_dim

    def forward(self, x):
        x = x.to(torch.float32)
        ret = self.layers(x)
        ret = ret.reshape(-1)
        return ret


def safe_read_dat(dat_name):
    try:
        path = r'../data/' + dat_name
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            content = content.strip()
            data_str = [value for value in content.split(' ') if value]
            data = list(map(float, data_str))
            data = torch.tensor(data, device=torch.device(device))
            return data
    except OSError:
        print("Cannot find file {} in current directory".format(path))
        return []


def normalize(X, mean, std):
    return (X - mean) / std


def unnormalize(X, mean, std):
    return X * std + mean


def _t2n(x):
    return x.detach().cpu().numpy()

def compare_result():
    data_matlab = np.array(pd.read_csv('coefs.csv', header=None))
    data_C = np.array(pd.read_csv('coefs_C.csv', header=None))
    alpha = torch.tensor(data_matlab[0, :], device=torch.device(device))
    beta = torch.tensor(data_matlab[1, :], device=torch.device(device))
    dele = torch.tensor(data_matlab[2, :], device=torch.device(device))
    r2_list = []
    error_list = []

    temp = hifi_F16.hifi_C(alpha, beta, dele)
    Cx_MLP = temp[0]
    Cx_matlab = data_matlab[3, :]
    Cx_C = data_C[3, :]
    Cz_MLP = temp[1]
    Cz_matlab = data_matlab[4, :]
    Cz_C = data_C[4, :]
    Cm_MLP = temp[2]
    Cm_matlab = data_matlab[5, :]
    # Cm_C = data_C[5, :]
    Cy_MLP = temp[3]
    Cy_matlab = data_matlab[6, :]
    # Cy_C = data_C[6, :]
    Cn_MLP = temp[4]
    Cn_matlab = data_matlab[7, :]
    # Cn_C = data_C[7, :]
    Cl_MLP = temp[5]
    Cl_matlab = data_matlab[8, :]
    # Cl_C = data_C[8, :]
    r2_list.append(r2_score(Cx_matlab, _t2n(Cx_MLP)))
    r2_list.append(r2_score(Cz_matlab, _t2n(Cz_MLP)))
    r2_list.append(r2_score(Cm_matlab, _t2n(Cm_MLP)))
    r2_list.append(r2_score(Cy_matlab, _t2n(Cy_MLP)))
    r2_list.append(r2_score(Cn_matlab, _t2n(Cn_MLP)))
    r2_list.append(r2_score(Cl_matlab, _t2n(Cl_MLP)))
    error_list.append(np.mean(np.abs(_t2n(Cx_MLP) - Cx_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cz_MLP) - Cz_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cm_MLP) - Cm_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cy_MLP) - Cy_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cn_MLP) - Cn_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cl_MLP) - Cl_matlab)))
    print('Cx:', r2_score(Cx_matlab, _t2n(Cx_MLP)))
    print('Cz:', r2_score(Cz_matlab, _t2n(Cz_MLP)))
    print('Cm:', r2_score(Cm_matlab, _t2n(Cm_MLP)))
    print('Cy:', r2_score(Cy_matlab, _t2n(Cy_MLP)))
    print('Cn:', r2_score(Cn_matlab, _t2n(Cn_MLP)))
    print('Cl:', r2_score(Cl_matlab, _t2n(Cl_MLP)))

    temp = hifi_F16.hifi_damping(alpha)
    Cxq_MLP = temp[0]
    Cxq_matlab = data_matlab[9, :]
    # Cxq_C = data_C[9, :]
    Cyr_MLP = temp[1]
    Cyr_matlab = data_matlab[10, :]
    # Cyr_C = data_C[10, :]
    Cyp_MLP = temp[2]
    Cyp_matlab = data_matlab[11, :]
    # Cyp_C = data_C[11, :]
    Czq_MLP = temp[3]
    Czq_matlab = data_matlab[12, :]
    # Czq_C = data_C[12, :]
    Clr_MLP = temp[4]
    Clr_matlab = data_matlab[13, :]
    # Clr_C = data_C[13, :]
    Clp_MLP = temp[5]
    Clp_matlab = data_matlab[14, :]
    # Clp_C = data_C[14, :]
    Cmq_MLP = temp[6]
    Cmq_matlab = data_matlab[15, :]
    # Cmq_C = data_C[15, :]
    Cnr_MLP = temp[7]
    Cnr_matlab = data_matlab[16, :]
    # Cnr_C = data_C[16, :]
    Cnp_MLP = temp[8]
    Cnp_matlab = data_matlab[17, :]
    # Cnp_C = data_C[17, :]
    r2_list.append(r2_score(Cxq_matlab, _t2n(Cxq_MLP)))
    r2_list.append(r2_score(Cyr_matlab, _t2n(Cyr_MLP)))
    r2_list.append(r2_score(Cyp_matlab, _t2n(Cyp_MLP)))
    r2_list.append(r2_score(Czq_matlab, _t2n(Czq_MLP)))
    r2_list.append(r2_score(Clr_matlab, _t2n(Clr_MLP)))
    r2_list.append(r2_score(Clp_matlab, _t2n(Clp_MLP)))
    r2_list.append(r2_score(Cmq_matlab, _t2n(Cmq_MLP)))
    r2_list.append(r2_score(Cnr_matlab, _t2n(Cnr_MLP)))
    r2_list.append(r2_score(Cnp_matlab, _t2n(Cnp_MLP)))
    error_list.append(np.mean(np.abs(_t2n(Cxq_MLP) - Cxq_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cyr_MLP) - Cyr_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cyp_MLP) - Cyp_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Czq_MLP) - Czq_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Clr_MLP) - Clr_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Clp_MLP) - Clp_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cmq_MLP) - Cmq_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cnr_MLP) - Cnr_matlab)))
    error_list.append(np.mean(np.abs(_t2n(Cnp_MLP) - Cnp_matlab)))
    print('Cxq:', r2_score(Cxq_matlab, _t2n(Cxq_MLP)))
    print('Cyr:', r2_score(Cyr_matlab, _t2n(Cyr_MLP)))
    print('Cyp:', r2_score(Cyp_matlab, _t2n(Cyp_MLP)))
    print('Czq:', r2_score(Czq_matlab, _t2n(Czq_MLP)))
    print('Clr:', r2_score(Clr_matlab, _t2n(Clr_MLP)))
    print('Clp:', r2_score(Clp_matlab, _t2n(Clp_MLP)))
    print('Cmq:', r2_score(Cmq_matlab, _t2n(Cmq_MLP)))
    print('Cnr:', r2_score(Cnr_matlab, _t2n(Cnr_MLP)))
    print('Cnp:', r2_score(Cnp_matlab, _t2n(Cnp_MLP)))

    alpha = torch.tensor(data_matlab[0, :400], device=torch.device(device))
    beta = torch.tensor(data_matlab[1, :400], device=torch.device(device))
    temp = hifi_F16.hifi_C_lef(alpha, beta)
    delta_Cx_lef_MLP = temp[0]
    delta_Cx_lef_matlab = data_matlab[18, :400]
    # delta_Cx_lef_C = data_C[18, :]
    delta_Cz_lef_MLP = temp[1]
    delta_Cz_lef_matlab = data_matlab[19, :400]
    # delta_Cz_lef_C = data_C[19, :]
    delta_Cm_lef_MLP = temp[2]
    delta_Cm_lef_matlab = data_matlab[20, :400]
    # delta_Cm_lef_C = data_C[20, :]
    delta_Cy_lef_MLP = temp[3]
    delta_Cy_lef_matlab = data_matlab[21, :400]
    # delta_Cy_lef_C = data_C[21, :]
    delta_Cn_lef_MLP = temp[4]
    delta_Cn_lef_matlab = data_matlab[22, :400]
    # delta_Cn_lef_C = data_C[22, :]
    delta_Cl_lef_MLP = temp[5]
    delta_Cl_lef_matlab = data_matlab[23, :400]
    # delta_Cl_lef_C = data_C[23, :]
    r2_list.append(r2_score(delta_Cx_lef_matlab, _t2n(delta_Cx_lef_MLP)))
    r2_list.append(r2_score(delta_Cz_lef_matlab, _t2n(delta_Cz_lef_MLP)))
    r2_list.append(r2_score(delta_Cm_lef_matlab, _t2n(delta_Cm_lef_MLP)))
    r2_list.append(r2_score(delta_Cy_lef_matlab, _t2n(delta_Cy_lef_MLP)))
    r2_list.append(r2_score(delta_Cn_lef_matlab, _t2n(delta_Cn_lef_MLP)))
    r2_list.append(r2_score(delta_Cl_lef_matlab, _t2n(delta_Cl_lef_MLP)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cx_lef_MLP) - delta_Cx_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cz_lef_MLP) - delta_Cz_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cm_lef_MLP) - delta_Cm_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cy_lef_MLP) - delta_Cy_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cn_lef_MLP) - delta_Cn_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cl_lef_MLP) - delta_Cl_lef_matlab)))
    print('delta_Cx_lef:', r2_score(delta_Cx_lef_matlab, _t2n(delta_Cx_lef_MLP)))
    print('delta_Cz_lef:', r2_score(delta_Cz_lef_matlab, _t2n(delta_Cz_lef_MLP)))
    print('delta_Cm_lef:', r2_score(delta_Cm_lef_matlab, _t2n(delta_Cm_lef_MLP)))
    print('delta_Cy_lef:', r2_score(delta_Cy_lef_matlab, _t2n(delta_Cy_lef_MLP)))
    print('delta_Cn_lef:', r2_score(delta_Cn_lef_matlab, _t2n(delta_Cn_lef_MLP)))
    print('delta_Cl_lef:', r2_score(delta_Cl_lef_matlab, _t2n(delta_Cl_lef_MLP)))

    alpha = torch.tensor(data_matlab[0, :400], device=torch.device(device))
    temp = hifi_F16.hifi_damping_lef(alpha)
    delta_Cxq_lef_MLP = temp[0]
    delta_Cxq_lef_matlab = data_matlab[24, :400]
    # delta_Cxq_lef_C = data_C[24, :]
    delta_Cyr_lef_MLP = temp[1]
    delta_Cyr_lef_matlab = data_matlab[25, :400]
    # delta_Cyr_lef_C = data_C[25, :]
    delta_Cyp_lef_MLP = temp[2]
    delta_Cyp_lef_matlab = data_matlab[26, :400]
    # delta_Cyp_lef_C = data_C[26, :]
    delta_Czq_lef_MLP = temp[3]
    delta_Czq_lef_matlab = data_matlab[27, :400]
    # delta_Czq_lef_C = data_C[27, :]
    delta_Clr_lef_MLP = temp[4]
    delta_Clr_lef_matlab = data_matlab[28, :400]
    # delta_Clr_lef_C = data_C[28, :]
    delta_Clp_lef_MLP = temp[5]
    delta_Clp_lef_matlab = data_matlab[29, :400]
    # delta_Clp_lef_C = data_C[29, :]
    delta_Cmq_lef_MLP = temp[6]
    delta_Cmq_lef_matlab = data_matlab[30, :400]
    # delta_Cmq_lef_C = data_C[30, :]
    delta_Cnr_lef_MLP = temp[7]
    delta_Cnr_lef_matlab = data_matlab[31, :400]
    # delta_Cnr_lef_C = data_C[31, :]
    delta_Cnp_lef_MLP = temp[8]
    delta_Cnp_lef_matlab = data_matlab[32, :400]
    # delta_Cnp_lef_C = data_C[32, :]
    r2_list.append(r2_score(delta_Cxq_lef_matlab, _t2n(delta_Cxq_lef_MLP)))
    r2_list.append(r2_score(delta_Cyr_lef_matlab, _t2n(delta_Cyr_lef_MLP)))
    r2_list.append(r2_score(delta_Cyp_lef_matlab, _t2n(delta_Cyp_lef_MLP)))
    r2_list.append(r2_score(delta_Czq_lef_matlab, _t2n(delta_Czq_lef_MLP)))
    r2_list.append(r2_score(delta_Clr_lef_matlab, _t2n(delta_Clr_lef_MLP)))
    r2_list.append(r2_score(delta_Clp_lef_matlab, _t2n(delta_Clp_lef_MLP)))
    r2_list.append(r2_score(delta_Cmq_lef_matlab, _t2n(delta_Cmq_lef_MLP)))
    r2_list.append(r2_score(delta_Cnr_lef_matlab, _t2n(delta_Cnr_lef_MLP)))
    r2_list.append(r2_score(delta_Cnp_lef_matlab, _t2n(delta_Cnp_lef_MLP)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cxq_lef_MLP) - delta_Cxq_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cyr_lef_MLP) - delta_Cyr_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cyp_lef_MLP) - delta_Cyp_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Czq_lef_MLP) - delta_Czq_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Clr_lef_MLP) - delta_Clr_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Clp_lef_MLP) - delta_Clp_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cmq_lef_MLP) - delta_Cmq_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cnr_lef_MLP) - delta_Cnr_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cnp_lef_MLP) - delta_Cnp_lef_matlab)))
    print('delta_Cxq_lef:', r2_score(delta_Cxq_lef_matlab, _t2n(delta_Cxq_lef_MLP)))
    print('delta_Cyr_lef:', r2_score(delta_Cyr_lef_matlab, _t2n(delta_Cyr_lef_MLP)))
    print('delta_Cyp_lef:', r2_score(delta_Cyp_lef_matlab, _t2n(delta_Cyp_lef_MLP)))
    print('delta_Czq_lef:', r2_score(delta_Czq_lef_matlab, _t2n(delta_Czq_lef_MLP)))
    print('delta_Clr_lef:', r2_score(delta_Clr_lef_matlab, _t2n(delta_Clr_lef_MLP)))
    print('delta_Clp_lef:', r2_score(delta_Clp_lef_matlab, _t2n(delta_Clp_lef_MLP)))
    print('delta_Cmq_lef:', r2_score(delta_Cmq_lef_matlab, _t2n(delta_Cmq_lef_MLP)))
    print('delta_Cnr_lef:', r2_score(delta_Cnr_lef_matlab, _t2n(delta_Cnr_lef_MLP)))
    print('delta_Cnp_lef:', r2_score(delta_Cnp_lef_matlab, _t2n(delta_Cnp_lef_MLP)))

    alpha = torch.tensor(data_matlab[0, :], device=torch.device(device))
    beta = torch.tensor(data_matlab[1, :], device=torch.device(device))
    temp = hifi_F16.hifi_rudder(alpha, beta)
    delta_Cy_r30_MLP = temp[0]
    delta_Cy_r30_matlab = data_matlab[33, :]
    # delta_Cy_r30_C = data_C[33, :]
    delta_Cn_r30_MLP = temp[1]
    delta_Cn_r30_matlab = data_matlab[34, :]
    # delta_Cn_r30_C = data_C[34, :]
    delta_Cl_r30_MLP = temp[2]
    delta_Cl_r30_matlab = data_matlab[35, :]
    # delta_Cl_r30_C = data_C[35, :]
    r2_list.append(r2_score(delta_Cy_r30_matlab, _t2n(delta_Cy_r30_MLP)))
    r2_list.append(r2_score(delta_Cn_r30_matlab, _t2n(delta_Cn_r30_MLP)))
    r2_list.append(r2_score(delta_Cl_r30_matlab, _t2n(delta_Cl_r30_MLP)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cy_r30_MLP) - delta_Cy_r30_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cn_r30_MLP) - delta_Cn_r30_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cl_r30_MLP) - delta_Cl_r30_matlab)))
    print('delta_Cy_r30:', r2_score(delta_Cy_r30_matlab, _t2n(delta_Cy_r30_MLP)))
    print('delta_Cn_r30:', r2_score(delta_Cn_r30_matlab, _t2n(delta_Cn_r30_MLP)))
    print('delta_Cl_r30:', r2_score(delta_Cl_r30_matlab, _t2n(delta_Cl_r30_MLP)))

    alpha = torch.tensor(data_matlab[0, :400], device=torch.device(device))
    beta = torch.tensor(data_matlab[1, :400], device=torch.device(device))
    temp = hifi_F16.hifi_ailerons(alpha, beta)
    delta_Cy_a20_MLP = temp[0]
    delta_Cy_a20_matlab = data_matlab[36, :400]
    # delta_Cy_a20_C = data_C[36, :]
    delta_Cy_a20_lef_MLP = temp[1]
    delta_Cy_a20_lef_matlab = data_matlab[39, :400]
    delta_Cy_a20_lef_C = data_C[39, :]
    delta_Cn_a20_MLP = temp[2]
    delta_Cn_a20_matlab = data_matlab[37, :400]
    # delta_Cn_a20_C = data_C[37, :]
    delta_Cn_a20_lef_MLP = temp[3]
    delta_Cn_a20_lef_matlab = data_matlab[40, :400]
    # delta_Cn_a20_lef_C = data_C[40, :]
    delta_Cl_a20_MLP = temp[4]
    delta_Cl_a20_matlab = data_matlab[38, :400]
    # delta_Cl_a20_C = data_C[38, :]
    delta_Cl_a20_lef_MLP = temp[5]
    delta_Cl_a20_lef_matlab = data_matlab[41, :400]
    # delta_Cl_a20_lef_C = data_C[41, :]
    r2_list.append(r2_score(delta_Cy_a20_matlab, _t2n(delta_Cy_a20_MLP)))
    r2_list.append(r2_score(delta_Cy_a20_lef_matlab, _t2n(delta_Cy_a20_lef_MLP)))
    r2_list.append(r2_score(delta_Cn_a20_matlab, _t2n(delta_Cn_a20_MLP)))
    r2_list.append(r2_score(delta_Cn_a20_lef_matlab, _t2n(delta_Cn_a20_lef_MLP)))
    r2_list.append(r2_score(delta_Cl_a20_matlab, _t2n(delta_Cl_a20_MLP)))
    r2_list.append(r2_score(delta_Cl_a20_lef_matlab, _t2n(delta_Cl_a20_lef_MLP)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cy_a20_MLP) - delta_Cy_a20_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cy_a20_lef_MLP) - delta_Cy_a20_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cn_a20_MLP) - delta_Cn_a20_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cn_a20_lef_MLP) - delta_Cn_a20_lef_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cl_a20_MLP) - delta_Cl_a20_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cl_a20_lef_MLP) - delta_Cl_a20_lef_matlab)))
    print('delta_Cy_a20:', r2_score(delta_Cy_a20_matlab, _t2n(delta_Cy_a20_MLP)))
    print('delta_Cy_a20_lef:', r2_score(delta_Cy_a20_lef_matlab, _t2n(delta_Cy_a20_lef_MLP)))
    print('delta_Cn_a20:', r2_score(delta_Cn_a20_matlab, _t2n(delta_Cn_a20_MLP)))
    print('delta_Cn_a20_lef:', r2_score(delta_Cn_a20_lef_matlab, _t2n(delta_Cn_a20_lef_MLP)))
    print('delta_Cl_a20:', r2_score(delta_Cl_a20_matlab, _t2n(delta_Cl_a20_MLP)))
    print('delta_Cl_a20_lef:', r2_score(delta_Cl_a20_lef_matlab, _t2n(delta_Cl_a20_lef_MLP)))

    alpha = torch.tensor(data_matlab[0, :], device=torch.device(device))
    beta = torch.tensor(data_matlab[1, :], device=torch.device(device))
    temp = hifi_F16.hifi_other_coeffs(alpha, dele)
    delta_Cnbeta_MLP = temp[0]
    delta_Cnbeta_matlab = data_matlab[42, :]
    # delta_Cnbeta_C = data_C[42, :]
    delta_Clbeta_MLP = temp[1]
    delta_Clbeta_matlab = data_matlab[43, :]
    # delta_Clbeta_C = data_C[43, :]
    delta_Cm_MLP = temp[2]
    delta_Cm_matlab = data_matlab[44, :]
    # delta_Cm_C = data_C[44, :]
    eta_el_MLP = temp[3]
    eta_el_matlab = data_matlab[45, :]
    # eta_el_C = data_C[45, :]
    delta_Cm_ds_MLP = temp[4]
    delta_Cm_ds_matlab = data_matlab[46, :]
    # delta_Cm_ds_C = data_C[46, :]
    r2_list.append(r2_score(delta_Cnbeta_matlab, _t2n(delta_Cnbeta_MLP)))
    r2_list.append(r2_score(delta_Clbeta_matlab, _t2n(delta_Clbeta_MLP)))
    r2_list.append(r2_score(delta_Cm_matlab, _t2n(delta_Cm_MLP)))
    r2_list.append(r2_score(eta_el_matlab, _t2n(eta_el_MLP)))
    r2_list.append(r2_score(delta_Cm_ds_matlab, _t2n(delta_Cm_ds_MLP)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cnbeta_MLP) - delta_Cnbeta_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Clbeta_MLP) - delta_Clbeta_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cm_MLP) - delta_Cm_matlab)))
    error_list.append(np.mean(np.abs(_t2n(eta_el_MLP) - eta_el_matlab)))
    error_list.append(np.mean(np.abs(_t2n(delta_Cm_ds_MLP) - delta_Cm_ds_matlab)))
    print('delta_Cnbeta:', r2_score(delta_Cnbeta_matlab, _t2n(delta_Cnbeta_MLP)))
    print('delta_Clbeta:', r2_score(delta_Clbeta_matlab, _t2n(delta_Clbeta_MLP)))
    print('delta_Cm:', r2_score(delta_Cm_matlab, _t2n(delta_Cm_MLP)))
    print('eta_el:', r2_score(eta_el_matlab, _t2n(eta_el_MLP)))
    print('delta_Cm_ds:', r2_score(delta_Cm_ds_matlab, _t2n(delta_Cm_ds_MLP)))
    
    # plot Cx
    t = np.arange(Cx_matlab.shape[0])
    plt.plot(t, Cx_matlab, label='matlab', color='r')
    plt.plot(t, Cx_C, label='C', color='g')
    plt.plot(t, _t2n(Cx_MLP), label='MLP', color='b')
    plt.legend()
    plt.show()
    # plot Cz
    t = np.arange(Cz_matlab.shape[0])
    plt.plot(t, Cz_matlab, label='matlab', color='r')
    plt.plot(t, Cz_C, label='C', color='g')
    plt.plot(t, _t2n(Cz_MLP), label='MLP', color='b')
    plt.legend()
    plt.show()
    # plot delta_Cy_lef
    t = np.arange(delta_Cy_lef_matlab.shape[0])
    plt.plot(t, delta_Cy_lef_matlab, label='matlab', color='r')
    # plt.plot(t, delta_Cy_lef_C, label='C', color='g')
    plt.plot(t, _t2n(delta_Cy_lef_MLP), label='MLP', color='b')
    plt.legend()
    plt.show()
    # plot delta_Cy_a20_lef
    t = np.arange(delta_Cy_a20_lef_matlab.shape[0])
    plt.plot(t, delta_Cy_a20_lef_matlab, label='matlab', color='r')
    # plt.plot(t, delta_Cy_a20_lef_C, label='C', color='g')
    plt.plot(t, _t2n(delta_Cy_a20_lef_MLP), label='MLP', color='b')
    plt.legend()
    plt.show()
    # plot eta_el
    t = np.arange(eta_el_matlab.shape[0])
    plt.plot(t, eta_el_matlab, label='matlab', color='r')
    # plt.plot(t, eta_el_C, label='C', color='g')
    plt.plot(t, _t2n(eta_el_MLP), label='MLP', color='b')
    plt.legend()
    plt.show()


hifi_F16 = hifi.hifi_F16()
compare_result()
