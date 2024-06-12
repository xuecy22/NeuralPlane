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

def predict_result():
    alpha = normalize(ALPHA1, ALPHA1_mean, ALPHA1_std)
    beta = normalize(BETA1, BETA1_mean, BETA1_std)
    dele = normalize(DH1, DH1_mean, DH1_std)
    i = np.arange(5)
    input = torch.hstack((alpha[4 * i].reshape(-1, 1), beta[4 * i].reshape(-1, 1)))
    input = torch.hstack((input, dele.reshape(-1, 1)))
    # Cx
    output = unnormalize(Cx_model.forward(input), Cx_mean, Cx_std)
    data = torch.hstack((ALPHA1[4 * i].reshape(-1, 1), BETA1[4 * i].reshape(-1, 1)))
    data = torch.hstack((data, DH1.reshape(-1, 1)))
    data = torch.hstack((data, output.reshape(-1, 1)))
    data = _t2n(data)
    writer = pd.ExcelWriter('result.xlsx')
    data_df = pd.DataFrame(data)
    data_df.to_excel(writer, 'Cx', float_format='%.5f')
    writer.save()
    # Cz
    output = unnormalize(Cz_model.forward(input), Cz_mean, Cz_std)
    data = torch.hstack((ALPHA1[4 * i].reshape(-1, 1), BETA1[4 * i].reshape(-1, 1)))
    data = torch.hstack((data, DH1.reshape(-1, 1)))
    data = torch.hstack((data, output.reshape(-1, 1)))
    data = _t2n(data)
    data_df = pd.DataFrame(data)
    data_df.to_excel(writer, 'Cz', float_format='%.5f')
    writer.save()
    # Cm
    output = unnormalize(Cm_model.forward(input), Cm_mean, Cm_std)
    data = torch.hstack((ALPHA1[4 * i].reshape(-1, 1), BETA1[4 * i].reshape(-1, 1)))
    data = torch.hstack((data, DH1.reshape(-1, 1)))
    data = torch.hstack((data, output.reshape(-1, 1)))
    data = _t2n(data)
    data_df = pd.DataFrame(data)
    data_df.to_excel(writer, 'Cm', float_format='%.5f')
    writer.save()
    # Cy
    input = torch.hstack((alpha[4 * i].reshape(-1, 1), beta[4 * i].reshape(-1, 1)))
    output = unnormalize(Cy_model.forward(input), Cy_mean, Cy_std)
    data = torch.hstack((ALPHA1[4 * i].reshape(-1, 1), BETA1[4 * i].reshape(-1, 1)))
    data = torch.hstack((data, output.reshape(-1, 1)))
    data = _t2n(data)
    data_df = pd.DataFrame(data)
    data_df.to_excel(writer, 'Cy', float_format='%.5f')
    writer.save()
    # Cn
    dele = normalize(DH2, DH2_mean, DH2_std)
    i = np.arange(3)
    input = torch.hstack((alpha[6 * i].reshape(-1, 1), beta[6 * i].reshape(-1, 1)))
    input = torch.hstack((input, dele.reshape(-1, 1)))
    output = unnormalize(Cn_model.forward(input), Cn_mean, Cn_std)
    data = torch.hstack((ALPHA1[6 * i].reshape(-1, 1), BETA1[6 * i].reshape(-1, 1)))
    data = torch.hstack((data, DH2.reshape(-1, 1)))
    data = torch.hstack((data, output.reshape(-1, 1)))
    data = _t2n(data)
    data_df = pd.DataFrame(data)
    data_df.to_excel(writer, 'Cn', float_format='%.5f')
    writer.save()
    # Cl
    output = unnormalize(Cl_model.forward(input), Cl_mean, Cl_std)
    data = torch.hstack((ALPHA1[6 * i].reshape(-1, 1), BETA1[6 * i].reshape(-1, 1)))
    data = torch.hstack((data, DH2.reshape(-1, 1)))
    data = torch.hstack((data, output.reshape(-1, 1)))
    data = _t2n(data)
    data_df = pd.DataFrame(data)
    data_df.to_excel(writer, 'Cl', float_format='%.5f')
    writer.save()
    writer.close()

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

    temp = hifi_F16.hifi_C_lef(alpha, beta)
    delta_Cx_lef_MLP = temp[0]
    delta_Cx_lef_matlab = data_matlab[18, :]
    # delta_Cx_lef_C = data_C[18, :]
    delta_Cz_lef_MLP = temp[1]
    delta_Cz_lef_matlab = data_matlab[19, :]
    # delta_Cz_lef_C = data_C[19, :]
    delta_Cm_lef_MLP = temp[2]
    delta_Cm_lef_matlab = data_matlab[20, :]
    # delta_Cm_lef_C = data_C[20, :]
    delta_Cy_lef_MLP = temp[3]
    delta_Cy_lef_matlab = data_matlab[21, :]
    # delta_Cy_lef_C = data_C[21, :]
    delta_Cn_lef_MLP = temp[4]
    delta_Cn_lef_matlab = data_matlab[22, :]
    # delta_Cn_lef_C = data_C[22, :]
    delta_Cl_lef_MLP = temp[5]
    delta_Cl_lef_matlab = data_matlab[23, :]
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

    temp = hifi_F16.hifi_damping_lef(alpha)
    delta_Cxq_lef_MLP = temp[0]
    delta_Cxq_lef_matlab = data_matlab[24, :]
    # delta_Cxq_lef_C = data_C[24, :]
    delta_Cyr_lef_MLP = temp[1]
    delta_Cyr_lef_matlab = data_matlab[25, :]
    # delta_Cyr_lef_C = data_C[25, :]
    delta_Cyp_lef_MLP = temp[2]
    delta_Cyp_lef_matlab = data_matlab[26, :]
    # delta_Cyp_lef_C = data_C[26, :]
    delta_Czq_lef_MLP = temp[3]
    delta_Czq_lef_matlab = data_matlab[27, :]
    # delta_Czq_lef_C = data_C[27, :]
    delta_Clr_lef_MLP = temp[4]
    delta_Clr_lef_matlab = data_matlab[28, :]
    # delta_Clr_lef_C = data_C[28, :]
    delta_Clp_lef_MLP = temp[5]
    delta_Clp_lef_matlab = data_matlab[29, :]
    # delta_Clp_lef_C = data_C[29, :]
    delta_Cmq_lef_MLP = temp[6]
    delta_Cmq_lef_matlab = data_matlab[30, :]
    # delta_Cmq_lef_C = data_C[30, :]
    delta_Cnr_lef_MLP = temp[7]
    delta_Cnr_lef_matlab = data_matlab[31, :]
    # delta_Cnr_lef_C = data_C[31, :]
    delta_Cnp_lef_MLP = temp[8]
    delta_Cnp_lef_matlab = data_matlab[32, :]
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

    temp = hifi_F16.hifi_ailerons(alpha, beta)
    delta_Cy_a20_MLP = temp[0]
    delta_Cy_a20_matlab = data_matlab[36, :]
    # delta_Cy_a20_C = data_C[36, :]
    delta_Cy_a20_lef_MLP = temp[1]
    delta_Cy_a20_lef_matlab = data_matlab[37, :]
    delta_Cy_a20_lef_C = data_C[37, :]
    delta_Cn_a20_MLP = temp[2]
    delta_Cn_a20_matlab = data_matlab[38, :]
    # delta_Cn_a20_C = data_C[38, :]
    delta_Cn_a20_lef_MLP = temp[3]
    delta_Cn_a20_lef_matlab = data_matlab[39, :]
    # delta_Cn_a20_lef_C = data_C[39, :]
    delta_Cl_a20_MLP = temp[4]
    delta_Cl_a20_matlab = data_matlab[40, :]
    # delta_Cl_a20_C = data_C[40, :]
    delta_Cl_a20_lef_MLP = temp[5]
    delta_Cl_a20_lef_matlab = data_matlab[41, :]
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
    # plot delta_Cy_a20_lef
    t = np.arange(delta_Cy_a20_lef_matlab.shape[0])
    plt.plot(t, delta_Cy_a20_lef_matlab, label='matlab', color='r')
    plt.plot(t, delta_Cy_a20_lef_C, label='C', color='g')
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
ALPHA1 = safe_read_dat(r'ALPHA1.dat')
ALPHA1_mean = torch.mean(ALPHA1)
ALPHA1_std = torch.std(ALPHA1)
ALPHA2 = safe_read_dat(r'ALPHA2.dat')
ALPHA2_mean = torch.mean(ALPHA2)
ALPHA2_std = torch.std(ALPHA2)
BETA1 = safe_read_dat(r'BETA1.dat')
BETA1_mean = torch.mean(BETA1)
BETA1_std = torch.std(BETA1)
DH1 = safe_read_dat(r'DH1.dat')
DH1_mean = torch.mean(DH1)
DH1_std = torch.std(DH1)
DH2 = safe_read_dat(r'DH2.dat')
DH2_mean = torch.mean(DH2)
DH2_std = torch.std(DH2)
Cx = safe_read_dat(r'CX0120_ALPHA1_BETA1_DH1_201.dat')
Cx_mean = torch.mean(Cx)
Cx_std = torch.std(Cx)
Cx_model = MLP(3, 1, [20, 10]).to(device=device)
Cx_model.load_state_dict(torch.load('./Cx.pth'))
Cz = safe_read_dat(r'CZ0120_ALPHA1_BETA1_DH1_301.dat')
Cz_mean = torch.mean(Cz)
Cz_std = torch.std(Cz)
Cz_model = MLP(3, 1, [20, 10]).to(device=device)
Cz_model.load_state_dict(torch.load('./Cz.pth'))
Cm = safe_read_dat(r'CM0120_ALPHA1_BETA1_DH1_101.dat')
Cm_mean = torch.mean(Cm)
Cm_std = torch.std(Cm)
Cm_model = MLP(3, 1, [20, 10]).to(device=device)
Cm_model.load_state_dict(torch.load('./Cm.pth'))
Cy = safe_read_dat(r'CY0320_ALPHA1_BETA1_401.dat')
Cy_mean = torch.mean(Cy)
Cy_std = torch.std(Cy)
Cy_model = MLP(2, 1, [20, 10]).to(device=device)
Cy_model.load_state_dict(torch.load('./Cy.pth'))
Cn = safe_read_dat(r'CN0120_ALPHA1_BETA1_DH2_501.dat')
Cn_mean = torch.mean(Cn)
Cn_std = torch.std(Cn)
Cn_model = MLP(3, 1, [20, 10]).to(device=device)
Cn_model.load_state_dict(torch.load('./Cn.pth'))
Cl = safe_read_dat(r'CL0120_ALPHA1_BETA1_DH2_601.dat')
Cl_mean = torch.mean(Cl)
Cl_std = torch.std(Cl)
Cl_model = MLP(3, 1, [20, 10]).to(device=device)
Cl_model.load_state_dict(torch.load('./Cl.pth'))
Cx_lef = safe_read_dat(r'CX0820_ALPHA2_BETA1_202.dat')
Cx_lef_model = MLP(2, 1, [20, 10]).to(device=device)
Cx_lef_model.load_state_dict(torch.load('./Cx_lef.pth'))
Cz_lef = safe_read_dat(r'CZ0820_ALPHA2_BETA1_302.dat')
Cz_lef_model = MLP(2, 1, [20, 10]).to(device=device)
Cz_lef_model.load_state_dict(torch.load('./Cz_lef.pth'))
Cm_lef = safe_read_dat(r'CM0820_ALPHA2_BETA1_102.dat')
Cm_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
Cm_lef_model.load_state_dict(torch.load('./Cm_lef.pth'))
Cy_lef = safe_read_dat(r'CY0820_ALPHA2_BETA1_402.dat')
Cy_lef_model = MLP(2, 1, [20, 10]).to(device=device)
Cy_lef_model.load_state_dict(torch.load('./Cy_lef.pth'))
Cn_lef = safe_read_dat(r'CN0820_ALPHA2_BETA1_502.dat')
Cn_lef_model = MLP(2, 1, [20, 10]).to(device=device)
Cn_lef_model.load_state_dict(torch.load('./Cn_lef.pth'))
Cl_lef = safe_read_dat(r'CL0820_ALPHA2_BETA1_602.dat')
Cl_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
Cl_lef_model.load_state_dict(torch.load('./Cl_lef.pth'))
CXq = safe_read_dat(r'CX1120_ALPHA1_204.dat')
CXq_model = MLP(1, 1, [20, 10]).to(device=device)
CXq_model.load_state_dict(torch.load('./CXq.pth'))
CZq = safe_read_dat(r'CZ1120_ALPHA1_304.dat')
CZq_model = MLP(1, 1, [20, 10]).to(device=device)
CZq_model.load_state_dict(torch.load('./CZq.pth'))
CMq = safe_read_dat(r'CM1120_ALPHA1_104.dat')
CMq_model = MLP(1, 1, [20, 10]).to(device=device)
CMq_model.load_state_dict(torch.load('./CMq.pth'))
CYp = safe_read_dat(r'CY1220_ALPHA1_408.dat')
CYp_model = MLP(1, 1, [20, 10]).to(device=device)
CYp_model.load_state_dict(torch.load('./CYp.pth'))
CYr = safe_read_dat(r'CY1320_ALPHA1_406.dat')
CYr_model = MLP(1, 1, [20, 10]).to(device=device)
CYr_model.load_state_dict(torch.load('./CYr.pth'))
CNr = safe_read_dat(r'CN1320_ALPHA1_506.dat')
CNr_model = MLP(1, 1, [10, 10]).to(device=device)
CNr_model.load_state_dict(torch.load('./CNr.pth'))
CNp = safe_read_dat(r'CN1220_ALPHA1_508.dat')
CNp_model = MLP(1, 1, [20, 10]).to(device=device)
CNp_model.load_state_dict(torch.load('./CNp.pth'))
CLp = safe_read_dat(r'CL1220_ALPHA1_608.dat')
CLp_model = MLP(1, 1, [20, 10]).to(device=device)
CLp_model.load_state_dict(torch.load('./CLp.pth'))
CLr = safe_read_dat(r'CL1320_ALPHA1_606.dat')
CLr_model = MLP(1, 1, [40, 20]).to(device=device)
CLr_model.load_state_dict(torch.load('./CLr.pth'))
delta_CXq_lef = safe_read_dat(r'CX1420_ALPHA2_205.dat')
delta_CXq_lef_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CXq_lef_model.load_state_dict(torch.load('./delta_CXq_lef.pth'))
delta_CYr_lef = safe_read_dat(r'CY1620_ALPHA2_407.dat')
delta_CYr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CYr_lef_model.load_state_dict(torch.load('./delta_CYr_lef.pth'))
delta_CYp_lef = safe_read_dat(r'CY1520_ALPHA2_409.dat')
delta_CYp_lef_model = MLP(1, 1, [40, 20]).to(device=device)
delta_CYp_lef_model.load_state_dict(torch.load('./delta_CYp_lef.pth'))
delta_CZq_lef = safe_read_dat(r'CZ1420_ALPHA2_305.dat')
delta_CZq_lef_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CZq_lef_model.load_state_dict(torch.load('./delta_CZq_lef.pth'))
delta_CLr_lef = safe_read_dat(r'CL1620_ALPHA2_607.dat')
delta_CLr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CLr_lef_model.load_state_dict(torch.load('./delta_CLr_lef.pth'))
delta_CLp_lef = safe_read_dat(r'CL1520_ALPHA2_609.dat')
delta_CLp_lef_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CLp_lef_model.load_state_dict(torch.load('./delta_CLp_lef.pth'))
delta_CMq_lef = safe_read_dat(r'CM1420_ALPHA2_105.dat')
delta_CMq_lef_model = MLP(1, 1, [40, 20]).to(device=device)
delta_CMq_lef_model.load_state_dict(torch.load('./delta_CMq_lef.pth'))
delta_CNr_lef = safe_read_dat(r'CN1620_ALPHA2_507.dat')
delta_CNr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CNr_lef_model.load_state_dict(torch.load('./delta_CNr_lef.pth'))
delta_CNp_lef = safe_read_dat(r'CN1520_ALPHA2_509.dat')
delta_CNp_lef_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CNp_lef_model.load_state_dict(torch.load('./delta_CNp_lef.pth'))
Cy_r30 = safe_read_dat(r'CY0720_ALPHA1_BETA1_405.dat')
Cy_r30_model = MLP(2, 1, [20, 10]).to(device=device)
Cy_r30_model.load_state_dict(torch.load('./Cy_r30.pth'))
Cn_r30 = safe_read_dat(r'CN0720_ALPHA1_BETA1_503.dat')
Cn_r30_model = MLP(2, 1, [20, 10]).to(device=device)
Cn_r30_model.load_state_dict(torch.load('./Cn_r30.pth'))
Cl_r30 = safe_read_dat(r'CL0720_ALPHA1_BETA1_603.dat')
Cl_r30_model = MLP(2, 1, [20, 10]).to(device=device)
Cl_r30_model.load_state_dict(torch.load('./Cl_r30.pth'))
Cy_a20 = safe_read_dat(r'CY0620_ALPHA1_BETA1_403.dat')
Cy_a20_model = MLP(2, 1, [20, 10]).to(device=device)
Cy_a20_model.load_state_dict(torch.load('./Cy_a20.pth'))
Cy_a20_lef = safe_read_dat(r'CY0920_ALPHA2_BETA1_404.dat')
Cy_a20_lef_model = MLP(2, 1, [20, 10]).to(device=device)
Cy_a20_lef_model.load_state_dict(torch.load('./Cy_a20_lef.pth'))
Cn_a20 = safe_read_dat(r'CN0620_ALPHA1_BETA1_504.dat')
Cn_a20_model = MLP(2, 1, [20, 10]).to(device=device)
Cn_a20_model.load_state_dict(torch.load('./Cn_a20.pth'))
Cn_a20_lef = safe_read_dat(r'CN0920_ALPHA2_BETA1_505.dat')
Cn_a20_lef_model = MLP(2, 1, [20, 10]).to(device=device)
Cn_a20_lef_model.load_state_dict(torch.load('./Cn_a20_lef.pth'))
Cl_a20 = safe_read_dat(r'CL0620_ALPHA1_BETA1_604.dat')
Cl_a20_model = MLP(2, 1, [20, 10]).to(device=device)
Cl_a20_model.load_state_dict(torch.load('./Cl_a20.pth'))
Cl_a20_lef = safe_read_dat(r'CL0920_ALPHA2_BETA1_605.dat')
Cl_a20_lef_model = MLP(2, 1, [20, 10]).to(device=device)
Cl_a20_lef_model.load_state_dict(torch.load('./Cl_a20_lef.pth'))
delta_CNbeta = safe_read_dat(r'CN9999_ALPHA1_brett.dat')
delta_CNbeta_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CNbeta_model.load_state_dict(torch.load('./delta_CNbeta.pth'))
delta_CLbeta = safe_read_dat(r'CL9999_ALPHA1_brett.dat')
delta_CLbeta_model = MLP(1, 1, [20, 10]).to(device=device)
delta_CLbeta_model.load_state_dict(torch.load('./delta_CLbeta.pth'))
delta_Cm = safe_read_dat(r'CM9999_ALPHA1_brett.dat')
delta_Cm_model = MLP(1, 1, [20, 10]).to(device=device)
delta_Cm_model.load_state_dict(torch.load('./delta_Cm.pth'))
eta_el = safe_read_dat(r'ETA_DH1_brett.dat')
eta_el_model = MLP(1, 1, [20, 10]).to(device=device)
eta_el_model.load_state_dict(torch.load('./eta_el.pth'))
compare_result()
