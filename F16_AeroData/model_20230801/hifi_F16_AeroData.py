import torch
import torch.nn as nn
import os


HIFI_GLOBAL_TXT_CONTENT = {}
device = "cuda:0"
path = os.getcwd()


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
        self.out_dim = out_dim

    def forward(self, x):
        x = x.to(torch.float32)
        ret = self.layers(x)
        ret = ret.reshape(-1)
        return ret
    

def normalize(X, mean, std):
    return (X - mean) / std


def unnormalize(X, mean, std):
    return X * std + mean


class hifi_F16():
    def __init__(self) -> None:
        self.ALPHA1_mean = torch.mean(self.safe_read_dat(r'ALPHA1.dat'))
        self.ALPHA1_std = torch.std(self.safe_read_dat(r'ALPHA1.dat'))
        self.ALPHA2_mean = torch.mean(self.safe_read_dat(r'ALPHA2.dat'))
        self.ALPHA2_std = torch.std(self.safe_read_dat(r'ALPHA2.dat'))
        self.BETA1_mean = torch.mean(self.safe_read_dat(r'BETA1.dat'))
        self.BETA1_std = torch.std(self.safe_read_dat(r'BETA1.dat'))
        self.DH1_mean = torch.mean(self.safe_read_dat(r'DH1.dat'))
        self.DH1_std = torch.std(self.safe_read_dat(r'DH1.dat'))
        self.DH2_mean = torch.mean(self.safe_read_dat(r'DH2.dat'))
        self.DH2_std = torch.std(self.safe_read_dat(r'DH2.dat'))
        self.Cx_mean = torch.mean(self.safe_read_dat(r'CX0120_ALPHA1_BETA1_DH1_201.dat'))
        self.Cx_std = torch.std(self.safe_read_dat(r'CX0120_ALPHA1_BETA1_DH1_201.dat'))
        self.Cx_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cx_model.load_state_dict(torch.load(path + '/model/Cx.pth'))
        self.Cz_mean = torch.mean(self.safe_read_dat(r'CZ0120_ALPHA1_BETA1_DH1_301.dat'))
        self.Cz_std = torch.std(self.safe_read_dat(r'CZ0120_ALPHA1_BETA1_DH1_301.dat'))
        self.Cz_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cz_model.load_state_dict(torch.load(path + '/model/Cz.pth'))
        self.Cm_mean = torch.mean(self.safe_read_dat(r'CM0120_ALPHA1_BETA1_DH1_101.dat'))
        self.Cm_std = torch.std(self.safe_read_dat(r'CM0120_ALPHA1_BETA1_DH1_101.dat'))
        self.Cm_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cm_model.load_state_dict(torch.load(path + '/model/Cm.pth'))
        self.Cy_mean = torch.mean(self.safe_read_dat(r'CY0320_ALPHA1_BETA1_401.dat'))
        self.Cy_std = torch.std(self.safe_read_dat(r'CY0320_ALPHA1_BETA1_401.dat'))
        self.Cy_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cy_model.load_state_dict(torch.load(path + '/model/Cy.pth'))
        self.Cn_mean = torch.mean(self.safe_read_dat(r'CN0120_ALPHA1_BETA1_DH2_501.dat'))
        self.Cn_std = torch.std(self.safe_read_dat(r'CN0120_ALPHA1_BETA1_DH2_501.dat'))
        self.Cn_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cn_model.load_state_dict(torch.load(path + '/model/Cn.pth'))
        self.Cl_mean = torch.mean(self.safe_read_dat(r'CL0120_ALPHA1_BETA1_DH2_601.dat'))
        self.Cl_std = torch.std(self.safe_read_dat(r'CL0120_ALPHA1_BETA1_DH2_601.dat'))
        self.Cl_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cl_model.load_state_dict(torch.load(path + '/model/Cl.pth'))
        self.Cx_lef_mean = torch.mean(self.safe_read_dat(r'CX0820_ALPHA2_BETA1_202.dat'))
        self.Cx_lef_std = torch.std(self.safe_read_dat(r'CX0820_ALPHA2_BETA1_202.dat'))
        self.Cx_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cx_lef_model.load_state_dict(torch.load(path + '/model/Cx_lef.pth'))
        self.Cz_lef_mean = torch.mean(self.safe_read_dat(r'CZ0820_ALPHA2_BETA1_302.dat'))
        self.Cz_lef_std = torch.std(self.safe_read_dat(r'CZ0820_ALPHA2_BETA1_302.dat'))
        self.Cz_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cz_lef_model.load_state_dict(torch.load(path + '/model/Cz_lef.pth'))
        self.Cm_lef_mean = torch.mean(self.safe_read_dat(r'CM0820_ALPHA2_BETA1_102.dat'))
        self.Cm_lef_std = torch.std(self.safe_read_dat(r'CM0820_ALPHA2_BETA1_102.dat'))
        self.Cm_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.Cm_lef_model.load_state_dict(torch.load(path + '/model/Cm_lef.pth'))
        self.Cy_lef_mean = torch.mean(self.safe_read_dat(r'CY0820_ALPHA2_BETA1_402.dat'))
        self.Cy_lef_std = torch.std(self.safe_read_dat(r'CY0820_ALPHA2_BETA1_402.dat'))
        self.Cy_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cy_lef_model.load_state_dict(torch.load(path + '/model/Cy_lef.pth'))
        self.Cn_lef_mean = torch.mean(self.safe_read_dat(r'CN0820_ALPHA2_BETA1_502.dat'))
        self.Cn_lef_std = torch.std(self.safe_read_dat(r'CN0820_ALPHA2_BETA1_502.dat'))
        self.Cn_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cn_lef_model.load_state_dict(torch.load(path + '/model/Cn_lef.pth'))
        self.Cl_lef_mean = torch.mean(self.safe_read_dat(r'CL0820_ALPHA2_BETA1_602.dat'))
        self.Cl_lef_std = torch.std(self.safe_read_dat(r'CL0820_ALPHA2_BETA1_602.dat'))
        self.Cl_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.Cl_lef_model.load_state_dict(torch.load(path + '/model/Cl_lef.pth'))
        self.CXq_mean = torch.mean(self.safe_read_dat(r'CX1120_ALPHA1_204.dat'))
        self.CXq_std = torch.std(self.safe_read_dat(r'CX1120_ALPHA1_204.dat'))
        self.CXq_model = MLP(1, 1, [20, 10]).to(device=device)
        self.CXq_model.load_state_dict(torch.load(path + '/model/CXq.pth'))
        self.CZq_mean = torch.mean(self.safe_read_dat(r'CZ1120_ALPHA1_304.dat'))
        self.CZq_std = torch.std(self.safe_read_dat(r'CZ1120_ALPHA1_304.dat'))
        self.CZq_model = MLP(1, 1, [20, 10]).to(device=device)
        self.CZq_model.load_state_dict(torch.load(path + '/model/CZq.pth'))
        self.CMq_mean = torch.mean(self.safe_read_dat(r'CM1120_ALPHA1_104.dat'))
        self.CMq_std = torch.std(self.safe_read_dat(r'CM1120_ALPHA1_104.dat'))
        self.CMq_model = MLP(1, 1, [20, 10]).to(device=device)
        self.CMq_model.load_state_dict(torch.load(path + '/model/CMq.pth'))
        self.CYp_mean = torch.mean(self.safe_read_dat(r'CY1220_ALPHA1_408.dat'))
        self.CYp_std = torch.std(self.safe_read_dat(r'CY1220_ALPHA1_408.dat'))
        self.CYp_model = MLP(1, 1, [20, 10]).to(device=device)
        self.CYp_model.load_state_dict(torch.load(path + '/model/CYp.pth'))
        self.CYr_mean = torch.mean(self.safe_read_dat(r'CY1320_ALPHA1_406.dat'))
        self.CYr_std = torch.std(self.safe_read_dat(r'CY1320_ALPHA1_406.dat'))
        self.CYr_model = MLP(1, 1, [20, 10]).to(device=device)
        self.CYr_model.load_state_dict(torch.load(path + '/model/CYr.pth'))
        self.CNr_mean = torch.mean(self.safe_read_dat(r'CN1320_ALPHA1_506.dat'))
        self.CNr_std = torch.std(self.safe_read_dat(r'CN1320_ALPHA1_506.dat'))
        self.CNr_model = MLP(1, 1, [10, 10]).to(device=device)
        self.CNr_model.load_state_dict(torch.load(path + '/model/CNr.pth'))
        self.CNp_mean = torch.mean(self.safe_read_dat(r'CN1220_ALPHA1_508.dat'))
        self.CNp_std = torch.std(self.safe_read_dat(r'CN1220_ALPHA1_508.dat'))
        self.CNp_model = MLP(1, 1, [20, 10]).to(device=device)
        self.CNp_model.load_state_dict(torch.load(path + '/model/CNp.pth'))
        self.CLp_mean = torch.mean(self.safe_read_dat(r'CL1220_ALPHA1_608.dat'))
        self.CLp_std = torch.std(self.safe_read_dat(r'CL1220_ALPHA1_608.dat'))
        self.CLp_model = MLP(1, 1, [20, 10]).to(device=device)
        self.CLp_model.load_state_dict(torch.load(path + '/model/CLp.pth'))
        self.CLr_mean = torch.mean(self.safe_read_dat(r'CL1320_ALPHA1_606.dat'))
        self.CLr_std = torch.std(self.safe_read_dat(r'CL1320_ALPHA1_606.dat'))
        self.CLr_model = MLP(1, 1, [40, 20]).to(device=device)
        self.CLr_model.load_state_dict(torch.load(path + '/model/CLr.pth'))
        self.delta_CXq_lef_mean = torch.mean(self.safe_read_dat(r'CX1420_ALPHA2_205.dat'))
        self.delta_CXq_lef_std = torch.std(self.safe_read_dat(r'CX1420_ALPHA2_205.dat'))
        self.delta_CXq_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CXq_lef_model.load_state_dict(torch.load(path + '/model/delta_CXq_lef.pth'))
        self.delta_CYr_lef_mean = torch.mean(self.safe_read_dat(r'CY1620_ALPHA2_407.dat'))
        self.delta_CYr_lef_std = torch.std(self.safe_read_dat(r'CY1620_ALPHA2_407.dat'))
        self.delta_CYr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CYr_lef_model.load_state_dict(torch.load(path + '/model/delta_CYr_lef.pth'))
        self.delta_CYp_lef_mean = torch.mean(self.safe_read_dat(r'CY1520_ALPHA2_409.dat'))
        self.delta_CYp_lef_std = torch.std(self.safe_read_dat(r'CY1520_ALPHA2_409.dat'))
        self.delta_CYp_lef_model = MLP(1, 1, [40, 20]).to(device=device)
        self.delta_CYp_lef_model.load_state_dict(torch.load(path + '/model/delta_CYp_lef.pth'))
        self.delta_CZq_lef_mean = torch.mean(self.safe_read_dat(r'CZ1420_ALPHA2_305.dat'))
        self.delta_CZq_lef_std = torch.std(self.safe_read_dat(r'CZ1420_ALPHA2_305.dat'))
        self.delta_CZq_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CZq_lef_model.load_state_dict(torch.load(path + '/model/delta_CZq_lef.pth'))
        self.delta_CLr_lef_mean = torch.mean(self.safe_read_dat(r'CL1620_ALPHA2_607.dat'))
        self.delta_CLr_lef_std = torch.std(self.safe_read_dat(r'CL1620_ALPHA2_607.dat'))
        self.delta_CLr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CLr_lef_model.load_state_dict(torch.load(path + '/model/delta_CLr_lef.pth'))
        self.delta_CLp_lef_mean = torch.mean(self.safe_read_dat(r'CL1520_ALPHA2_609.dat'))
        self.delta_CLp_lef_std = torch.std(self.safe_read_dat(r'CL1520_ALPHA2_609.dat'))
        self.delta_CLp_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CLp_lef_model.load_state_dict(torch.load(path + '/model/delta_CLp_lef.pth'))
        self.delta_CMq_lef_mean = torch.mean(self.safe_read_dat(r'CM1420_ALPHA2_105.dat'))
        self.delta_CMq_lef_std = torch.std(self.safe_read_dat(r'CM1420_ALPHA2_105.dat'))
        self.delta_CMq_lef_model = MLP(1, 1, [40, 20]).to(device=device)
        self.delta_CMq_lef_model.load_state_dict(torch.load(path + '/model/delta_CMq_lef.pth'))
        self.delta_CNr_lef_mean = torch.mean(self.safe_read_dat(r'CN1620_ALPHA2_507.dat'))
        self.delta_CNr_lef_std = torch.std(self.safe_read_dat(r'CN1620_ALPHA2_507.dat'))
        self.delta_CNr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CNr_lef_model.load_state_dict(torch.load(path + '/model/delta_CNr_lef.pth'))
        self.delta_CNp_lef_mean = torch.mean(self.safe_read_dat(r'CN1520_ALPHA2_509.dat'))
        self.delta_CNp_lef_std = torch.std(self.safe_read_dat(r'CN1520_ALPHA2_509.dat'))
        self.delta_CNp_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CNp_lef_model.load_state_dict(torch.load(path + '/model/delta_CNp_lef.pth'))
        self.Cy_r30_mean = torch.mean(self.safe_read_dat(r'CY0720_ALPHA1_BETA1_405.dat'))
        self.Cy_r30_std = torch.std(self.safe_read_dat(r'CY0720_ALPHA1_BETA1_405.dat'))
        self.Cy_r30_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cy_r30_model.load_state_dict(torch.load(path + '/model/Cy_r30.pth'))
        self.Cn_r30_mean = torch.mean(self.safe_read_dat(r'CN0720_ALPHA1_BETA1_503.dat'))
        self.Cn_r30_std = torch.std(self.safe_read_dat(r'CN0720_ALPHA1_BETA1_503.dat'))
        self.Cn_r30_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cn_r30_model.load_state_dict(torch.load(path + '/model/Cn_r30.pth'))
        self.Cl_r30_mean = torch.mean(self.safe_read_dat(r'CL0720_ALPHA1_BETA1_603.dat'))
        self.Cl_r30_std = torch.std(self.safe_read_dat(r'CL0720_ALPHA1_BETA1_603.dat'))
        self.Cl_r30_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cl_r30_model.load_state_dict(torch.load(path + '/model/Cl_r30.pth'))
        self.Cy_a20_mean = torch.mean(self.safe_read_dat(r'CY0620_ALPHA1_BETA1_403.dat'))
        self.Cy_a20_std = torch.std(self.safe_read_dat(r'CY0620_ALPHA1_BETA1_403.dat'))
        self.Cy_a20_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cy_a20_model.load_state_dict(torch.load(path + '/model/Cy_a20.pth'))
        self.Cy_a20_lef_mean = torch.mean(self.safe_read_dat(r'CY0920_ALPHA2_BETA1_404.dat'))
        self.Cy_a20_lef_std = torch.std(self.safe_read_dat(r'CY0920_ALPHA2_BETA1_404.dat')) 
        self.Cy_a20_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cy_a20_lef_model.load_state_dict(torch.load(path + '/model/Cy_a20_lef.pth'))
        self.Cn_a20_mean = torch.mean(self.safe_read_dat(r'CN0620_ALPHA1_BETA1_504.dat'))
        self.Cn_a20_std = torch.std(self.safe_read_dat(r'CN0620_ALPHA1_BETA1_504.dat'))
        self.Cn_a20_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cn_a20_model.load_state_dict(torch.load(path + '/model/Cn_a20.pth'))
        self.Cn_a20_lef_mean = torch.mean(self.safe_read_dat(r'CN0920_ALPHA2_BETA1_505.dat'))
        self.Cn_a20_lef_std = torch.std(self.safe_read_dat(r'CN0920_ALPHA2_BETA1_505.dat'))
        self.Cn_a20_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cn_a20_lef_model.load_state_dict(torch.load(path + '/model/Cn_a20_lef.pth'))
        self.Cl_a20_mean = torch.mean(self.safe_read_dat(r'CL0620_ALPHA1_BETA1_604.dat'))
        self.Cl_a20_std = torch.std(self.safe_read_dat(r'CL0620_ALPHA1_BETA1_604.dat'))
        self.Cl_a20_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cl_a20_model.load_state_dict(torch.load(path + '/model/Cl_a20.pth'))
        self.Cl_a20_lef_mean = torch.mean(self.safe_read_dat(r'CL0920_ALPHA2_BETA1_605.dat'))
        self.Cl_a20_lef_std = torch.std(self.safe_read_dat(r'CL0920_ALPHA2_BETA1_605.dat'))
        self.Cl_a20_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cl_a20_lef_model.load_state_dict(torch.load(path + '/model/Cl_a20_lef.pth'))
        self.delta_CNbeta_mean = torch.mean(self.safe_read_dat(r'CN9999_ALPHA1_brett.dat'))
        self.delta_CNbeta_std = torch.std(self.safe_read_dat(r'CN9999_ALPHA1_brett.dat'))
        self.delta_CNbeta_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CNbeta_model.load_state_dict(torch.load(path + '/model/delta_CNbeta.pth'))
        self.delta_CLbeta_mean = torch.mean(self.safe_read_dat(r'CL9999_ALPHA1_brett.dat'))
        self.delta_CLbeta_std = torch.std(self.safe_read_dat(r'CL9999_ALPHA1_brett.dat')) 
        self.delta_CLbeta_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_CLbeta_model.load_state_dict(torch.load(path + '/model/delta_CLbeta.pth'))
        self.delta_Cm_mean = torch.mean(self.safe_read_dat(r'CM9999_ALPHA1_brett.dat'))
        self.delta_Cm_std = torch.std(self.safe_read_dat(r'CM9999_ALPHA1_brett.dat'))
        self.delta_Cm_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cm_model.load_state_dict(torch.load(path + '/model/delta_Cm.pth'))
        self.eta_el_mean = torch.mean(self.safe_read_dat(r'ETA_DH1_brett.dat'))
        self.eta_el_std = torch.std(self.safe_read_dat(r'ETA_DH1_brett.dat'))
        self.eta_el_model = MLP(1, 1, [20, 10]).to(device=device)
        self.eta_el_model.load_state_dict(torch.load(path + '/model/eta_el.pth'))

    def safe_read_dat(self, dat_name):
        try:
            if dat_name in HIFI_GLOBAL_TXT_CONTENT:
                return HIFI_GLOBAL_TXT_CONTENT.get(dat_name)

            data_path = path + r'/data/' + dat_name
            with open(data_path, 'r', encoding='utf-8') as file:
                content = file.read()
                content = content.strip()
                data_str = [value for value in content.split(' ') if value]
                data = list(map(float, data_str))
                data = torch.tensor(data, device=torch.device(device))
                HIFI_GLOBAL_TXT_CONTENT[dat_name] = data
                return data
        except OSError:
            print("Cannot find file {} in current directory".format(data_path))
            return []

    def _Cx(self, alpha, beta, dele):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        dele = normalize(dele, self.DH1_mean, self.DH1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        return unnormalize(self.Cx_model.forward(input), self.Cx_mean, self.Cx_std)

    def _Cz(self, alpha, beta, dele):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        dele = normalize(dele, self.DH1_mean, self.DH1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        return unnormalize(self.Cz_model.forward(input), self.Cz_mean, self.Cz_std)

    def _Cm(self, alpha, beta, dele):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        dele = normalize(dele, self.DH1_mean, self.DH1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        return unnormalize(self.Cm_model.forward(input), self.Cm_mean, self.Cm_std)

    def _Cy(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cy_model.forward(input), self.Cy_mean, self.Cy_std)

    def _Cn(self, alpha, beta, dele):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        dele = normalize(dele, self.DH2_mean, self.DH2_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        return unnormalize(self.Cn_model.forward(input), self.Cn_mean, self.Cn_std)

    def _Cl(self, alpha, beta, dele):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        dele = normalize(dele, self.DH2_mean, self.DH2_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        return unnormalize(self.Cl_model.forward(input), self.Cl_mean, self.Cl_std)

    def _Cx_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cx_lef_model.forward(input), self.Cx_lef_mean, self.Cx_lef_std)

    def _Cz_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cz_lef_model.forward(input), self.Cz_lef_mean, self.Cz_lef_std)

    def _Cm_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cm_lef_model.forward(input), self.Cm_lef_mean, self.Cm_lef_std)

    def _Cy_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cy_lef_model.forward(input), self.Cy_lef_mean, self.Cy_lef_std)

    def _Cn_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cn_lef_model.forward(input), self.Cn_lef_mean, self.Cn_lef_std)

    def _Cl_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cl_lef_model.forward(input), self.Cl_lef_mean, self.Cl_lef_std)

    def _CXq(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CXq_model.forward(input), self.CXq_mean, self.CXq_std)

    def _CZq(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CZq_model.forward(input), self.CZq_mean, self.CZq_std)

    def _CMq(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CMq_model.forward(input), self.CMq_mean, self.CMq_std)

    def _CYp(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CYp_model.forward(input), self.CYp_mean, self.CYp_std)

    def _CYr(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CYr_model.forward(input), self.CYr_mean, self.CYr_std)

    def _CNr(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CNr_model.forward(input), self.CNr_mean, self.CNr_std)

    def _CNp(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CNp_model.forward(input), self.CNp_mean, self.CNp_std)

    def _CLp(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CLp_model.forward(input), self.CLp_mean, self.CLp_std)

    def _CLr(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.CLr_model.forward(input), self.CLr_mean, self.CLr_std)

    def _delta_CXq_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CXq_lef_model.forward(input), self.delta_CXq_lef_mean, self.delta_CXq_lef_std)

    def _delta_CYr_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CYr_lef_model.forward(input), self.delta_CYr_lef_mean, self.delta_CYr_lef_std)

    def _delta_CYp_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CYp_lef_model.forward(input), self.delta_CYp_lef_mean, self.delta_CYp_lef_std)

    def _delta_CZq_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CZq_lef_model.forward(input), self.delta_CZq_lef_mean, self.delta_CZq_lef_std)

    def _delta_CLr_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CLr_lef_model.forward(input), self.delta_CLr_lef_mean, self.delta_CLr_lef_std)

    def _delta_CLp_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CLp_lef_model.forward(input), self.delta_CLp_lef_mean, self.delta_CLp_lef_std)

    def _delta_CMq_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CMq_lef_model.forward(input), self.delta_CMq_lef_mean, self.delta_CMq_lef_std)

    def _delta_CNr_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CNr_lef_model.forward(input), self.delta_CNr_lef_mean, self.delta_CNr_lef_std)

    def _delta_CNp_lef(self, alpha):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CNp_lef_model.forward(input), self.delta_CNp_lef_mean, self.delta_CNp_lef_std)

    def _Cy_r30(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cy_r30_model.forward(input), self.Cy_r30_mean, self.Cy_r30_std)

    def _Cn_r30(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cn_r30_model.forward(input), self.Cn_r30_mean, self.Cn_r30_std)

    def _Cl_r30(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cl_r30_model.forward(input), self.Cl_r30_mean, self.Cl_r30_std)

    def _Cy_a20(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cy_a20_model.forward(input), self.Cy_a20_mean, self.Cy_a20_std)

    def _Cy_a20_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cy_a20_lef_model.forward(input), self.Cy_a20_lef_mean, self.Cy_a20_lef_std)

    def _Cn_a20(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cn_a20_model.forward(input), self.Cn_a20_mean, self.Cn_a20_std)

    def _Cn_a20_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cn_a20_lef_model.forward(input), self.Cn_a20_lef_mean, self.Cn_a20_lef_std)

    def _Cl_a20(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cl_a20_model.forward(input), self.Cl_a20_mean, self.Cl_a20_std)

    def _Cl_a20_lef(self, alpha, beta):
        alpha = normalize(alpha, self.ALPHA2_mean, self.ALPHA2_std)
        beta = normalize(beta, self.BETA1_mean, self.BETA1_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        return unnormalize(self.Cl_a20_lef_model.forward(input), self.Cl_a20_lef_mean, self.Cl_a20_lef_std)

    def _delta_CNbeta(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CNbeta_model.forward(input), self.delta_CNbeta_mean, self.delta_CNbeta_std)

    def _delta_CLbeta(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_CLbeta_model.forward(input), self.delta_CLbeta_mean, self.delta_CLbeta_std)

    def _delta_Cm(self, alpha):
        alpha = normalize(alpha, self.ALPHA1_mean, self.ALPHA1_std)
        input = alpha.reshape(-1, 1)
        return unnormalize(self.delta_Cm_model.forward(input), self.delta_Cm_mean, self.delta_Cm_std)

    def _eta_el(self, el):
        el = normalize(el, self.DH1_mean, self.DH1_std)
        input = el.reshape(-1, 1)
        return unnormalize(self.eta_el_model.forward(input), self.eta_el_mean, self.eta_el_std)

    def hifi_C(self, alpha, beta, el):
        return (
            self._Cx(alpha, beta, el),
            self._Cz(alpha, beta, el),
            self._Cm(alpha, beta, el),
            self._Cy(alpha, beta),
            self._Cn(alpha, beta, el),
            self._Cl(alpha, beta, el),
        )

    def hifi_damping(self, alpha):
        return (
            self._CXq(alpha),
            self._CYr(alpha),
            self._CYp(alpha),
            self._CZq(alpha),
            self._CLr(alpha),
            self._CLp(alpha),
            self._CMq(alpha),
            self._CNr(alpha),
            self._CNp(alpha),
        )

    def hifi_C_lef(self, alpha, beta):
        zero = torch.zeros_like(alpha)
        return (
            self._Cx_lef(alpha, beta) - self._Cx(alpha, beta, zero),
            self._Cz_lef(alpha, beta) - self._Cz(alpha, beta, zero),
            self._Cm_lef(alpha, beta) - self._Cm(alpha, beta, zero),
            self._Cy_lef(alpha, beta) - self._Cy(alpha, beta),
            self._Cn_lef(alpha, beta) - self._Cn(alpha, beta, zero),
            self._Cl_lef(alpha, beta) - self._Cl(alpha, beta, zero),
        )

    def hifi_damping_lef(self, alpha):
        return (
            self._delta_CXq_lef(alpha),
            self._delta_CYr_lef(alpha),
            self._delta_CYp_lef(alpha),
            self._delta_CZq_lef(alpha),
            self._delta_CLr_lef(alpha),
            self._delta_CLp_lef(alpha),
            self._delta_CMq_lef(alpha),
            self._delta_CNr_lef(alpha),
            self._delta_CNp_lef(alpha),
        )

    def hifi_rudder(self, alpha, beta):
        zero = torch.zeros_like(alpha)
        return (
            self._Cy_r30(alpha, beta) - self._Cy(alpha, beta),
            self._Cn_r30(alpha, beta) - self._Cn(alpha, beta, zero),
            self._Cl_r30(alpha, beta) - self._Cl(alpha, beta, zero),
        )

    def hifi_ailerons(self, alpha, beta):
        zero = torch.zeros_like(alpha)
        return (
            self._Cy_a20(alpha, beta) - self._Cy(alpha, beta),
            self._Cy_a20_lef(alpha, beta) - self._Cy_lef(alpha, beta) -
            (self._Cy_a20(alpha, beta) - self._Cy(alpha, beta)),
            self._Cn_a20(alpha, beta) - self._Cn(alpha, beta, zero),
            self._Cn_a20_lef(alpha, beta) - self._Cn_lef(alpha, beta) -
            (self._Cn_a20(alpha, beta) - self._Cn(alpha, beta, zero)),
            self._Cl_a20(alpha, beta) - self._Cl(alpha, beta, zero),
            self._Cl_a20_lef(alpha, beta) - self._Cl_lef(alpha, beta) -
            (self._Cl_a20(alpha, beta) - self._Cl(alpha, beta, zero)),
        )

    def hifi_other_coeffs(self, alpha, el):
        zero = torch.zeros_like(alpha)
        return (
            self._delta_CNbeta(alpha),
            self._delta_CLbeta(alpha),
            self._delta_Cm(alpha),
            self._eta_el(el),
            zero,
        )
