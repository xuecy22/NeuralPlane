import torch
import torch.nn as nn
import pandas as pd
import os


HIFI_GLOBAL_TXT_CONTENT = {}
device = "cuda:0"
path = os.path.dirname(os.path.realpath(__file__))


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
    def __init__(self, device=device):
        super().__init__()
        self.data = pd.read_csv(path +'/model/mean_std.csv')
        self.Cx_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cx_model.load_state_dict(torch.load(path + '/model/Cx.pth', map_location=device))
        self.Cz_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cz_model.load_state_dict(torch.load(path + '/model/Cz.pth', map_location=device))
        self.Cm_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cm_model.load_state_dict(torch.load(path + '/model/Cm.pth', map_location=device))
        self.Cy_model = MLP(2, 1, [20, 10]).to(device=device)
        self.Cy_model.load_state_dict(torch.load(path + '/model/Cy.pth', map_location=device))
        self.Cn_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cn_model.load_state_dict(torch.load(path + '/model/Cn.pth', map_location=device))
        self.Cl_model = MLP(3, 1, [20, 10]).to(device=device)
        self.Cl_model.load_state_dict(torch.load(path + '/model/Cl.pth', map_location=device))
        self.Cxq_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Cxq_model.load_state_dict(torch.load(path + '/model/Cxq.pth', map_location=device))
        self.Czq_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Czq_model.load_state_dict(torch.load(path + '/model/Czq.pth', map_location=device))
        self.Cmq_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Cmq_model.load_state_dict(torch.load(path + '/model/Cmq.pth', map_location=device))
        self.Cyp_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Cyp_model.load_state_dict(torch.load(path + '/model/Cyp.pth', map_location=device))
        self.Cyr_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Cyr_model.load_state_dict(torch.load(path + '/model/Cyr.pth', map_location=device))
        self.Cnr_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Cnr_model.load_state_dict(torch.load(path + '/model/Cnr.pth', map_location=device))
        self.Cnp_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Cnp_model.load_state_dict(torch.load(path + '/model/Cnp.pth', map_location=device))
        self.Clp_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Clp_model.load_state_dict(torch.load(path + '/model/Clp.pth', map_location=device))
        self.Clr_model = MLP(1, 1, [20, 10]).to(device=device)
        self.Clr_model.load_state_dict(torch.load(path + '/model/Clr.pth', map_location=device))
        self.delta_Cx_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.delta_Cx_lef_model.load_state_dict(torch.load(path + '/model/delta_Cx_lef.pth', map_location=device))
        self.delta_Cz_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cz_lef_model.load_state_dict(torch.load(path + '/model/delta_Cz_lef.pth', map_location=device))
        self.delta_Cm_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cm_lef_model.load_state_dict(torch.load(path + '/model/delta_Cm_lef.pth', map_location=device))
        self.delta_Cy_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cy_lef_model.load_state_dict(torch.load(path + '/model/delta_Cy_lef.pth', map_location=device))
        self.delta_Cn_lef_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cn_lef_model.load_state_dict(torch.load(path + '/model/delta_Cn_lef.pth', map_location=device))
        self.delta_Cl_lef_model = MLP(2, 1, [20, 10]).to(device=device)
        self.delta_Cl_lef_model.load_state_dict(torch.load(path + '/model/delta_Cl_lef.pth', map_location=device))
        self.delta_Cxq_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cxq_lef_model.load_state_dict(torch.load(path + '/model/delta_Cxq_lef.pth', map_location=device))
        self.delta_Cyr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cyr_lef_model.load_state_dict(torch.load(path + '/model/delta_Cyr_lef.pth', map_location=device))
        self.delta_Cyp_lef_model = MLP(1, 1, [20, 10, 5]).to(device=device)
        self.delta_Cyp_lef_model.load_state_dict(torch.load(path + '/model/delta_Cyp_lef.pth', map_location=device))
        self.delta_Czq_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Czq_lef_model.load_state_dict(torch.load(path + '/model/delta_Czq_lef.pth', map_location=device))
        self.delta_Clr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Clr_lef_model.load_state_dict(torch.load(path + '/model/delta_Clr_lef.pth', map_location=device))
        self.delta_Clp_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Clp_lef_model.load_state_dict(torch.load(path + '/model/delta_Clp_lef.pth', map_location=device))
        self.delta_Cmq_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cmq_lef_model.load_state_dict(torch.load(path + '/model/delta_Cmq_lef.pth', map_location=device))
        self.delta_Cnr_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cnr_lef_model.load_state_dict(torch.load(path + '/model/delta_Cnr_lef.pth', map_location=device))
        self.delta_Cnp_lef_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cnp_lef_model.load_state_dict(torch.load(path + '/model/delta_Cnp_lef.pth', map_location=device))
        self.delta_Cy_r30_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cy_r30_model.load_state_dict(torch.load(path + '/model/delta_Cy_r30.pth', map_location=device))
        self.delta_Cn_r30_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cn_r30_model.load_state_dict(torch.load(path + '/model/delta_Cn_r30.pth', map_location=device))
        self.delta_Cl_r30_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cl_r30_model.load_state_dict(torch.load(path + '/model/delta_Cl_r30.pth', map_location=device))
        self.delta_Cy_a20_model = MLP(2, 1, [20, 10, 10]).to(device=device)
        self.delta_Cy_a20_model.load_state_dict(torch.load(path + '/model/delta_Cy_a20.pth', map_location=device))
        self.delta_Cy_a20_lef_model = MLP(2, 1, [20, 20, 10]).to(device=device)
        self.delta_Cy_a20_lef_model.load_state_dict(torch.load(path + '/model/delta_Cy_a20_lef.pth', map_location=device))
        self.delta_Cn_a20_model = MLP(2, 1, [20, 10, 5]).to(device=device)
        self.delta_Cn_a20_model.load_state_dict(torch.load(path + '/model/delta_Cn_a20.pth', map_location=device))
        self.delta_Cn_a20_lef_model = MLP(2, 1, [20, 20, 10]).to(device=device)
        self.delta_Cn_a20_lef_model.load_state_dict(torch.load(path + '/model/delta_Cn_a20_lef.pth', map_location=device))
        self.delta_Cl_a20_model = MLP(2, 1, [20, 10]).to(device=device)
        self.delta_Cl_a20_model.load_state_dict(torch.load(path + '/model/delta_Cl_a20.pth', map_location=device))
        self.delta_Cl_a20_lef_model = MLP(2, 1, [20, 20, 10]).to(device=device)
        self.delta_Cl_a20_lef_model.load_state_dict(torch.load(path + '/model/delta_Cl_a20_lef.pth', map_location=device))
        self.delta_Cnbeta_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cnbeta_model.load_state_dict(torch.load(path + '/model/delta_Cnbeta.pth', map_location=device))
        self.delta_Clbeta_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Clbeta_model.load_state_dict(torch.load(path + '/model/delta_Clbeta.pth', map_location=device))
        self.delta_Cm_model = MLP(1, 1, [20, 10]).to(device=device)
        self.delta_Cm_model.load_state_dict(torch.load(path + '/model/delta_Cm.pth', map_location=device))
        self.eta_el_model = MLP(1, 1, [20, 10]).to(device=device)
        self.eta_el_model.load_state_dict(torch.load(path + '/model/eta_el.pth', map_location=device))

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
        
    @torch.no_grad()
    def _Cx(self, alpha, beta, dele):
        name = self.data['name']
        index = list(name).index('Cx')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        el_mean = self.data['el_mean'][index]
        el_std = self.data['el_std'][index]
        dele = normalize(dele, el_mean, el_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cx_model.forward(input), mean, std)

    @torch.no_grad()
    def _Cz(self, alpha, beta, dele):
        name = self.data['name']
        index = list(name).index('Cz')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        el_mean = self.data['el_mean'][index]
        el_std = self.data['el_std'][index]
        dele = normalize(dele, el_mean, el_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cz_model.forward(input), mean, std)

    @torch.no_grad()
    def _Cm(self, alpha, beta, dele):
        name = self.data['name']
        index = list(name).index('Cm')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        el_mean = self.data['el_mean'][index]
        el_std = self.data['el_std'][index]
        dele = normalize(dele, el_mean, el_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cm_model.forward(input), mean, std)

    @torch.no_grad()
    def _Cy(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('Cy')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cy_model.forward(input), mean, std)

    @torch.no_grad()
    def _Cn(self, alpha, beta, dele):
        name = self.data['name']
        index = list(name).index('Cn')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        el_mean = self.data['el_mean'][index]
        el_std = self.data['el_std'][index]
        dele = normalize(dele, el_mean, el_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cn_model.forward(input), mean, std)

    @torch.no_grad()
    def _Cl(self, alpha, beta, dele):
        name = self.data['name']
        index = list(name).index('Cl')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        el_mean = self.data['el_mean'][index]
        el_std = self.data['el_std'][index]
        dele = normalize(dele, el_mean, el_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        input = torch.hstack((input, dele.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cl_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cx_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cx_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cx_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cz_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cz_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cz_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cm_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cm_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cm_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cy_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cy_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cy_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cn_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cn_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cn_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cl_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cl_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cl_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _CXq(self, alpha):
        name = self.data['name']
        index = list(name).index('Cxq')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cxq_model.forward(input), mean, std)

    @torch.no_grad()
    def _CZq(self, alpha):
        name = self.data['name']
        index = list(name).index('Czq')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Czq_model.forward(input), mean, std)

    @torch.no_grad()
    def _CMq(self, alpha):
        name = self.data['name']
        index = list(name).index('Cmq')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cmq_model.forward(input), mean, std)

    @torch.no_grad()
    def _CYp(self, alpha):
        name = self.data['name']
        index = list(name).index('Cyp')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cyp_model.forward(input), mean, std)

    @torch.no_grad()
    def _CYr(self, alpha):
        name = self.data['name']
        index = list(name).index('Cyr')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cyr_model.forward(input), mean, std)

    @torch.no_grad()
    def _CNr(self, alpha):
        name = self.data['name']
        index = list(name).index('Cnr')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cnr_model.forward(input), mean, std)

    @torch.no_grad()
    def _CNp(self, alpha):
        name = self.data['name']
        index = list(name).index('Cnp')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Cnp_model.forward(input), mean, std)

    @torch.no_grad()
    def _CLp(self, alpha):
        name = self.data['name']
        index = list(name).index('Clp')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Clp_model.forward(input), mean, std)

    @torch.no_grad()
    def _CLr(self, alpha):
        name = self.data['name']
        index = list(name).index('Clr')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.Clr_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CXq_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cxq_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cxq_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CYr_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cyr_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cyr_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CYp_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cyp_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cyp_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CZq_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Czq_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Czq_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CLr_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Clr_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Clr_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CLp_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Clp_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Clp_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CMq_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cmq_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cmq_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CNr_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cnr_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cnr_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CNp_lef(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cnp_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cnp_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cy_r30(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cy_r30')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cy_r30_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cn_r30(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cn_r30')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cn_r30_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cl_r30(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cl_r30')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cl_r30_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cy_a20(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cy_a20')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cy_a20_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cy_a20_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cy_a20_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cy_a20_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cn_a20(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cn_a20')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cn_a20_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cn_a20_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cn_a20_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cn_a20_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cl_a20(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cl_a20')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cl_a20_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cl_a20_lef(self, alpha, beta):
        name = self.data['name']
        index = list(name).index('delta_Cl_a20_lef')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        beta_mean = self.data['beta_mean'][index]
        beta_std = self.data['beta_std'][index]
        beta = normalize(beta, beta_mean, beta_std)
        input = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cl_a20_lef_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CNbeta(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cnbeta')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cnbeta_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_CLbeta(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Clbeta')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Clbeta_model.forward(input), mean, std)

    @torch.no_grad()
    def _delta_Cm(self, alpha):
        name = self.data['name']
        index = list(name).index('delta_Cm')
        alpha_mean = self.data['alpha_mean'][index]
        alpha_std = self.data['alpha_std'][index]
        alpha = normalize(alpha, alpha_mean, alpha_std)
        input = alpha.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.delta_Cm_model.forward(input), mean, std)

    @torch.no_grad()
    def _eta_el(self, el):
        name = self.data['name']
        index = list(name).index('eta_el')
        el_mean = self.data['el_mean'][index]
        el_std = self.data['el_std'][index]
        el = normalize(el, el_mean, el_std)
        input = el.reshape(-1, 1)
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return unnormalize(self.eta_el_model.forward(input), mean, std)

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
        return (
            self._delta_Cx_lef(alpha, beta),
            self._delta_Cz_lef(alpha, beta),
            self._delta_Cm_lef(alpha, beta),
            self._delta_Cy_lef(alpha, beta),
            self._delta_Cn_lef(alpha, beta),
            self._delta_Cl_lef(alpha, beta),
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
        return (
            self._delta_Cy_r30(alpha, beta),
            self._delta_Cn_r30(alpha, beta),
            self._delta_Cl_r30(alpha, beta),
        )

    def hifi_ailerons(self, alpha, beta):
        return (
            self._delta_Cy_a20(alpha, beta),
            self._delta_Cy_a20_lef(alpha, beta),
            self._delta_Cn_a20(alpha, beta),
            self._delta_Cn_a20_lef(alpha, beta),
            self._delta_Cl_a20(alpha, beta),
            self._delta_Cl_a20_lef(alpha, beta),
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
