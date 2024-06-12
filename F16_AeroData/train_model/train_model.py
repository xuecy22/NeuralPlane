import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import csv
import matplotlib.pyplot as plt
from hifi_F16_AeroData import hifi_F16
import pandas as pd
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
        self.out_dim = out_dim

    def forward(self, x):
        x = x.to(torch.float32)
        ret = self.layers(x)
        return ret


class MyDataset(Dataset):
    def __init__(self, input, output, transform=None):
        super().__init__()
        self.transform = transform
        self.input = input
        self.output = output
    
    def __getitem__(self, index):
        input = self.input[index]
        output = self.output[index]
        return input, output

    def __len__(self):
        return len(self.input)


def safe_read_dat(dat_name):
    try:
        path = r'../data/' + dat_name
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            content = content.strip()
            data_str = [value for value in content.split(' ') if value]
            data = list(map(float, data_str))
            data = np.array(data)
            return data
    except OSError:
        print("Cannot find file {} in current directory".format(path))
        return []


def normalize(X):
    return (X - torch.mean(X)) / torch.std(X)

def _t2n(x):
    return x.detach().cpu().numpy()

# def generate_data3d(X, Y):
#     train_X = None
#     train_Y = None
#     for i in range(X[0].shape[0]):
#         for j in range(X[1].shape[0]):
#             for k in range(X[2].shape[0]):
#                 index = i + X[0].shape[0] * j + X[0].shape[0] * X[1].shape[0] * k
#                 if train_X is None:
#                     train_X = np.array([X[0][i], X[1][j], X[2][k]])
#                     train_Y = np.array([Y[index]])
#                 else:
#                     train_X = np.vstack((train_X, np.array([X[0][i], X[1][j], X[2][k]])))
#                     train_Y = np.vstack((train_Y, np.array([Y[index]])))
#     return train_X, train_Y


# def generate_data2d(X, Y):
#     train_X = None
#     train_Y = None
#     for i in range(X[0].shape[0]):
#         for j in range(X[1].shape[0]):
#             index = i + X[0].shape[0] * j
#             if train_X is None:
#                 train_X = np.array([X[0][i], X[1][j]])
#                 train_Y = np.array([Y[index]])
#             else:
#                 train_X = np.vstack((train_X, np.array([X[0][i], X[1][j]])))
#                 train_Y = np.vstack((train_Y, np.array([Y[index]])))
#     return train_X, train_Y


# def generate_data1d(X, Y):
#     train_X = None
#     train_Y = None
#     for i in range(X[0].shape[0]):
#         index = i
#         if train_X is None:
#             train_X = np.array([X[0][i]])
#             train_Y = np.array([Y[index]])
#         else:
#             train_X = np.vstack((train_X, np.array([X[0][i]])))
#             train_Y = np.vstack((train_Y, np.array([Y[index]])))
#     return train_X, train_Y


# def interpn(X, Y):
#     train_X = None
#     train_Y = None
#     for i in range(X.shape[0] - 1):
#         deltaX = X[i + 1, 0] - X[i, 0]
#         deltaY = Y[i + 1, 0] - Y[i, 0]
#         point_num = math.ceil(deltaX / 0.005)
#         for j in range(point_num):
#             if train_X is None:
#                 train_X = np.array([X[i, 0] + j * deltaX / point_num])
#                 train_Y = np.array([Y[i, 0] + j * deltaY / point_num])
#             else:
#                 train_X = np.vstack((train_X, np.array([X[i, 0] + j * deltaX / point_num])))
#                 train_Y = np.vstack((train_Y, np.array([Y[i, 0] + j * deltaY / point_num])))
#     return train_X, train_Y

def adjust_opt(optimizer, epoch):
    if epoch == 500:
        lr = 5e-3
    elif epoch == 750:
        lr = 1e-3
    elif epoch == 900:
        lr = 5e-4
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_X, train_Y, file_name):
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, shuffle=True)
    train_db = MyDataset(X_train, y_train)
    test_db = MyDataset(X_test, y_test)
    BATCH_SIZE = 32
    train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=BATCH_SIZE, shuffle=True)
    model = MLP(3, 1, [20, 10])
    model = model.to(device)
    loss = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.006, momentum=0.9, weight_decay=5e-4)
    num_epochs = 1000
    train_loss_list = []
    train_r2_list = []
    test_loss_list = []
    test_r2_list = []
    max_test_r2 = 0.97
    best_model = {}
    min_test_loss = 0
    for epoch in range(num_epochs):
        train_loss = 0
        pred_all = None
        output_all = None
        model.train()
        adjust_opt(optimizer, epoch)
        for step, data in enumerate(train_loader):
            X, y = data
            X = X.to(device)
            X = X.type(torch.cuda.FloatTensor)
            y = y.to(device)
            y = y.type(torch.cuda.FloatTensor)
            out = model(X)
            loss_value = loss(out, y)
            optimizer.zero_grad()
            loss_value.backward(retain_graph=True)
            optimizer.step()
            train_loss += float(loss_value)
            if output_all is None:
                output_all = y
            else:
                output_all = torch.cat([output_all, y])
            if pred_all is None:
                pred_all = out
            else:
                pred_all = torch.cat([pred_all, out])
        train_loss_list.append(train_loss / len(train_loader))
        train_y = output_all.cpu().detach().numpy()
        train_pred = pred_all.cpu().detach().numpy()
        train_r2 = r2_score(train_y, train_pred)
        train_r2_list.append(train_r2)
        print('epoch', epoch, ':')
        print('train_loss:', train_loss_list[-1])
        print('train_r2:', train_r2_list[-1])
        test_loss = 0
        pred_all = None
        output_all = None
        model.eval()
        for step, data in enumerate(test_loader):
            X, y = data
            X = X.to(device)
            X = X.type(torch.cuda.FloatTensor)
            y = y.to(device)
            y = y.type(torch.cuda.FloatTensor)
            out = model(X)
            loss_value = loss(out, y)
            test_loss += float(loss_value)
            if output_all is None:
                output_all = y
            else:
                output_all = torch.cat([output_all, y])
            if pred_all is None:
                pred_all = out
            else:
                pred_all = torch.cat([pred_all, out])
            torch.cuda.empty_cache()
        test_y = output_all.cpu().detach().numpy()
        test_pred = pred_all.cpu().detach().numpy()
        test_loss_list.append(test_loss / len(test_loader))
        test_r2 = r2_score(test_y, test_pred)
        test_r2_list.append(test_r2)
        print('test_loss:', test_loss_list[-1])
        print('test_r2:', test_r2_list[-1])
        print('max_test_r2:', max_test_r2)
        if test_r2 > max_test_r2:
            best_model = model.state_dict()
            min_test_loss = test_loss_list[-1]
            max_test_r2 = test_r2
    torch.save(best_model, "./model1/" + file_name + "-" + str(max_test_r2) + "-" + str(min_test_loss) + ".pth")
    # tmp = open("./train_result/" + file_name + "_result_loss" + ".csv", 'w', newline='')
    # csv_write = csv.writer(tmp)
    # csv_write.writerow(["train_loss", "test_loss", "test_r2"])
    # for i in range(len(train_loss_list)):
    #     csv_write.writerow([train_loss_list[i], test_loss_list[i], test_r2_list[i]])
    # tmp.close()


ALPHA1 = safe_read_dat(r'ALPHA1.dat')
ALPHA2 = safe_read_dat(r'ALPHA2.dat')
BETA1 = safe_read_dat(r'BETA1.dat')
DH1 = safe_read_dat(r'DH1.dat')
DH2 = safe_read_dat(r'DH2.dat')
# Cx = safe_read_dat(r'CX0120_ALPHA1_BETA1_DH1_201.dat')
# Cx = normalize(Cx)
# Cz = safe_read_dat(r'CZ0120_ALPHA1_BETA1_DH1_301.dat')
# Cz = normalize(Cz)
# Cm = safe_read_dat(r'CM0120_ALPHA1_BETA1_DH1_101.dat')
# Cm = normalize(Cm)
# Cy = safe_read_dat(r'CY0320_ALPHA1_BETA1_401.dat')
# Cy = normalize(Cy)
# Cn = safe_read_dat(r'CN0120_ALPHA1_BETA1_DH2_501.dat')
# Cn = normalize(Cn)
# Cl = safe_read_dat(r'CL0120_ALPHA1_BETA1_DH2_601.dat')
# Cl = normalize(Cl)
# Cx_lef = safe_read_dat(r'CX0820_ALPHA2_BETA1_202.dat')
# Cx_lef = normalize(Cx_lef)
# Cz_lef = safe_read_dat(r'CZ0820_ALPHA2_BETA1_302.dat')
# Cz_lef = normalize(Cz_lef)
# Cm_lef = safe_read_dat(r'CM0820_ALPHA2_BETA1_102.dat')
# Cm_lef = normalize(Cm_lef)
# Cy_lef = safe_read_dat(r'CY0820_ALPHA2_BETA1_402.dat')
# Cy_lef = normalize(Cy_lef)
# Cn_lef = safe_read_dat(r'CN0820_ALPHA2_BETA1_502.dat')
# Cn_lef = normalize(Cn_lef)
# Cl_lef = safe_read_dat(r'CL0820_ALPHA2_BETA1_602.dat')
# Cl_lef = normalize(Cl_lef)
# CXq = safe_read_dat(r'CX1120_ALPHA1_204.dat')
# CXq = normalize(CXq)
# CZq = safe_read_dat(r'CZ1120_ALPHA1_304.dat')
# CZq = normalize(CZq)
# CMq = safe_read_dat(r'CM1120_ALPHA1_104.dat')
# CMq = normalize(CMq)
# CYp = safe_read_dat(r'CY1220_ALPHA1_408.dat')
# CYp = normalize(CYp)
# CYr = safe_read_dat(r'CY1320_ALPHA1_406.dat')
# CYr = normalize(CYr)
# CNr = safe_read_dat(r'CN1320_ALPHA1_506.dat')
# CNr = normalize(CNr)
# CNp = safe_read_dat(r'CN1220_ALPHA1_508.dat')
# CNp = normalize(CNp)
# CLp = safe_read_dat(r'CL1220_ALPHA1_608.dat')
# CLp = normalize(CLp)
# CLr = safe_read_dat(r'CL1320_ALPHA1_606.dat')
# CLr = normalize(CLr)
# delta_CXq_lef = safe_read_dat(r'CX1420_ALPHA2_205.dat')
# delta_CXq_lef = normalize(delta_CXq_lef)
# delta_CYr_lef = safe_read_dat(r'CY1620_ALPHA2_407.dat')
# delta_CYr_lef = normalize(delta_CYr_lef)
# delta_CYp_lef = safe_read_dat(r'CY1520_ALPHA2_409.dat')
# delta_CYp_lef = normalize(delta_CYp_lef)
# delta_CZq_lef = safe_read_dat(r'CZ1420_ALPHA2_305.dat')
# delta_CZq_lef = normalize(delta_CZq_lef)
# delta_CLr_lef = safe_read_dat(r'CL1620_ALPHA2_607.dat')
# delta_CLr_lef = normalize(delta_CLr_lef)
# delta_CLp_lef = safe_read_dat(r'CL1520_ALPHA2_609.dat')
# delta_CLp_lef = normalize(delta_CLp_lef)
# delta_CMq_lef = safe_read_dat(r'CM1420_ALPHA2_105.dat')
# delta_CMq_lef = normalize(delta_CMq_lef)
# delta_CNr_lef = safe_read_dat(r'CN1620_ALPHA2_507.dat')
# delta_CNr_lef = normalize(delta_CNr_lef)
# delta_CNp_lef = safe_read_dat(r'CN1520_ALPHA2_509.dat')
# delta_CNp_lef = normalize(delta_CNp_lef)
# Cy_r30 = safe_read_dat(r'CY0720_ALPHA1_BETA1_405.dat')
# Cy_r30 = normalize(Cy_r30)
# Cn_r30 = safe_read_dat(r'CN0720_ALPHA1_BETA1_503.dat')
# Cn_r30 = normalize(Cn_r30)
# Cl_r30 = safe_read_dat(r'CL0720_ALPHA1_BETA1_603.dat')
# Cl_r30 = normalize(Cl_r30)
# Cy_a20 = safe_read_dat(r'CY0620_ALPHA1_BETA1_403.dat')
# Cy_a20 = normalize(Cy_a20)
# Cy_a20_lef = safe_read_dat(r'CY0920_ALPHA2_BETA1_404.dat')
# Cy_a20_lef = normalize(Cy_a20_lef)
# Cn_a20 = safe_read_dat(r'CN0620_ALPHA1_BETA1_504.dat')
# Cn_a20 = normalize(Cn_a20)
# Cn_a20_lef = safe_read_dat(r'CN0920_ALPHA2_BETA1_505.dat')
# Cn_a20_lef = normalize(Cn_a20_lef)
# Cl_a20 = safe_read_dat(r'CL0620_ALPHA1_BETA1_604.dat')
# Cl_a20 = normalize(Cl_a20)
# Cl_a20_lef = safe_read_dat(r'CL0920_ALPHA2_BETA1_605.dat')
# Cl_a20_lef = normalize(Cl_a20_lef)
# delta_CNbeta = safe_read_dat(r'CN9999_ALPHA1_brett.dat')
# delta_CNbeta = normalize(delta_CNbeta)
# delta_CLbeta = safe_read_dat(r'CL9999_ALPHA1_brett.dat')
# delta_CLbeta = normalize(delta_CLbeta)
# delta_Cm = safe_read_dat(r'CM9999_ALPHA1_brett.dat')
# delta_Cm = normalize(delta_Cm)
# eta_el = safe_read_dat(r'ETA_DH1_brett.dat')
# eta_el = normalize(eta_el)

hifi = hifi_F16()
raw_alpha1 = np.linspace(ALPHA1[0], ALPHA1[-1], 30)
raw_beta = np.linspace(BETA1[0], BETA1[-1], 30)
raw_el = np.linspace(DH1[0], DH1[-1], 30)
alpha = np.tile(raw_alpha1.reshape(-1, 1), raw_beta.shape[0] * raw_el.shape[0])
alpha = alpha.reshape(-1)
beta = np.tile(raw_beta.reshape(-1, 1), raw_el.shape[0])
beta = beta.reshape(-1)
beta = np.tile(beta, raw_alpha1.shape[0])
el = np.tile(raw_el, raw_alpha1.shape[0] * raw_beta.shape[0])
alpha = torch.tensor(alpha, device=torch.device(device), requires_grad=True)
beta = torch.tensor(beta, device=torch.device(device), requires_grad=True)
el = torch.tensor(el, device=torch.device(device), requires_grad=True)
# temp = hifi.hifi_C(alpha, beta, el)
Cx = hifi._Cx(alpha, beta, el)
Cz = hifi._Cz(alpha, beta, el)
Cm = hifi._Cm(alpha, beta, el)
Cy = hifi._Cy(alpha, beta)
Cn = hifi._Cn(alpha, beta, el)
Cl = hifi._Cl(alpha, beta, el)

# tmp = open("./mean_std.csv", 'w', newline='')
# csv_write = csv.writer(tmp)
# csv_write.writerow(["name", "alpha_mean", "alpha_std", "beta_mean", "beta_std", "el_mean", "el_std", "mean", "std"])
# csv_write.writerow(["Cx", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), _t2n(torch.mean(el)), _t2n(torch.std(el)), _t2n(torch.mean(Cx)), _t2n(torch.std(Cx))])
# csv_write.writerow(["Cz", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), _t2n(torch.mean(el)), _t2n(torch.std(el)), _t2n(torch.mean(Cz)), _t2n(torch.std(Cz))])
# csv_write.writerow(["Cm", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), _t2n(torch.mean(el)), _t2n(torch.std(el)), _t2n(torch.mean(Cm)), _t2n(torch.std(Cm))])
# csv_write.writerow(["Cn", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), _t2n(torch.mean(el)), _t2n(torch.std(el)), _t2n(torch.mean(Cn)), _t2n(torch.std(Cn))])
# csv_write.writerow(["Cl", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), _t2n(torch.mean(el)), _t2n(torch.std(el)), _t2n(torch.mean(Cl)), _t2n(torch.std(Cl))])
# tmp.close()

alpha = normalize(alpha)
beta = normalize(beta)
el = normalize(el)
Cx = normalize(Cx)
Cz = normalize(Cz)
Cm = normalize(Cm)
Cn = normalize(Cn)
Cl = normalize(Cl)

train_X = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
train_X = torch.hstack((train_X, el.reshape(-1, 1)))
train_Y = Cx.reshape(-1, 1)
train(train_X=train_X, train_Y=train_Y, file_name="Cx")
train_Y = Cz.reshape(-1, 1)
train(train_X=train_X, train_Y=train_Y, file_name="Cz")
train_Y = Cm.reshape(-1, 1)
train(train_X=train_X, train_Y=train_Y, file_name="Cm")
train_Y = Cn.reshape(-1, 1)
train(train_X=train_X, train_Y=train_Y, file_name="Cn")
train_Y = Cl.reshape(-1, 1)
train(train_X=train_X, train_Y=train_Y, file_name="Cl")

# alpha = np.linspace(ALPHA1[0], ALPHA1[-1], 4000)
# alpha = torch.tensor(alpha, device=torch.device(device), requires_grad=True)
# temp = hifi.hifi_damping(alpha)
# Cxq = temp[0]
# Cyr = temp[1]
# Cyp = temp[2]
# Czq = temp[3]
# Clr = temp[4]
# Clp = temp[5]
# Cmq = temp[6]
# Cnr = temp[7]
# Cnp = temp[8]

# tmp = open("./mean_std.csv", 'a', newline='')
# csv_write = csv.writer(tmp)
# csv_write.writerow(["Cxq", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Cxq)), _t2n(torch.std(Cxq))])
# csv_write.writerow(["Cyr", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Cyr)), _t2n(torch.std(Cyr))])
# csv_write.writerow(["Cyp", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Cyp)), _t2n(torch.std(Cyp))])
# csv_write.writerow(["Czq", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Czq)), _t2n(torch.std(Czq))])
# csv_write.writerow(["Clr", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Clr)), _t2n(torch.std(Clr))])
# csv_write.writerow(["Clp", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Clp)), _t2n(torch.std(Clp))])
# csv_write.writerow(["Cmq", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Cmq)), _t2n(torch.std(Cmq))])
# csv_write.writerow(["Cnr", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Cnr)), _t2n(torch.std(Cnr))])
# csv_write.writerow(["Cnp", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(Cnp)), _t2n(torch.std(Cnp))])
# tmp.close()

# alpha = normalize(alpha)
# Cxq = normalize(Cxq)
# Cyr = normalize(Cyr)
# Cyp = normalize(Cyp)
# Czq = normalize(Czq)
# Clr = normalize(Clr)
# Clp = normalize(Clp)
# Cmq = normalize(Cmq)
# Cnr = normalize(Cnr)
# Cnp = normalize(Cnp)

# train_X = alpha.reshape(-1, 1)
# train_Y = Cxq.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Cxq")
# train_Y = Cyr.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Cyr")
# train_Y = Cyp.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Cyp")
# train_Y = Czq.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Czq")
# train_Y = Clr.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Clr")
# train_Y = Clp.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Clp")
# train_Y = Cmq.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Cmq")
# train_Y = Cnr.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Cnr")
# train_Y = Cnp.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Cnp")

# raw_alpha2 = np.linspace(ALPHA2[0], ALPHA2[-1], 150)
# raw_beta = np.linspace(BETA1[0], BETA1[-1], 150)
# alpha = np.tile(raw_alpha2.reshape(-1, 1), raw_beta.shape[0])
# alpha = alpha.reshape(-1)
# beta = np.tile(raw_beta, raw_alpha2.shape[0])
# alpha = torch.tensor(alpha, device=torch.device(device), requires_grad=True)
# beta = torch.tensor(beta, device=torch.device(device), requires_grad=True)
# temp = hifi.hifi_C_lef(alpha, beta)
# delta_Cx_lef = temp[0]
# delta_Cz_lef = temp[1]
# delta_Cm_lef = temp[2]
# delta_Cy_lef = temp[3]
# delta_Cn_lef = temp[4]
# delta_Cl_lef = temp[5]

# tmp = open("./mean_std.csv", 'a', newline='')
# csv_write = csv.writer(tmp)
# csv_write.writerow(["delta_Cx_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cx_lef)), _t2n(torch.std(delta_Cx_lef))])
# csv_write.writerow(["delta_Cz_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cz_lef)), _t2n(torch.std(delta_Cz_lef))])
# csv_write.writerow(["delta_Cm_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cm_lef)), _t2n(torch.std(delta_Cm_lef))])
# csv_write.writerow(["delta_Cy_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cy_lef)), _t2n(torch.std(delta_Cy_lef))])
# csv_write.writerow(["delta_Cn_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cn_lef)), _t2n(torch.std(delta_Cn_lef))])
# csv_write.writerow(["delta_Cl_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cl_lef)), _t2n(torch.std(delta_Cl_lef))])
# tmp.close()

# alpha = normalize(alpha)
# beta = normalize(beta)
# delta_Cx_lef = normalize(delta_Cx_lef)
# delta_Cz_lef = normalize(delta_Cz_lef)
# delta_Cm_lef = normalize(delta_Cm_lef)
# delta_Cy_lef = normalize(delta_Cy_lef)
# delta_Cn_lef = normalize(delta_Cn_lef)
# delta_Cl_lef = normalize(delta_Cl_lef)

# train_X = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
# train_Y = delta_Cx_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cx_lef")
# train_Y = delta_Cz_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cz_lef")
# train_Y = delta_Cm_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cm_lef")
# train_Y = delta_Cy_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cy_lef")
# train_Y = delta_Cn_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cn_lef")
# train_Y = delta_Cl_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cl_lef")

# alpha = np.linspace(ALPHA2[0], ALPHA2[-1], 4000)
# alpha = torch.tensor(alpha, device=torch.device(device), requires_grad=True)
# temp = hifi.hifi_damping_lef(alpha)
# delta_Cxq_lef = temp[0]
# delta_Cyr_lef = temp[1]
# delta_Cyp_lef = temp[2]
# delta_Czq_lef = temp[3]
# delta_Clr_lef = temp[4]
# delta_Clp_lef = temp[5]
# delta_Cmq_lef = temp[6]
# delta_Cnr_lef = temp[7]
# delta_Cnp_lef = temp[8]

# tmp = open("./mean_std.csv", 'a', newline='')
# csv_write = csv.writer(tmp)
# csv_write.writerow(["delta_Cxq_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cxq_lef)), _t2n(torch.std(delta_Cxq_lef))])
# csv_write.writerow(["delta_Cyr_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cyr_lef)), _t2n(torch.std(delta_Cyr_lef))])
# csv_write.writerow(["delta_Cyp_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cyp_lef)), _t2n(torch.std(delta_Cyp_lef))])
# csv_write.writerow(["delta_Czq_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Czq_lef)), _t2n(torch.std(delta_Czq_lef))])
# csv_write.writerow(["delta_Clr_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Clr_lef)), _t2n(torch.std(delta_Clr_lef))])
# csv_write.writerow(["delta_Clp_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Clp_lef)), _t2n(torch.std(delta_Clp_lef))])
# csv_write.writerow(["delta_Cmq_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cmq_lef)), _t2n(torch.std(delta_Cmq_lef))])
# csv_write.writerow(["delta_Cnr_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cnr_lef)), _t2n(torch.std(delta_Cnr_lef))])
# csv_write.writerow(["delta_Cnp_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cnp_lef)), _t2n(torch.std(delta_Cnp_lef))])
# tmp.close()

# alpha = normalize(alpha)
# delta_Cxq_lef = normalize(delta_Cxq_lef)
# delta_Cyr_lef = normalize(delta_Cyr_lef)
# delta_Cyp_lef = normalize(delta_Cyp_lef)
# delta_Czq_lef = normalize(delta_Czq_lef)
# delta_Clr_lef = normalize(delta_Clr_lef)
# delta_Clp_lef = normalize(delta_Clp_lef)
# delta_Cmq_lef = normalize(delta_Cmq_lef)
# delta_Cnr_lef = normalize(delta_Cnr_lef)
# delta_Cnp_lef = normalize(delta_Cnp_lef)

# train_X = alpha.reshape(-1, 1)
# train_Y = delta_Cxq_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cxq_lef")
# train_Y = delta_Cyr_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cyr_lef")
# train_Y = delta_Cyp_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cyp_lef")
# train_Y = delta_Czq_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Czq_lef")
# train_Y = delta_Clr_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Clr_lef")
# train_Y = delta_Clp_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Clp_lef")
# train_Y = delta_Cmq_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cmq_lef")
# train_Y = delta_Cnr_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cnr_lef")
# train_Y = delta_Cnp_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cnp_lef")

# raw_alpha1 = np.linspace(ALPHA1[0], ALPHA1[-1], 150)
# raw_beta = np.linspace(BETA1[0], BETA1[-1], 150)
# alpha = np.tile(raw_alpha1.reshape(-1, 1), raw_beta.shape[0])
# alpha = alpha.reshape(-1)
# beta = np.tile(raw_beta, raw_alpha1.shape[0])
# alpha = torch.tensor(alpha, device=torch.device(device), requires_grad=True)
# beta = torch.tensor(beta, device=torch.device(device), requires_grad=True)
# temp = hifi.hifi_rudder(alpha, beta)
# delta_Cy_r30 = temp[0]
# delta_Cn_r30 = temp[1]
# delta_Cl_r30 = temp[2]

# tmp = open("./mean_std.csv", 'a', newline='')
# csv_write = csv.writer(tmp)
# csv_write.writerow(["delta_Cy_r30", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cy_r30)), _t2n(torch.std(delta_Cy_r30))])
# csv_write.writerow(["delta_Cn_r30", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cn_r30)), _t2n(torch.std(delta_Cn_r30))])
# csv_write.writerow(["delta_Cl_r30", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cl_r30)), _t2n(torch.std(delta_Cl_r30))])
# tmp.close()

# alpha = normalize(alpha)
# beta = normalize(beta)
# delta_Cy_r30 = normalize(delta_Cy_r30)
# delta_Cn_r30 = normalize(delta_Cn_r30)
# delta_Cl_r30 = normalize(delta_Cl_r30)

# train_X = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
# train_Y = delta_Cy_r30.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cy_r30")
# train_Y = delta_Cn_r30.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cn_r30")
# train_Y = delta_Cl_r30.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cl_r30")

# raw_alpha2 = np.linspace(ALPHA2[0], ALPHA2[-1], 150)
# raw_beta = np.linspace(BETA1[0], BETA1[-1], 150)
# alpha = np.tile(raw_alpha2.reshape(-1, 1), raw_beta.shape[0])
# alpha = alpha.reshape(-1)
# beta = np.tile(raw_beta, raw_alpha2.shape[0])
# alpha = torch.tensor(alpha, device=torch.device(device), requires_grad=True)
# beta = torch.tensor(beta, device=torch.device(device), requires_grad=True)
# zero = torch.zeros_like(alpha)
# temp = hifi_F16.hifi_ailerons(alpha, beta)
# Cy = hifi._Cy(alpha, beta)
# delta_Cy_a20 = hifi._Cy_a20(alpha, beta) - hifi._Cy(alpha, beta)
# delta_Cy_a20_lef = hifi._Cy_a20_lef(alpha, beta) - hifi._Cy_lef(alpha, beta) - (hifi._Cy_a20(alpha, beta) - hifi._Cy(alpha, beta))
# delta_Cn_a20 = hifi._Cn_a20(alpha, beta) - hifi._Cn(alpha, beta, zero)
# delta_Cn_a20_lef = hifi._Cn_a20_lef(alpha, beta) - hifi._Cn_lef(alpha, beta) - (hifi._Cn_a20(alpha, beta) - hifi._Cn(alpha, beta, zero))
# delta_Cl_a20 = hifi._Cl_a20(alpha, beta) - hifi._Cl(alpha, beta, zero)
# delta_Cl_a20_lef = hifi._Cl_a20_lef(alpha, beta) - hifi._Cl_lef(alpha, beta) - (hifi._Cl_a20(alpha, beta) - hifi._Cl(alpha, beta, zero))

# tmp = open("./mean_std.csv", 'a', newline='')
# csv_write = csv.writer(tmp)
# csv_write.writerow(["Cy", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(Cy)), _t2n(torch.std(Cy))])
# csv_write.writerow(["delta_Cy_a20", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cy_a20)), _t2n(torch.std(delta_Cy_a20))])
# csv_write.writerow(["delta_Cn_a20", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cn_a20)), _t2n(torch.std(delta_Cn_a20))])
# csv_write.writerow(["delta_Cl_a20", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cl_a20)), _t2n(torch.std(delta_Cl_a20))])
# csv_write.writerow(["delta_Cy_a20_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cy_a20_lef)), _t2n(torch.std(delta_Cy_a20_lef))])
# csv_write.writerow(["delta_Cn_a20_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cn_a20_lef)), _t2n(torch.std(delta_Cn_a20_lef))])
# csv_write.writerow(["delta_Cl_a20_lef", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), _t2n(torch.mean(beta)), _t2n(torch.std(beta)), 0, 0, _t2n(torch.mean(delta_Cl_a20_lef)), _t2n(torch.std(delta_Cl_a20_lef))])
# tmp.close()

# alpha = normalize(alpha)
# beta = normalize(beta)
# Cy = normalize(Cy)
# delta_Cy_a20 = normalize(delta_Cy_a20)
# delta_Cn_a20 = normalize(delta_Cn_a20)
# delta_Cl_a20 = normalize(delta_Cl_a20)
# delta_Cy_a20_lef = normalize(delta_Cy_a20_lef)
# delta_Cn_a20_lef = normalize(delta_Cn_a20_lef)
# delta_Cl_a20_lef = normalize(delta_Cl_a20_lef)

# train_X = torch.hstack((alpha.reshape(-1, 1), beta.reshape(-1, 1)))
# train_Y = Cy.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="Cy")
# train_Y = delta_Cy_a20.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cy_a20")
# train_Y = delta_Cn_a20.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cn_a20")
# train_Y = delta_Cl_a20.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cl_a20")
# train_Y = delta_Cy_a20_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cy_a20_lef")
# train_Y = delta_Cn_a20_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cn_a20_lef")
# train_Y = delta_Cl_a20_lef.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cl_a20_lef")

# alpha = np.linspace(ALPHA1[0], ALPHA1[-1], 4000)
# alpha = torch.tensor(alpha, device=torch.device(device), requires_grad=True)
# el = np.linspace(DH1[0], DH1[-1], 4000)
# el = torch.tensor(el, device=torch.device(device), requires_grad=True)
# temp = hifi.hifi_other_coeffs(alpha, el)
# delta_Cnbeta = temp[0]
# delta_Clbeta = temp[1]
# delta_Cm = temp[2]
# eta_el = temp[3]
# # delta_Cm_ds = temp[4]

# tmp = open("./mean_std.csv", 'a', newline='')
# csv_write = csv.writer(tmp)
# csv_write.writerow(["delta_Cnbeta", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cnbeta)), _t2n(torch.std(delta_Cnbeta))])
# csv_write.writerow(["delta_Clbeta", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Clbeta)), _t2n(torch.std(delta_Clbeta))])
# csv_write.writerow(["delta_Cm", _t2n(torch.mean(alpha)), _t2n(torch.std(alpha)), 0, 0, 0, 0, _t2n(torch.mean(delta_Cm)), _t2n(torch.std(delta_Cm))])
# csv_write.writerow(["eta_el", 0, 0, 0, 0, _t2n(torch.mean(el)), _t2n(torch.std(el)), _t2n(torch.mean(eta_el)), _t2n(torch.std(eta_el))])
# tmp.close()

# alpha = normalize(alpha)
# el = normalize(el)
# delta_Cnbeta = normalize(delta_Cnbeta)
# delta_Clbeta = normalize(delta_Clbeta)
# delta_Cm = normalize(delta_Cm)
# eta_el = normalize(eta_el)

# train_X = alpha.reshape(-1, 1)
# train_Y = delta_Cnbeta.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cnbeta")
# train_Y = delta_Clbeta.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Clbeta")
# train_Y = delta_Cm.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="delta_Cm")
# train_X = el.reshape(-1, 1)
# train_Y = eta_el.reshape(-1, 1)
# train(train_X=train_X, train_Y=train_Y, file_name="eta_el")

# train_X, train_Y = generate_data1d([ALPHA2], delta_CXq_lef)
# train_X, train_Y = interpn(train_X, train_Y)
# plt.plot(train_X[:, 0], train_Y[:, 0])
# plt.show()
# X = ALPHA2.reshape(-1, 1).repeat(BETA1.shape[0], 1)
# Y = BETA1.reshape(1, -1).repeat(ALPHA2.shape[0], 0)
# Z = np.zeros((ALPHA2.shape[0], BETA1.shape[0]))
# for i in range(ALPHA2.shape[0]):
#     for j in range(BETA1.shape[0]):
#         Z[i, j] = Cl_lef[i + j * ALPHA2.shape[0]]
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z)
# plt.show()
# train(train_X=train_X, train_Y=train_Y, file_name="eta_el")

data_matlab = np.array(pd.read_csv('coefs.csv', header=None))
data_C = np.array(pd.read_csv('coefs_C.csv', header=None))
alpha = torch.tensor(data_matlab[0, :400], device=torch.device(device))
beta = torch.tensor(data_matlab[1, :400], device=torch.device(device))
dele = torch.tensor(data_matlab[2, :400], device=torch.device(device))
temp = hifi.hifi_C_lef(alpha, beta)
delta_Cx_lef = temp[0]
delta_Cx_lef_matlab = data_matlab[18, :400]
# delta_Cx_lef_C = data_C[18, :]
delta_Cz_lef = temp[1]
delta_Cz_lef_matlab = data_matlab[19, :400]
# delta_Cz_lef_C = data_C[19, :]
delta_Cm_lef = temp[2]
delta_Cm_lef_matlab = data_matlab[20, :400]
# delta_Cm_lef_C = data_C[20, :]
delta_Cy_lef = temp[3]
delta_Cy_lef_matlab = data_matlab[21, :400]
# delta_Cy_lef_C = data_C[21, :]
delta_Cn_lef = temp[4]
delta_Cn_lef_matlab = data_matlab[22, :400]
# delta_Cn_lef_C = data_C[22, :]
delta_Cl_lef = temp[5]
delta_Cl_lef_matlab = data_matlab[23, :400]
# delta_Cl_lef_C = data_C[23, :]
print('delta_Cx_lef:', r2_score(delta_Cx_lef_matlab, _t2n(delta_Cx_lef)))
print('delta_Cz_lef:', r2_score(delta_Cz_lef_matlab, _t2n(delta_Cz_lef)))
print('delta_Cm_lef:', r2_score(delta_Cm_lef_matlab, _t2n(delta_Cm_lef)))
print('delta_Cy_lef:', r2_score(delta_Cy_lef_matlab, _t2n(delta_Cy_lef)))
print('delta_Cn_lef:', r2_score(delta_Cn_lef_matlab, _t2n(delta_Cn_lef)))
print('delta_Cl_lef:', r2_score(delta_Cl_lef_matlab, _t2n(delta_Cl_lef)))

temp = hifi.hifi_damping_lef(alpha)
delta_Cxq_lef = temp[0]
delta_Cxq_lef_matlab = data_matlab[24, :400]
# delta_Cxq_lef_C = data_C[24, :]
delta_Cyr_lef = temp[1]
delta_Cyr_lef_matlab = data_matlab[25, :400]
# delta_Cyr_lef_C = data_C[25, :]
delta_Cyp_lef = temp[2]
delta_Cyp_lef_matlab = data_matlab[26, :400]
# delta_Cyp_lef_C = data_C[26, :]
delta_Czq_lef = temp[3]
delta_Czq_lef_matlab = data_matlab[27, :400]
# delta_Czq_lef_C = data_C[27, :]
delta_Clr_lef = temp[4]
delta_Clr_lef_matlab = data_matlab[28, :400]
# delta_Clr_lef_C = data_C[28, :]
delta_Clp_lef = temp[5]
delta_Clp_lef_matlab = data_matlab[29, :400]
# delta_Clp_lef_C = data_C[29, :]
delta_Cmq_lef = temp[6]
delta_Cmq_lef_matlab = data_matlab[30, :400]
# delta_Cmq_lef_C = data_C[30, :]
delta_Cnr_lef = temp[7]
delta_Cnr_lef_matlab = data_matlab[31, :400]
# delta_Cnr_lef_C = data_C[31, :]
delta_Cnp_lef = temp[8]
delta_Cnp_lef_matlab = data_matlab[32, :400]
# delta_Cnp_lef_C = data_C[32, :]
print('delta_Cxq_lef:', r2_score(delta_Cxq_lef_matlab, _t2n(delta_Cxq_lef)))
print('delta_Cyr_lef:', r2_score(delta_Cyr_lef_matlab, _t2n(delta_Cyr_lef)))
print('delta_Cyp_lef:', r2_score(delta_Cyp_lef_matlab, _t2n(delta_Cyp_lef)))
print('delta_Czq_lef:', r2_score(delta_Czq_lef_matlab, _t2n(delta_Czq_lef)))
print('delta_Clr_lef:', r2_score(delta_Clr_lef_matlab, _t2n(delta_Clr_lef)))
print('delta_Clp_lef:', r2_score(delta_Clp_lef_matlab, _t2n(delta_Clp_lef)))
print('delta_Cmq_lef:', r2_score(delta_Cmq_lef_matlab, _t2n(delta_Cmq_lef)))
print('delta_Cnr_lef:', r2_score(delta_Cnr_lef_matlab, _t2n(delta_Cnr_lef)))
print('delta_Cnp_lef:', r2_score(delta_Cnp_lef_matlab, _t2n(delta_Cnp_lef)))

temp = hifi.hifi_ailerons(alpha, beta)
delta_Cy_a20 = temp[0]
delta_Cy_a20_matlab = data_matlab[36, :400]
# delta_Cy_a20_C = data_C[36, :]
delta_Cy_a20_lef = temp[1]
delta_Cy_a20_lef_matlab = data_matlab[39, :400]
# delta_Cy_a20_lef_C = data_C[39, :]
delta_Cn_a20 = temp[2]
delta_Cn_a20_matlab = data_matlab[37, :400]
# delta_Cn_a20_C = data_C[37, :]
delta_Cn_a20_lef = temp[3]
delta_Cn_a20_lef_matlab = data_matlab[40, :400]
# delta_Cn_a20_lef_C = data_C[40, :]
delta_Cl_a20 = temp[4]
delta_Cl_a20_matlab = data_matlab[38, :400]
# delta_Cl_a20_C = data_C[38, :]
delta_Cl_a20_lef = temp[5]
delta_Cl_a20_lef_matlab = data_matlab[41, :400]
# delta_Cl_a20_lef_C = data_C[41, :]
print('delta_Cy_a20:', r2_score(delta_Cy_a20_matlab, _t2n(delta_Cy_a20)))
print('delta_Cy_a20_lef:', r2_score(delta_Cy_a20_lef_matlab, _t2n(delta_Cy_a20_lef)))
print('delta_Cn_a20:', r2_score(delta_Cn_a20_matlab, _t2n(delta_Cn_a20)))
print('delta_Cn_a20_lef:', r2_score(delta_Cn_a20_lef_matlab, _t2n(delta_Cn_a20_lef)))
print('delta_Cl_a20:', r2_score(delta_Cl_a20_matlab, _t2n(delta_Cl_a20)))
print('delta_Cl_a20_lef:', r2_score(delta_Cl_a20_lef_matlab, _t2n(delta_Cl_a20_lef)))