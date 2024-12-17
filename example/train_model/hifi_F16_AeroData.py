from .mexndinterp import interpn
import torch
import time

HIFI_GLOBAL_TXT_CONTENT = {}
device = "cuda:0"


class hifi_F16():
    def __init__(self) -> None:
        self.ALPHA1 = self.safe_read_dat(r'ALPHA1.dat')
        self.ALPHA2 = self.safe_read_dat(r'ALPHA2.dat')
        self.BETA1 = self.safe_read_dat(r'BETA1.dat')
        self.DH1 = self.safe_read_dat(r'DH1.dat')
        self.DH2 = self.safe_read_dat(r'DH2.dat')
        self.Cx = self.safe_read_dat(r'CX0120_ALPHA1_BETA1_DH1_201.dat')
        self.Cz = self.safe_read_dat(r'CZ0120_ALPHA1_BETA1_DH1_301.dat')
        self.Cm = self.safe_read_dat(r'CM0120_ALPHA1_BETA1_DH1_101.dat')
        self.Cy = self.safe_read_dat(r'CY0320_ALPHA1_BETA1_401.dat')
        self.Cn = self.safe_read_dat(r'CN0120_ALPHA1_BETA1_DH2_501.dat')
        self.Cl = self.safe_read_dat(r'CL0120_ALPHA1_BETA1_DH2_601.dat')
        self.Cx_lef = self.safe_read_dat(r'CX0820_ALPHA2_BETA1_202.dat')
        self.Cz_lef = self.safe_read_dat(r'CZ0820_ALPHA2_BETA1_302.dat')
        self.Cm_lef = self.safe_read_dat(r'CM0820_ALPHA2_BETA1_102.dat')
        self.Cy_lef = self.safe_read_dat(r'CY0820_ALPHA2_BETA1_402.dat')
        self.Cn_lef = self.safe_read_dat(r'CN0820_ALPHA2_BETA1_502.dat')
        self.Cl_lef = self.safe_read_dat(r'CL0820_ALPHA2_BETA1_602.dat')
        self.CXq = self.safe_read_dat(r'CX1120_ALPHA1_204.dat')
        self.CZq = self.safe_read_dat(r'CZ1120_ALPHA1_304.dat')
        self.CMq = self.safe_read_dat(r'CM1120_ALPHA1_104.dat')
        self.CYp = self.safe_read_dat(r'CY1220_ALPHA1_408.dat')
        self.CYr = self.safe_read_dat(r'CY1320_ALPHA1_406.dat')
        self.CNr = self.safe_read_dat(r'CN1320_ALPHA1_506.dat')
        self.CNp = self.safe_read_dat(r'CN1220_ALPHA1_508.dat')
        self.CLp = self.safe_read_dat(r'CL1220_ALPHA1_608.dat')
        self.CLr = self.safe_read_dat(r'CL1320_ALPHA1_606.dat')
        self.delta_CXq_lef = self.safe_read_dat(r'CX1420_ALPHA2_205.dat')
        self.delta_CYr_lef = self.safe_read_dat(r'CY1620_ALPHA2_407.dat')
        self.delta_CYp_lef = self.safe_read_dat(r'CY1520_ALPHA2_409.dat')
        self.delta_CZq_lef = self.safe_read_dat(r'CZ1420_ALPHA2_305.dat')
        self.delta_CLr_lef = self.safe_read_dat(r'CL1620_ALPHA2_607.dat')
        self.delta_CLp_lef = self.safe_read_dat(r'CL1520_ALPHA2_609.dat')
        self.delta_CMq_lef = self.safe_read_dat(r'CM1420_ALPHA2_105.dat')
        self.delta_CNr_lef = self.safe_read_dat(r'CN1620_ALPHA2_507.dat')
        self.delta_CNp_lef = self.safe_read_dat(r'CN1520_ALPHA2_509.dat')
        self.Cy_r30 = self.safe_read_dat(r'CY0720_ALPHA1_BETA1_405.dat')
        self.Cn_r30 = self.safe_read_dat(r'CN0720_ALPHA1_BETA1_503.dat')
        self.Cl_r30 = self.safe_read_dat(r'CL0720_ALPHA1_BETA1_603.dat')
        self.Cy_a20 = self.safe_read_dat(r'CY0620_ALPHA1_BETA1_403.dat')
        self.Cy_a20_lef = self.safe_read_dat(r'CY0920_ALPHA2_BETA1_404.dat')
        self.Cn_a20 = self.safe_read_dat(r'CN0620_ALPHA1_BETA1_504.dat')
        self.Cn_a20_lef = self.safe_read_dat(r'CN0920_ALPHA2_BETA1_505.dat')
        self.Cl_a20 = self.safe_read_dat(r'CL0620_ALPHA1_BETA1_604.dat')
        self.Cl_a20_lef = self.safe_read_dat(r'CL0920_ALPHA2_BETA1_605.dat')
        self.delta_CNbeta = self.safe_read_dat(r'CN9999_ALPHA1_brett.dat')
        self.delta_CLbeta = self.safe_read_dat(r'CL9999_ALPHA1_brett.dat')
        self.delta_Cm = self.safe_read_dat(r'CM9999_ALPHA1_brett.dat')
        self.eta_el = self.safe_read_dat(r'ETA_DH1_brett.dat')

    def safe_read_dat(self, dat_name):
        try:
            if dat_name in HIFI_GLOBAL_TXT_CONTENT:
                return HIFI_GLOBAL_TXT_CONTENT.get(dat_name)

            path = r'../data/' + dat_name
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                content = content.strip()
                data_str = [value for value in content.split(' ') if value]
                data = list(map(float, data_str))
                data = torch.tensor(data, device=torch.device(device))
                HIFI_GLOBAL_TXT_CONTENT[dat_name] = data
                return data
        except OSError:
            print("Cannot find file {} in current directory".format(path))
            return []

    def _Cx(self, alpha, beta, dele):
        gain = [alpha, beta, dele]
        ndinfo = [20, 19, 5]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        X.append(self.DH1)
        result = interpn(X, self.Cx, gain, ndinfo)
        return result

    def _Cz(self, alpha, beta, dele):
        gain = [alpha, beta, dele]
        ndinfo = [20, 19, 5]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        X.append(self.DH1)
        return interpn(X, self.Cz, gain, ndinfo)

    def _Cm(self, alpha, beta, dele):
        gain = [alpha, beta, dele]
        ndinfo = [20, 19, 5]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        X.append(self.DH1)
        return interpn(X, self.Cm, gain, ndinfo)

    def _Cy(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [20, 19]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        return interpn(X, self.Cy, gain, ndinfo)

    def _Cn(self, alpha, beta, dele):
        gain = [alpha, beta, dele]
        ndinfo = [20, 19, 3]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        X.append(self.DH2)
        return interpn(X, self.Cn, gain, ndinfo)

    def _Cl(self, alpha, beta, dele):
        gain = [alpha, beta, dele]
        ndinfo = [20, 19, 3]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        X.append(self.DH2)
        return interpn(X, self.Cl, gain, ndinfo)

    def _Cx_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cx_lef, gain, ndinfo)

    def _Cz_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cz_lef, gain, ndinfo)

    def _Cm_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cm_lef, gain, ndinfo)

    def _Cy_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cy_lef, gain, ndinfo)

    def _Cn_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cn_lef, gain, ndinfo)

    def _Cl_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cl_lef, gain, ndinfo)

    def _CXq(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CXq, gain, ndinfo)

    def _CZq(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CZq, gain, ndinfo)

    def _CMq(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CMq, gain, ndinfo)

    def _CYp(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CYp, gain, ndinfo)

    def _CYr(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CYr, gain, ndinfo)

    def _CNr(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CNr, gain, ndinfo)

    def _CNp(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CNp, gain, ndinfo)

    def _CLp(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CLp, gain, ndinfo)

    def _CLr(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.CLr, gain, ndinfo)

    def _delta_CXq_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CXq_lef, gain, ndinfo)

    def _delta_CYr_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CYr_lef, gain, ndinfo)

    def _delta_CYp_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CYp_lef, gain, ndinfo)

    def _delta_CZq_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CZq_lef, gain, ndinfo)

    def _delta_CLr_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CLr_lef, gain, ndinfo)

    def _delta_CLp_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CLp_lef, gain, ndinfo)

    def _delta_CMq_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CMq_lef, gain, ndinfo)

    def _delta_CNr_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CNr_lef, gain, ndinfo)

    def _delta_CNp_lef(self, alpha):
        gain = [alpha]
        ndinfo = [14]
        X = []
        X.append(self.ALPHA2)
        return interpn(X, self.delta_CNp_lef, gain, ndinfo)

    def _Cy_r30(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [20, 19]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        return interpn(X, self.Cy_r30, gain, ndinfo)

    def _Cn_r30(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [20, 19]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        return interpn(X, self.Cn_r30, gain, ndinfo)

    def _Cl_r30(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [20, 19]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        return interpn(X, self.Cl_r30, gain, ndinfo)

    def _Cy_a20(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [20, 19]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        return interpn(X, self.Cy_a20, gain, ndinfo)

    def _Cy_a20_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cy_a20_lef, gain, ndinfo)

    def _Cn_a20(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [20, 19]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        return interpn(X, self.Cn_a20, gain, ndinfo)

    def _Cn_a20_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cn_a20_lef, gain, ndinfo)

    def _Cl_a20(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [20, 19]
        X = []
        X.append(self.ALPHA1)
        X.append(self.BETA1)
        return interpn(X, self.Cl_a20, gain, ndinfo)

    def _Cl_a20_lef(self, alpha, beta):
        gain = [alpha, beta]
        ndinfo = [14, 19]
        X = []
        X.append(self.ALPHA2)
        X.append(self.BETA1)
        return interpn(X, self.Cl_a20_lef, gain, ndinfo)

    def _delta_CNbeta(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.delta_CNbeta, gain, ndinfo)

    def _delta_CLbeta(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.delta_CLbeta, gain, ndinfo)

    def _delta_Cm(self, alpha):
        gain = [alpha]
        ndinfo = [20]
        X = []
        X.append(self.ALPHA1)
        return interpn(X, self.delta_Cm, gain, ndinfo)

    def _eta_el(self, el):
        gain = [el]
        ndinfo = [5]
        X = []
        X.append(self.DH1)
        return interpn(X, self.eta_el, gain, ndinfo)

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
