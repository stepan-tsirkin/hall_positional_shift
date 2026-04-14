

from wannierberri.calculators.static import StaticCalculator
from wannierberri.formula import Formula_ln
from wannierberri.formula.covariant import DerQuantumMetric_ab_d
from wannierberri.symmetry.point_symmetry import transform_ident, transform_odd
from wannierberri.utility import cached_einsum, alpha_A, beta_A


from wannierberri.factors import elementary_charge, hbar, m_spin_prefactor, angstrom

# Positional shift comes in units of Ang^3, multiplied by band gradient (eV/Ang) and f' (1/eV) and integrated over d3k (1/Ang^3), the final unit is Ang. 
# First, convert to m, then multiply by e/hbar (1/(T*m^2)), and then by e^2/hbar ( S = 1/Ohm), to get the final unit S/(T*m).
factor_hall_pos_shift = angstrom * elementary_charge**3 / hbar**2

import numpy as np




class PositionalShiftFormula(Formula_ln):

    def __init__(self, data_k,
                 spin_part=False,
                 morb_part=False,
                 metric_part=False,
                 **parameters):
        super().__init__(data_k, **parameters)
        
        assert any([spin_part, morb_part, metric_part]
                   ), "at least one part should be included in PositionalShiftFormula"
        self.spin_part = spin_part
        self.morb_part = morb_part
        self.metric_part = metric_part
        if self.morb_part or self.spin_part:
            self.A = data_k.get_E1(external_terms=self.external_terms)
            if self.morb_part:
                self.M = data_k.get_M1(external_terms=self.external_terms, V_term=True, AH_term=True)
            if self.spin_part:
                self.S = data_k.covariant('SS')
        if self.metric_part:
            self.DerMetric = DerQuantumMetric_ab_d(
                data_k, external_terms=self.external_terms)
        self.dEig_inv = data_k.dEig_inv

        self.ndim = 2
        # self.transformTR = transform_odd
        # self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        res = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        rng = range(len(inn))
        if self.morb_part or self.spin_part:
            M = 0
            if self.morb_part:
                M += self.M[ik, inn][:, out]
            if self.spin_part:
                M += self.S.nl(ik, inn, out)  * m_spin_prefactor
            res[rng, rng, :, :] += 2*cached_einsum("nma,mnb,nm->nab",
                                    M, self.A[ik, out][:, inn], self.dEig_inv[ik, inn][:, out] ).real
        if self.metric_part:
            dGdb_c = self.DerMetric.nn(ik, inn, out)
            res[:, :, :, :] += 0.5 * (dGdb_c[:, :, alpha_A, :, beta_A] 
                                      - dGdb_c[:, :, beta_A, :, alpha_A]).transpose(1,2,0,3)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError(
            "ln not implemented for PositionalShiftFormula yet")


class HallPositionalShiftFormula(Formula_ln):



    def __init__(self, data_k,
                 spin_part=False,
                 morb_part=False,
                 metric_part=False,
                 **parameters):
        super().__init__(data_k, **parameters)
        self.V = data_k.covariant('Ham', commader=1)
        self.F = PositionalShiftFormula(data_k,
                                        spin_part=spin_part,
                                        morb_part=morb_part,
                                        metric_part=metric_part,
                                        **parameters)
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        v = self.V.nn(ik, inn, out)
        F = self.F.nn(ik, inn, out)
        vF = cached_einsum("lma,mnbc->lnabc", v, F)
        
        return (vF[:,:,alpha_A, : , beta_A] - vF[:,:,beta_A, : , alpha_A]
                ).transpose((1,2,0,3))

    def ln(self, ik, inn, out):
        raise NotImplementedError(
            "ln not implemented for HallPositionalShiftFormula yet")


class HallPositionalShift(StaticCalculator):

    def __init__(self,  **parameters):
        self.fder = 1
        self.Formula = HallPositionalShiftFormula
        super().__init__(constant_factor=factor_hall_pos_shift, **parameters)
