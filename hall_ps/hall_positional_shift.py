

from functools import lru_cache

from wannierberri.calculators.static import StaticCalculator
from wannierberri.formula import Formula_ln
from wannierberri.formula.covariant import DerQuantumMetric_ab_d
from wannierberri.symmetry.point_symmetry import transform_ident, transform_odd
from wannierberri.utility import cached_einsum, alpha_A, beta_A
from wannierberri.data_K.data_K_R import Data_K_R

from wannierberri.factors import elementary_charge, hbar, m_spin_prefactor, angstrom

# Positional shift comes in units of Ang^3, multiplied by band gradient (eV/Ang) and f' (1/eV) and integrated over d3k (1/Ang^3), the final unit is Ang. 
# First, convert to m, then multiply by e/hbar (1/(T*m^2)), and then by e^2/hbar ( S = 1/Ohm), to get the final unit S/(T*m).
factor_hall_pos_shift = angstrom * elementary_charge**3 / hbar**2

import numpy as np



class Data_K_pos_shift(Data_K_R):

    @lru_cache
    def get_M1_AH(self, external_terms=True,
               M1_term=True,
               V_term=True,
               spin=False,
               key_OO='rotAA', degen_thresh=1e-3,
               AH_term=False):
        ''' Magnetic dipole moment '''
        M = self.get_M1(external_terms=external_terms, spin=spin, orb=M1_term,
                        key_OO=key_OO, degen_thresh=degen_thresh)
        if V_term:
            Vn = self.delE_K
            Vnm_plus = (Vn[:, :, None, :] + Vn[:, None, :, :])
            A = self.get_E1(external_terms=external_terms, degen_thresh=degen_thresh)
            M += 0.5 * (Vnm_plus[:, :, :, alpha_A] * A[:, :, :, beta_A] -
                    Vnm_plus[:, :, :, beta_A] * A[:, :, :, alpha_A])
        if AH_term:
            En = self.E_K
            O_H = self.get_O1(external_terms=external_terms, key_OO=key_OO, degen_thresh=degen_thresh)
            Eln_minus = (En[:, :, None] - En[:, None, :])
            M += 0.25 * Eln_minus[:, :, :, None] * O_H
        return M


class PositionalShiftFormula(Formula_ln):

    def __init__(self, data_k,
                 spin_part=False,
                 morb_part=False,
                 AH_term=True,
                 V_term=True,
                 M1_term=True,
                 metric_part=False,
                 **parameters):
        super().__init__(data_k, **parameters)
        
        assert any([spin_part, morb_part, metric_part]
                   ), "at least one part should be included in PositionalShiftFormula"
        self.M_part = morb_part or spin_part
        self.metric_part = metric_part
        if self.M_part:
            self.A = data_k.get_E1(external_terms=self.external_terms)
            self.M = data_k.get_M1_AH(external_terms=self.external_terms, 
                                      V_term=morb_part and V_term, 
                                      AH_term=morb_part and AH_term,
                                      M1_term=morb_part and M1_term,
                                      spin=spin_part, 
                                      key_OO='rotAA', degen_thresh=1e-3)
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
        if self.M_part:
            M = self.M[ik, inn][:, out]
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
