

from wannierberri.calculators.static import StaticCalculator
from wannierberri.formula import Formula_ln
from wannierberri.formula.covariant import DerQuantumMetric_ab_d
from wannierberri.symmetry.point_symmetry import transform_ident, transform_odd
from wannierberri.utility import cached_einsum, alpha_A, beta_A
import numpy as np
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom




class PositionalShiftFormula(Formula_ln):

    from wannierberri.factors import m_spin_prefactor

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
                self.M = data_k.get_M1(external_terms=self.external_terms)
            if self.spin_part:
                from wannierberri.factors import m_spin_prefactor
                self.S = data_k.covariant('SS') * m_spin_prefactor
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
                M += self.M[ik, out][:, inn]
            if self.spin_part:
                M += self.S.nl(ik, inn, out) 
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
        self.transformTR = transform_odd
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
        super().__init__(**parameters)
