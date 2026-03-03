

from wannierberri.calculators.static import StaticCalculator
from wannierberri.formula import Formula_ln
from wannierberri.formula.covariant import DerDcov
from wannierberri.symmetry.point_symmetry import transform_ident, transform_odd
from wannierberri.utility import cached_einsum, alpha_A, beta_A
import numpy as np
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

####################################################
#    Spatially-dispersive conductivity tensor      #
####################################################

# To keep e^2/hbar as the global factor of SDCT

electron_g_factor = physical_constants['electron g factor'][0]
# chack if it is correct
spin_prefactor = electron_g_factor * hbar**2 / \
    (elementary_charge * electron_mass * angstrom**2)


class DerMetricTerm(Formula_ln):
    """
    formula for epsilon_bcd D_c g_da
    """

    def __init__(self, data_k, **parameters):
        super().__init__(data_k, **parameters)
        self.D = data_k.Dcov
        self.dD = DerDcov(data_k)
        if self.external_terms:
            self.A = data_k.covariant('AA')
            self.dA = data_k.covariant('AA', gender=1)

    def nn(self, ik, inn, out):
        Aln = 1j * self.D.ln(ik, inn, out)
        dAln = 1j * self.dD.ln(ik, inn, out)
        Anl = 1j * self.D.nl(ik, inn, out)
        dAnl = 1j * self.dD.nl(ik, inn, out)
        if self.external_terms:
            Aln += self.A.ln(ik, inn, out)
            dAln += self.dA.ln(ik, inn, out)
            Anl += self.A.nl(ik, inn, out)
            dAnl += self.dA.nl(ik, inn, out)
        term1 = cached_einsum("mla,lndc->mncda", Anl, dAln)
        term2 = cached_einsum("mlac,lnd->mncda", dAnl, Aln)
        res = term1 + term2
        res = 0.5*(res[:, :, alpha_A, beta_A] - res[:, :, beta_A, alpha_A])
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError("ln not implemented for DerMetricTerm yet")


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
        if self.external_terms:
            if self.morb_part:
                self.M = data_k.SDCT.M1
            self.A = data_k.A_H
        else:
            if self.morb_part:
                self.M = data_k.SDCT.M1_internal
            self.A = data_k.A_H_internal
        if self.spin_part:
            self.S = data_k.covariant('SS')
        if self.metric_part:
            self.DerMetric = DerMetricTerm(
                data_k, external_terms=self.external_terms)
        self.dEig_inv = data_k.dEig_inv

    def nn(self, ik, inn, out):
        res = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        rng = range(len(inn))
        if self.morb_part or self.spin_part:
            M = 0
            if self.morb_part:
                M += self.M[ik, out][:, inn]
            if self.spin_part:
                M += self.S.ln(ik, inn, out) * spin_prefactor
            res[rng, rng, :, :] += 2*cached_einsum("nma,mnb,nm->nab",
                                                   self.A[ik, inn][:, out],
                                                   M,
                                                   self.dEig_inv[ik,
                                                                 inn][:, out],
                                                   ).real
        if self.metric_part:
            res += self.DerMetric.nn(ik, inn, out)
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
        self.ndim = 1
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        v = self.V.nn(ik, inn, out)
        F = self.F.nn(ik, inn, out)
        vF = cached_einsum("mna,nmbc->abc", v, F)
        return vF - vF.swapaxes(0, 1)

    def ln(self, ik, inn, out):
        raise NotImplementedError(
            "ln not implemented for HallPositionalShiftFormula yet")


class HallPositionalShift(StaticCalculator):

    def __init__(self,  **parameters):
        self.fder = 1
        self.Formula = HallPositionalShiftFormula
        super().__init__(**parameters)
