from hall_ps.hall_positional_shift import HallPositionalShift
from hall_ps.models import model_TaRh2B2
from wannierberri.system import System_PythTB
from wannierberri.grid import Grid
from wannierberri import run
from wannierberri.calculators.static import Hall_classic_FermiSurf, AHC_Zeeman_spin, AHC_Zeeman_orb 


import numpy as np
model = model_TaRh2B2(t1=1, t2=0.5, t3=0.3)
system = System_PythTB(model, spin=True)

efermi = np.linspace(-2, 2, 41)

tetra = True

calculators = {
    "hall_positional_shift_morb":
        HallPositionalShift(Efermi=efermi, tetra=tetra,
                            kwargs_formula={"morb_part": True},),
    # "hall_positional_shift_spin":
    #     HallPositionalShift(Efermi=efermi,
    #                         kwargs_formula={"spin_part": True, "tetra":tetra},),

    "hall_positional_shift_metric":
        HallPositionalShift(Efermi=efermi, tetra=tetra,
                            kwargs_formula={"metric_part": True},),

    "hall_classical": HallPositionalShift(Efermi=efermi, tetra=tetra, 
                                           kwargs_formula={"metric_part": True},),

    "hall_MOmega": AHC_Zeeman_orb(Efermi=efermi, tetra=tetra),
}

grid = Grid(system=system, length=30, length_FFT=15)

run(system=system,
    grid=grid,
    calculators=calculators,
    adpt_num_iter=0,
    fout_name='hall_positional_shift',
    )
