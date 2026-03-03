from hall_positional_shift import HallPositionalShift
from wannierberri.models import Chiral_OSD
from wannierberri.system import System_PythTB
from wannierberri.grid import Grid
from wannierberri import run
import numpy as np
model = Chiral_OSD()
system = System_PythTB(model, spin=True)

efermi = np.linspace(-2, 2, 41)


calculators = {
    "hall_positional_shift_morb":
        HallPositionalShift(Efermi=efermi,
                            kwargs_formula={"morb_part": True}),
    "hall_positional_shift_spin":
        HallPositionalShift(Efermi=efermi,
                            kwargs_formula={"spin_part": True}),

    "hall_positional_shift_metric":
        HallPositionalShift(Efermi=efermi,
                            kwargs_formula={"metric_part": True}),
}

grid = Grid(system=system, length=30, length_FFT=15)

run(system=system,
    grid=grid,
    calculators=calculators,
    adpt_num_iter=0,
    fout_name='hall_positional_shift',
    )
