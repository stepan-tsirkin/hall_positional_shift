import pythtb
import numpy as np
from wannierberri.system import System_PythTB
from wannierberri.grid import Path
from wannierberri.evaluate_k import evaluate_k_path
from matplotlib import pyplot as plt



def model_TaRh2B2(t1=1, t2=0.5, t3=0.3):
    c=1.123

    lat = [[1.0, 0.0, 0], [-0.5, np.sqrt(3.0) / 2.0, 0], [0, 0, c]]
    orb = [[ -1/3, -1/3,0], [1/3, 0, 1/3],[0, 1/3, 2/3]]
    lattice = pythtb.Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1, 2])
    my_model = pythtb.TBModel(lattice)


    my_model.set_onsite([0,0,0])
    my_model.set_hop(t1, 0, 1, [0, 0, 0])
    my_model.set_hop(t2, 0, 1, [-1, -1, 0])
    my_model.set_hop(t3, 0, 1, [-1, 0, 0])
    my_model.set_hop(t1, 1, 2, [0, 0, 0])
    my_model.set_hop(t2, 1, 2, [1, 0, 0])
    my_model.set_hop(t3, 1, 2, [0, -1, 0])
    my_model.set_hop(t1, 2, 0, [0, 0, 1])
    my_model.set_hop(t2, 2, 0, [0, 1, 1])
    my_model.set_hop(t3, 2, 0, [1, 1, 1])
    
    return my_model



if __name__ == "__main__":
    t1 = 1
    t2 = 0.5
    t3 = 0.3


    system = System_PythTB(ptb_model=model_TaRh2B2(t1=t1, t2=t2, t3=t3),)

    points = {"Gamma": [0, 0, 0], "M": [0.5, 0, 0], "K": [1/3, 1/3, 0],
            "A": [0, 0, 0.5], "L": [0.5, 0, 0.5], "H": [1/3, 1/3, 0.5]}

    path_list = ["Gamma", "M", "K", "Gamma", "A", "L", "H", "A"]
    nodes = [points[l] for l in path_list]
    print(f"{nodes=}\n{path_list=}")

    path = Path.from_nodes(system=system, 
                        nodes=nodes,
                        labels=path_list,
                        dk=0.05)


    bands = evaluate_k_path(system, path=path)

    from wannierberri.result.tabresult import TABresult

    bands.plot_path_fat(path,
                        close_fig=False,
                        show_fig=False
                        )
    
    plt.title(f"{t1=}, {t2=}, {t3=}")
    plt.savefig(f"TaRh2B2_t1_{t1}_t2_{t2}_t3_{t3}.png", dpi=300)
                        